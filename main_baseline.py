import yaml
import os
import logging

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp, Adam, CompositeRule, StepClipping
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.roles import OUTPUT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.theano_expressions import l2_norm

import bricks
import initialization

import masonry
import conv3d
import crop
import util
from patchmonitor import PatchMonitoring, VideoPatchMonitoring

import dump
import tasks
import masonry


floatX = theano.config.floatX


def construct_model(hyperparameters, **kwargs):
    return masonry.construct_cnn(**hyperparameters)

def construct_monitors(algorithm, task_patches, x, x_shape,
                       graph, name, ram, model, cost,
                       n_spatial_dims, plot_url, patchmonitor_interval=100, **kwargs):
    channels = util.Channels()
    channels.extend(task.monitor_channels(graph))
    channels.append(algorithm.total_gradient_norm,
                    "total_gradient_norm")

    step_norms = util.Channels()
    step_norms.extend(util.named(l2_norm([algorithm.steps[param]]),
                                 "%s.step_norm" % name)
                      for name, param in model.get_parameter_dict().items())
    step_channels = step_norms.get_channels()

    #for activation in VariableFilter(roles=[OUTPUT])(graph.variables):
    #    quantity = activation.mean()
    #    quantity.name = "%s.mean" % util.get_path(activation)
    #    channels.append(quantity)

    data_independent_channels = util.Channels()
    for parameter in graph.parameters:
        if parameter.name in "gamma beta".split():
            quantity = parameter.mean()
            quantity.name = "%s.mean" % util.get_path(parameter)
            data_independent_channels.append(quantity)

    extensions = []
    extensions.append(TrainingDataMonitoring(
        step_channels, prefix="train", after_epoch=True))

    extensions.append(DataStreamMonitoring(data_independent_channels.get_channels(),
                                           data_stream=None, after_epoch=True))
    extensions.extend(DataStreamMonitoring((channels.get_channels() + [cost]),
                                           data_stream=task.get_stream(which, monitor=True,
                                                                       crop_lenght=input_shape[0]),
                                           prefix=which, after_epoch=True)
                      for which in "train valid test".split())
    return extensions

def construct_main_loop(name, task_name, input_shape,
                        batch_size, n_epochs,
                        learning_rate, hyperparameters, **kwargs):
    name = "%s_%s" % (name, task_name)
    hyperparameters["name"] = name

    task = tasks.get_task(**hyperparameters)
    hyperparameters["n_channels"] = task.n_channels

    theano.config.compute_test_value = "warn"

    x, x_shape, y = task.get_variables()

    model = construct_model(task=task, **hyperparameters)
    model.initialize()
    feat = model.apply(x)
    dim = np.prod(model.get_dim('output'))

    emitter = task.get_emitter(
        input_dim=dim,
        **hyperparameters)
    emitter.initialize()
    cost = emitter.cost(feat, y, 1)
    cost.name = "cost"

    print "setting up main loop..."
    graph = ComputationGraph(cost)
    uselessflunky = Model(cost)
    algorithm = GradientDescent(
        cost=cost,
        parameters=graph.parameters,
        step_rule=CompositeRule([StepClipping(1.),
                                 Adam(learning_rate=learning_rate)]))
    monitors = construct_monitors(
        x=x, x_shape=x_shape, y=y, cost=cost,
        algorithm=algorithm, task=task, model=uselessflunky, ram=model,
        graph=graph, **hyperparameters)
    main_loop = MainLoop(data_stream=task.get_stream("train", crop_lenght=input_shape[0]),
                         algorithm=algorithm,
                         extensions=(monitors +
                                     [FinishAfter(after_n_epochs=n_epochs),
                                      dump.DumpBest(name+'_best', channel_name='valid_error_rate'),
                                      dump.LightCheckpoint(name+"_checkpoint.zip", on_interrupt=False),
                                      #Checkpoint(name+'_checkpoint.pkl', every_n_epochs=10, on_interrupt=False),
                                      ProgressBar(),
                                      Timing(),
                                      Printing(),
                                      PrintingTo(name+"_log")]),
                         model=uselessflunky)
    return main_loop

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameters", help="YAML file from which to load hyperparameters")
    parser.add_argument("--parameters", help="npy/npz file from which to load parameters")

    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "defaults_baseline.yaml"), "rb") as f:
        hyperparameters = yaml.load(f)
    if args.hyperparameters:
        with open(args.hyperparameters, "rb") as f:
            hyperparameters.update(yaml.load(f))

    hyperparameters["hyperparameters"] = hyperparameters

    main_loop = construct_main_loop(**hyperparameters)

    if args.parameters:
        load_model_parameters(args.parameters, main_loop.model)

    print "training..."
    main_loop.run()
