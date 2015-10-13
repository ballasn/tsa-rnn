import sys, operator, logging
import itertools as it
import numbers

from collections import OrderedDict
from cStringIO import StringIO
import numpy as np

import theano.tensor.basic
import theano.sandbox.cuda.blas
import theano.printing
from theano.scan_module.scan_utils import equal_computations
import theano.tensor as T

from blocks.filter import VariableFilter

logger = logging.getLogger(__name__)

def broadcast_index(index, axes, ndim):
    dimshuffle_args = ['x'] * ndim
    if isinstance(axes, numbers.Integral):
        axes = [axes]
    for i, axis in enumerate(axes):
        dimshuffle_args[axis] = i
    return index.dimshuffle(*dimshuffle_args)

def broadcast_indices(index_specs, ndim):
    indices = []
    for index, axes in index_specs:
        indices.append(broadcast_index(index, axes, ndim))
    return indices

def subtensor(x, index_specs):
    indices = broadcast_indices(index_specs, x.ndim)
    return x[tuple(indices)]

# to handle non-unique monitoring channels without crashing and
# without silent loss of information
class Channels(object):
    def __init__(self):
        self.dikt = OrderedDict()

    def append(self, quantity, name=None):
        if name is not None:
            quantity = quantity.copy(name=name)
        self.dikt.setdefault(quantity.name, []).append(quantity)

    def extend(self, quantities):
        for quantity in quantities:
            self.append(quantity)

    def get_channels(self):
        channels = []
        for _, quantities in self.dikt.items():
            if len(quantities) == 1:
                channels.append(quantities[0])
            else:
                # name not unique; uniquefy
                for i, quantity in enumerate(quantities):
                    channels.append(quantity.copy(name="%s[%i]"
                                                  % (quantity.name, i)))
        return channels

# from http://stackoverflow.com/a/16571630
class StdoutLines(list):
    def __enter__(self):
        self._stringio = StringIO()
        self._stdout = sys.stdout
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def batched_tensordot(a, b, axes=2):
    return theano.tensor.basic._tensordot_as_dot(
        a, b, axes,
        dot=theano.sandbox.cuda.blas.batched_dot,
        batched=True)

def dedup(xs, equal=operator.is_):
    ys = []
    for x in xs:
        if not any(equal(x, y) for y in ys):
            ys.append(x)
    return ys

def get_recurrent_auxiliaries(names, graph, n_steps=None, require_in_graph=False):
    if require_in_graph:
        # ComputationGraph.auxiliary_variables includes auxiliaries
        # that may no longer be in the theano graph (except through
        # annotations). use `require_in_graph` to filter them out.
        all_variables = set(theano.gof.graph.ancestors(graph.outputs))

    variables = []
    for name in names:
        steps = VariableFilter(name=name)(graph.auxiliary_variables)
        if require_in_graph:
            steps = [step for step in steps if step in all_variables]
        steps = dedup(steps, equal=lambda a, b: equal_computations([a], [b]))

        if n_steps is not None:
            assert len(steps) == n_steps

        # a super crude sanity check to ensure these auxiliaries are
        # actually in chronological order
        assert all(_a < _b for _a, _b in 
                   (lambda _xs: zip(_xs, _xs[1:]))
                   ([len(theano.printing.debugprint(step, file="str"))
                     for step in steps]))

        variable = T.stack(steps)
        # move batch axis before rnn time axis
        variable = variable.dimshuffle(1, 0, *range(2, variable.ndim))
        variables.append(variable)
    return variables

from blocks.bricks.base import Brick, ApplicationCall

# attempt to fully qualify an annotated variable
def get_path(x):
    if isinstance(x, (T.TensorVariable,
                      # zzzzzzzzzzzzzzzzzzzzzzzzzzz
                      T.sharedvar.TensorSharedVariable,
                      T.compile.sharedvalue.SharedVariable)):
        paths = list(set(map(get_path, x.tag.annotations)))
        name = getattr(x.tag, "name", x.name)
        if len(paths) > 1:
            logger.warning(
                "get_path: variable %s has multiple possible origins, using first of [%s]"
                % (name, " ".join(paths)))
        return paths[0] + "/" + name
    elif isinstance(x, Brick):
        if x.parents:
            paths = list(set(map(get_path, x.parents)))
            if len(paths) > 1:
                logger.warning(
                    "get_path: brick %s has multiple parents, using first of [%s]"
                    % (x.name, " ".join(paths)))
            return paths[0] + "/" + x.name
        else:
            return "/" + x.name
    elif isinstance(x, ApplicationCall):
        return get_path(x.application.brick)
    else:
        raise TypeError()

# from theano.scan_module.scan_utils import map_variables

# def tag_for_replacement(variable, replacement, key):
#     # copy `variable` so that only the copy has the tag and any
#     # appearances of `variable` in `replacement` do not (or
#     # replacement will not terminate)
#     variable = variable.copy()
#     if not hasattr(variable.tag, "replacements"):
#         variable.tag.replacements = dict()
#     assert key not in variable.tag.replacements
#     variable.tag.replacements[key] = replacement
#     # return a copy so that tags put onto it by the caller are
#     # not lost after replacement
#     return variable.copy()

# def replace_by_tags(variables, keys):
#     def maybe_replace_one_once(variable):
#         for key in keys:
#             if key in variable.tag.replacements:
#                 replacement = variable.tag.replacements[key]
#                 logger.warning("replace_by_tags: %s: %s -> %s"
#                                % (key, variable, replacement))
#                 return replacement
#         return variable
#     def maybe_replace_one(variable):
#         while hasattr(variable.tag, "replacements"):
#             replacement = maybe_replace_one_once(variable)
#             if replacement is variable:
#                 break
#             variable = replacement
#         assert (not hasattr(variable.tag, "replacements") or
#                 not any(key in variable.tag.replacements
#                         for key in keys))
#         return variable
#     return map_variables(maybe_replace_one, variables)

# decorator to improve python's terrible argument error reporting
def checkargs(f):
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TypeError as e:
            type, value, traceback = sys.exc_info()
            if "takes exactly" in e.message:
                import inspect
                argspec = inspect.getargspec(f)
                required_args = argspec.args
                if argspec.defaults:
                    required_args = required_args[:-len(argspec.defaults)]
                required_args = [arg for arg in required_args
                                 if arg not in kwargs]
                missing_args = required_args[len(args):]
                if missing_args:
                    # reraise with more informative message
                    message = ("%s. not given: %s" %
                               (e.message, ", ".join(missing_args)))
                    raise type, (message,), traceback
            raise
    return g
