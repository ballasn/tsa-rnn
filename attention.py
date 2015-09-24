import operator
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

import theano
import theano.tensor as T

from blocks.bricks.base import application, Brick
from blocks.bricks import Initializable

import bricks
import initialization

import masonry

floatX = theano.config.floatX

# this belongs on RecurrentAttentionModel as a static method, but that breaks pickling
def static_map_to_input_space(location, scale, patch_shape, image_shape):
    # linearly map locations from (-1, 1) to image index space
    location = (location + 1) / 2 * image_shape
    # disallow negative scale
    scale *= scale > 0
    # translate scale such that scale = 0 corresponds to shrinking the
    # full image to fit into the patch, and the model can only zoom in
    # beyond that.  i.e. by default the model looks at a very coarse
    # version of the image, and can choose to selectively refine
    # regions
    scale += patch_shape / image_shape
    return location, scale

class RecurrentAttentionModel(bricks.BaseRecurrent):
    def __init__(self, rnn, cropper, emitter, embed_mlp,
                 attention_state_name, hyperparameters, **kwargs):
        super(RecurrentAttentionModel, self).__init__(**kwargs)

        self.cropper = cropper
        self.emitter = emitter
        self.rnn = rnn
        self.embed_mlp = embed_mlp

        self.construct_locator(**hyperparameters)
        self.construct_merger(output_dim=embed_mlp.input_dim,
                              **hyperparameters)

        # name of the RNN state that determines the parameters of the next glimpse
        self.attention_state_name = attention_state_name

        self.children.extend([self.rnn, self.cropper, self.emitter, self.embed_mlp])

        # states aren't known until now
        self.apply.outputs = self.rnn.apply.outputs
        self.compute_initial_state.outputs = self.rnn.apply.outputs

    def construct_merger(merge_mlp, output_dim, patch_transform,
                         batch_normalize, **kwargs):
        self.merge_mlp = merge_mlp
        self.patch_transform = patch_transform
        self.response_merge = bricks.Merge(
            input_names="area patch".split(),
            input_dims=[merge_mlp.brick.output_dim,
                        patch_transform.brick.output_dim],
            output_dim=output_dim,
            prototype=bricks.Linear(use_bias=False),
            child_prefix="response_merge")
        self.response_merge_activation = bricks.NormalizedActivation(
            shape=[output_dim],
            name="response_merge_activation",
            batch_normalize=batch_normalize)
        self.children.extend([
            self.response_merge_activation,
            self.response_merge,
            self.patch_transform.brick,
            self.merge_mlp.brick])

    def construct_locator(locate_mlp, n_spatial_dims,
                          location_std, scale_std, **kwargs):
        self.n_spatial_dims = n_spatial_dims
        self.locate_mlp = locate_mlp

        self.theta_from_area = bricks.Linear(
            input_dim=locate_mlp.brick.output_dim,
            output_dim=2*n_spatial_dims,
            name="theta_from_area")

        self.T_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(12345)
        self.location_std = location_std
        self.scale_std = scale_std

        self.children.extend([
            self.locate_mlp.brick,
            self.theta_from_area])

    def get_dim(self, name):
        try:
            return self.rnn.get_dim(name)
        except:
            return super(RecurrentAttentionModel, self).get_dim(name)

    @application
    def apply(self, x, **states):
        location, scale = self.locate(states[self.attention_state_name])
        patch = self.crop(x, location, scale)
        u = self.embed_mlp(self.merge(patch, location, scale))
        states = self.rnn.apply(inputs=u, iterate=False, as_dict=True, **states)
        return tuple(states.values())
        
    def locate(self, h):
        area = self.locate_mlp(h)
        theta = self.theta_from_area.apply(area)
        # going from area to theta is typically a huge reduction
        # in dimensionality; divide each output by the fan-in to avoid
        # large values
        theta /= self.theta_from_area.input_dim
        location, scale = (theta[:, :self.n_spatial_dims],
                           theta[:, self.n_spatial_dims:])
        location += self.T_rng.normal(location.shape, std=self.location_std)
        scale += self.T_rng.normal(scale.shape, std=self.scale_std)
        return location, scale

    def merge(self, patch, location, scale):
        patch = self.patch_transform(patch)
        area = self.merge_mlp(T.concatenate([location, scale], axis=1))
        parts = self.response_merge.apply(area, patch)
        return self.response_merge_activation.apply(response)

    @application
    def compute_initial_state(self, x):
        initial_states = self.rnn.initial_states(x.shape[0], as_dict=True)
        # condition on initial shrink-to-fit patch
        location = T.alloc(T.cast(0.0, floatX),
                           x.shape[0], self.cropper.n_spatial_dims)
        scale = T.zeros_like(location)
        patch = self.crop(x, location, scale)
        u = self.embed_mlp(self.merge(patch, location, scale))
        conditioned_states = self.rnn.apply(as_dict=True, inputs=u, iterate=False, **initial_states)
        return tuple(conditioned_states.values())

    def crop(self, x, location, scale):
        true_location, true_scale = self.map_to_input_space(location, scale)
        patch = self.cropper.apply(x, true_location, true_scale)
        self.add_auxiliary_variable(location, name="location")
        self.add_auxiliary_variable(scale, name="scale")
        self.add_auxiliary_variable(true_location, name="true_location")
        self.add_auxiliary_variable(true_scale, name="true_scale")
        self.add_auxiliary_variable(patch, name="patch")
        return patch

    def map_to_input_space(self, location, scale):
        return static_map_to_input_space(
            location, scale,
            T.cast(self.cropper.patch_shape, floatX),
            T.cast(self.cropper.image_shape, floatX))
