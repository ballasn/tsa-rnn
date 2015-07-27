import itertools as it
import numbers
from theano.compile import ViewOp
from collections import OrderedDict
from blocks.utils import named_copy
from blocks.initialization import NdarrayInitialization

import theano.tensor as T

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

class WithDifferentiableApproximation(ViewOp):
    __props__ = ()

    def make_node(self, fprop_output, bprop_output):
        # avoid theano wasting time computing the gradient of fprop_output
        fprop_output = theano.gradient.disconnected_grad(fprop_output)
        return gof.Apply(self, [fprop_output, bprop_output], [f.type()])

    def grad(self, wrt, input_gradients):
        import pdb; pdb.set_trace()
        # check that we need input_gradients[1] rather than input_gradients[:][1]
        return input_gradients[1]

def with_differentiable_approximation(fprop_output, bprop_output):
    return WithDifferentiableApproximation()(fprop_output, bprop_output)

# to handle non-unique monitoring channels without crashing and
# without silent loss of information
class Channels(object):
    def __init__(self):
        self.dikt = OrderedDict()

    def append(self, quantity, name=None):
        if name is not None:
            quantity = named_copy(quantity, name)
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
                    channels.append(named_copy(
                        quantity, "%s[%i]" % (i, quantity.name)))
        return channels

# L1-normalize along an axis (default: normalize columns, which for
# Linear bricks ensures each input is scaled by at most 1)
class NormalizedInitialization(NdarrayInitialization):
    def __init__(self, initialization, axis=0, **kwargs):
        self.initialization = initialization
        self.axis = axis

    def generate(self, rng, shape):
        x = self.initialization.generate(rng, shape)
        x /= abs(x).sum(axis=self.axis, keepdims=True)
        return x

def dict_merge(*dikts):
    result = OrderedDict()
    for dikt in dikts:
        result.update(dikt)
    return result

def named(x, name):
    x.name = name
    return x

# from http://stackoverflow.com/a/16571630
from cStringIO import StringIO
import sys

class StdoutLines(list):
    def __enter__(self):
        self._stringio = StringIO()
        self._stdout = sys.stdout
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

# https://groups.google.com/d/msg/theano-users/rBw1oA-DKgE/CBXjw8uvQroJ
def batched_tensordot(x, y, axes):
    try:
        axes = list(axes)
    except TypeError:
        axes = [range(x.ndim - axes, y.ndim), range(1, axes + 1)]

    n_sum_axes = len(axes[0])
    x_dimshuffle = [0] + range(1, x.ndim) + (["x"] * (y.ndim - 1 - n_sum_axes))
    y_dimshuffle = [0] + (["x"] * (x.ndim - 1 - n_sum_axes)) + range(1, y.ndim)

    # align summation axes
    for x_sum_axis in axes[0]:
        x_dimshuffle.remove(x_sum_axis)
        x_dimshuffle.insert(0, x_sum_axis)
    for y_sum_axis in axes[1]:
        y_dimshuffle.remove(y_sum_axis)
        y_dimshuffle.insert(0, y_sum_axis)

    z = (x.dimshuffle(*x_dimshuffle) * y.dimshuffle(*y_dimshuffle))
    z = z.sum(axis=range(n_sum_axes))
    return z
