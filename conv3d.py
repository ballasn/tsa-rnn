import operator
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

from blocks.bricks.base import application, Brick
from blocks.bricks import Initializable

import bricks
import initialization

import masonry


class Conv3d(bricks.Initializable):
    def __init__(self, input_shape,
                 name, hyperparameters, **kwargs):
        super(Conv3d, self).__init__(**kwargs)
        self.name = name
        self.cnn = masonry.construct_cnn(self.name, hyperparameters)

    @application
    def apply(self, x):
        return cnn.apply(x)
