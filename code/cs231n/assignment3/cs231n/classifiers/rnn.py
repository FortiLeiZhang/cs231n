from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128, cell_type='rnn', dtype=np.float32):
        pass

    def loss(self, features, captions):
        pass

    def sample(self, features, max_length=30):
        pass