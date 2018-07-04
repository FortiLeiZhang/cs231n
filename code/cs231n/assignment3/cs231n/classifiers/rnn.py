from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128, cell_type='rnn', dtype=np.float32):
        if cell_type not in ['rnn', 'lstm']:
            raise ValueError('Invalid cell_type "%s"' % cell_type)
        
        self.word_to_idx = word_to_idx
        self.input_dim = input_dim
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.dtype = dtype
        self.params = {}
        
        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx['<NULL>']
        
        self.params['W_feature'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b_feature'] = np.zeros(hidden_dim, )
        
        self.params['Wx'] = np.random.randn(wordvec_dim, hidden_dim) / np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.params['bh'] = np.zeros(hidden_dim, )
        self.params['Ws'] = np.random.randn(hidden_dim, vocab_size) / np.sqrt(hidden_dim)
        self.params['bs'] = np.zeros(vocab_size, )
        
        self.params['W_word'] = np.random.randn(vocab_size, wordvec_dim) / 100
        
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        loss = 0.0
        grads = {}
        
        caption_in = captions[:, :-1]
        caption_out = captions[:, 1:]
        mask = (caption_out != self._null)
        
        W_feature, b_feature = self.params['W_feature'], self.params['b_feature']
        h0, cache = affine_forward(features, W_feature, b_feature)
        
        W_word = self.params['W_word']
        x, cache = word_embedding_forward(caption_in, W_word)

        Wx, Wh, bh = self.params['Wx'], self.params['Wh'], self.params['bh']
        Ws, bs = self.params['Ws'], self.params['bs']
        h, cache = rnn_forward(x, h0, Wx, Wh, bh)
        out, cache = temporal_affine_forward(h, Ws, bs)
        loss, dx = temporal_softmax_loss(out, caption_out, mask)

        return loss, grads

    def sample(self, features, max_length=30):
        pass