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
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        
        dim_expand = {'rnn': 1, 'lstm': 4}[self.cell_type]
        
        self.params['W_feature'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b_feature'] = np.zeros(hidden_dim, )
        
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_expand * hidden_dim) / np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_expand * hidden_dim) / np.sqrt(hidden_dim)
        self.params['bh'] = np.zeros(dim_expand * hidden_dim, )
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
        h0, cache_feature = affine_forward(features, W_feature, b_feature)
        
        W_word = self.params['W_word']
        x, cache_word = word_embedding_forward(caption_in, W_word)

        Wx, Wh, bh = self.params['Wx'], self.params['Wh'], self.params['bh']
        if self.cell_type == 'rnn':
            h, cache = rnn_forward(x, h0, Wx, Wh, bh)
        elif self.cell_type == 'lstm':
            h, cache = lstm_forward(x, h0, Wx, Wh, bh)
        
        Ws, bs = self.params['Ws'], self.params['bs']
        out, cache_score = temporal_affine_forward(h, Ws, bs)

        loss, dscore = temporal_softmax_loss(out, caption_out, mask)
        
        dLossdh, dWs, dbs = temporal_affine_backward(dscore, cache_score)
        grads['Ws'] = dWs
        grads['bs'] = dbs
        
        if self.cell_type == 'rnn':
            dx, dh0, dWx, dWh, dbh = rnn_backward(dLossdh, cache)
        elif self.cell_type == 'lstm':
            dx, dh0, dWx, dWh, dbh = lstm_backward(dLossdh, cache)
        
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['bh'] = dbh
        
        dW_word = word_embedding_backward(dx, cache_word)
        grads['W_word'] = dW_word
        
        _, dW_feature, db_feature = affine_backward(dh0, cache_feature)
        grads['W_feature'] = dW_feature
        grads['b_feature'] = db_feature

        return loss, grads

    def sample(self, features, max_length=30):
        W_feature, b_feature = self.params['W_feature'], self.params['b_feature']
        W_word = self.params['W_word']
        Wx, Wh, bh = self.params['Wx'], self.params['Wh'], self.params['bh']
        Ws, bs = self.params['Ws'], self.params['bs']
        
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)
        captions[:, 0] = self._start

        h0, _ = affine_forward(features, W_feature, b_feature)
        x0, _ = word_embedding_forward(captions[:, 0], W_word)
        input_word = x0
        prev_h = h0
        
        if self.cell_type == 'lstm':
            prev_c = np.zeros_like(h0)
        
        for i in range(1, max_length):
            if self.cell_type == 'rnn':
                prev_h, _ = rnn_step_forward(input_word, prev_h, Wx, Wh, bh)
            elif self.cell_type == 'lstm':
                prev_h, prev_c, _ = lstm_step_forward(input_word, prev_h, prev_c, Wx, Wh, bh)

            score, _ = affine_forward(prev_h, Ws, bs)
            captions[:, i] = np.argmax(score, axis=1)

            input_word, _ = word_embedding_forward(captions[:, i], W_word)
          
        return captions    