from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet():
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
    
    def loss(self, X, y=None):
        grads = {}
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        grads['W1'] = np.zeros_like(W1)
        grads['b1'] = np.zeros_like(b1)
        grads['W2'] = np.zeros_like(W2)
        grads['b2'] = np.zeros_like(b2)
        
        relu_out, cache_relu_out = affine_relu_forward(X, W1, b1)
        scores, cache_out = affine_forward(relu_out, W2, b2)
        
        if y is None:
            return scores
        
        loss, Dscores = softmax_loss(scores, y)
        dx, grads['W2'], grads['b2'] = affine_backward(Dscores, cache_out)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dx, cache_relu_out)
        
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        grads['W2'] = grads['W2'] + self.reg * W2
        grads['W1'] = grads['W1'] + self.reg * W1
        return loss, grads
        
        
        
        
        
        
        
        
        
        
        