from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ThreeLayerConvNet(object):
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                dtype=np.float64):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        C, H, W = self.input_dim
        F = self.num_filters
        HH, WW = self.filter_size, self.filter_size
        
        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params['b1'] = np.zeros(F, )
        self.params['W2'] = weight_scale * np.random.randn(np.prod((F, H//2, W//2)), self.hidden_dim)
        self.params['b2'] = np.zeros(self.hidden_dim, )
        self.params['W3'] = weight_scale * np.random.randn(self.hidden_dim, self.num_classes)
        self.params['b3'] = np.zeros(self.num_classes, )
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def loss(self, X, y=None):
        loss, grads = 0.0, {}
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        C, H, W = self.input_dim
        
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) // 2}
        pool_param = {'stride': 2, 'pool_width': 2, 'pool_height': 2}
        
        layer1_out, layer1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        layer2_out, layer2_cache = affine_relu_forward(layer1_out, W2, b2)
        layer3_out, layer3_cache = affine_forward(layer2_out, W3, b3)
        
        if y is None:
            return layer3_out
        
        loss, dout = softmax_loss(layer3_out, y)
        loss += 0.5 * (self.reg * np.sum(W1 * W1) + self.reg * np.sum(W2 * W2) + self.reg * np.sum(W3 * W3))
        
        dout, dW3, db3 = affine_backward(dout, layer3_cache)
        dout, dW2, db2 = affine_relu_backward(dout, layer2_cache)
        dx, dW1, db1 = conv_relu_pool_backward(dout, layer1_cache)
        
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3
        
        return loss, grads