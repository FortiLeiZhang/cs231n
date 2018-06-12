from cs231n.layers import *
from cs231n.fast_layers import *

def affine_relu_forward(x, w, b):
    N = x.shape[0]
    
    aff_out = x.reshape(N, -1).dot(w) + b
    out = np.maximum(0, aff_out)
    cache = (x, w, b, aff_out)
    return out, cache

def affine_relu_backward(dout, cache):
    (x, w, b, aff_out) = cache
    N = x.shape[0]
    
    Daff_out = dout * (aff_out > 0)
    dx = Daff_out.dot(w.T).reshape(*x.shape)
    dw = x.reshape(N, -1).T.dot(Daff_out)
    db = np.sum(Daff_out, axis=0)
    return dx, dw, db

def conv_relu_forward(x, w, b, conv_param):
    pass

def conv_relu_backward(dout, cache):
    pass

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    pass

def conv_bn_relu_backward(dout, cache):
    pass

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    pass

def conv_relu_pool_backward(dout, cache):
    pass

