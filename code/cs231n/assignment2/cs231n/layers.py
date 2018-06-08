from builtins import range
import numpy as np

def affine_forward(x, w, b):  
    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b
    
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    (x, w, b) = cache
    N = x.shape[0]
    
    dx = dout.dot(w.T).reshape(*x.shape)
    dw = x.reshape(N, -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db