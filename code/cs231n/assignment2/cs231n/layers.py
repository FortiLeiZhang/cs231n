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

def relu_forward(x):
    out = np.maximum(0, x)
    cache = (x)
    return out, cache
    
def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx

def svm_loss(x, y):
    N = x.shape[0]
    
    correct_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] = -np.sum(dx, axis=1)
    dx = dx / N
    
    return loss, dx

def softmax_loss(x, y):
    N = x.shape[0]
    
    stable_x = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(stable_x)
    correct_exp_scores = exp_scores[np.arange(N), y]
    loss = -np.log(correct_exp_scores/np.sum(exp_scores, axis=1))
    loss = np.sum(loss) / N
    
    dx = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    dx[np.arange(N), y] -= 1
    dx = dx / N
    
    return loss, dx
    
    
    
    
    
    
    
    
    
    
    
    
    
    

