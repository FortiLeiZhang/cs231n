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

def batchnorm_forward(x, gamma, beta, bn_param):
    if bn_param is None:
        bn_param = {}

    N = x.shape[0]
    out, cache = None, None

    running_mean = bn_param.get('running_mean', np.zeros_like(beta))
    running_var = bn_param.get('running_var', np.zeros_like(gamma))
    mode = bn_param.get('mode', 'train')
    eps = bn_param.get('eps', 1e-8)
    momentum = bn_param.get('momentum', 0.9)
    
    if mode == 'train':
        x_mean = 1 / N * np.sum(x, axis=0)
        x_mean_0 = x - x_mean
        x_mean_0_sqr = x_mean_0 ** 2
        x_var = 1 / N * np.sum(x_mean_0_sqr, axis=0)
        x_std = np.sqrt(x_var + eps)
        inv_x_std = 1 / x_std
        x_hat = x_mean_0 * inv_x_std

        out = gamma * x_hat + beta
        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * x_mean
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * x_var
        
        cache = (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps)

    elif mode == 'test':
        x_hat = (x - running_mean) / (np.sqrt(running_var + eps))
        out = gamma * x_hat + beta
    
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache

def batchnorm_backward(dout, cache):
    (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps) = cache
    N = dout.shape[0]
    dx, dgamma, dbeta = None, None, None
    
    # out = gamma * x_hat + beta
    # (N,D) (D,)    (N,D)   (D,)
    Dx_hat = dout * gamma
    
    # x_hat = x_mean_0 * inv_x_std
    # (N,D)   (N,D)      (D,)
    Dx_mean_0 = Dx_hat * (inv_x_std)
    Dinv_x_std = np.sum(Dx_hat * (x_mean_0), axis=0)
    
    # inv_x_std = 1 / x_std
    # (D,)            (D,)
    Dx_std = Dinv_x_std * (- x_std ** (-2))
    
    # x_std = np.sqrt(x_var + eps)
    # (D,)           (D,)
    Dx_var = Dx_std * (0.5 * (x_var + eps) ** (-0.5))
    
    # x_var = 1 / N * np.sum(x_mean_0_sqr, axis=0)
    # (D,)                   (N,D)
    Dx_mean_0_sqr = Dx_var * (1 / N * np.ones_like(x_mean_0_sqr))
    
    # x_mean_0_sqr = x_mean_0 ** 2
    # (N,D)          (N,D)
    Dx_mean_0 += Dx_mean_0_sqr * (2 * x_mean_0)
    
    # x_mean_0 = x - x_mean
    # (N,D)     (N,D) (D,)
    Dx = Dx_mean_0 * (1)
    Dx_mean = np.sum(Dx_mean_0 * (-1), axis=0)
    
    # x_mean = 1 / N * np.sum(x, axis=0)
    # (D,)                   (N,D)
    Dx += Dx_mean * (1 / N * np.ones_like(x_hat))
    
    dx = Dx    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    pass

def layernorm_forward(x, gamma, beta, ln_param):
    pass

def layernorm_backward(dout, cache):
    pass

def dropout_forward(x, dropout_param):
    pass

def dropout_backward(dout, cache):
    pass

def conv_forward_naive(x, w, b, conv_param):
    pass

def conv_backward_naive(dout, cache):
    pass

def max_pool_forward_naive(x, pool_param):
    pass

def max_pool_backward_naive(dout, cache):
    pass

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    pass

def spatial_batchnorm_backward(dout, cache):
    pass

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    pass

def spatial_groupnorm_backward(dout, cache):
    pass

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    

