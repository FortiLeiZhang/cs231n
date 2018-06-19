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
    (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps) = cache
    N = dout.shape[0]
    dx, dgamma, dbeta = None, None, None
    
    dx = gamma * inv_x_std / N * (N * dout - np.sum(dout, axis=0) - inv_x_std ** 2 * x_mean_0 * np.sum(dout * x_mean_0, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    return dx, dgamma, dbeta    

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape

    _x = x.transpose((0, 2, 3, 1))
    _x = _x.reshape(-1, C) 
    
    _out, cache = batchnorm_forward(_x, gamma, beta, bn_param)
    
    _out = _out.reshape((N, H, W, C))
    out = _out.transpose((0, 3, 1, 2))

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape

    _dout = dout.transpose((0, 2, 3, 1))
    _dout = _dout.reshape(-1, C) 
    
    _dx, dgamma, dbeta = batchnorm_backward(_dout, cache)
    
    _dx = _dx.reshape((N, H, W, C))
    dx = _dx.transpose((0, 3, 1, 2))

    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param=None):
    if ln_param is None:
        ln_param = {}

    x = np.transpose(x, (1, 0))
    N = x.shape[0]
    out, cache = None, None
    
    eps = ln_param.get('eps', 1e-8)
    
    x_mean = 1 / N * np.sum(x, axis=0)
    x_mean_0 = x - x_mean
    x_mean_0_sqr = x_mean_0 ** 2
    x_var = 1 / N * np.sum(x_mean_0_sqr, axis=0)
    x_std = np.sqrt(x_var + eps)
    inv_x_std = 1 / x_std
    x_hat = x_mean_0 * inv_x_std
    
    out = gamma * np.transpose(x_hat, (1, 0)) + beta
    cache = (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps)
    
    return out, cache

def layernorm_backward(dout, cache):
    (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps) = cache
    
    dx, dgamma, dbeta = None, None, None
    
    # out = gamma * x_hat + beta
    # (N,D) (D,)    (N,D)   (D,)
    Dx_hat = dout * gamma
    Dx_hat = np.transpose(Dx_hat, (1, 0))
    N = Dx_hat.shape[0]
    
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
    
    dx = np.transpose(Dx, (1, 0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * np.transpose(x_hat, (1, 0)), axis=0)
    
    return dx, dgamma, dbeta

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    if gn_param is None:
        gn_param = {}
    eps = gn_param.get('eps', 1e-8)
    
    N, C, H, W = x.shape
    _x = x.reshape((N, G, C//G, H, W))
    
    x_mean = 1 / (C//G*H*W) * np.sum(_x, axis=(2, 3, 4), keepdims=True)
    x_mean_0 = _x - x_mean
    x_mean_0_sqr = x_mean_0 ** 2
    x_var = 1 / (C//G*H*W) * np.sum(x_mean_0_sqr, axis=(2, 3, 4), keepdims=True)
    x_std = np.sqrt(x_var + eps)
    inv_x_std = 1 / x_std
    x_hat = x_mean_0 * inv_x_std
    
    x_hat = x_hat.reshape((N, C, H, W))
    out = gamma * x_hat + beta
    
    cache = (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps, G)
    
    return out, cache

def spatial_groupnorm_backward(dout, cache):
    (x_mean, x_mean_0, x_mean_0_sqr, x_var, x_std, inv_x_std, x_hat, gamma, eps, G) = cache
    N, C, H, W = dout.shape

    Dx_hat = dout * gamma
    Dx_hat = Dx_hat.reshape((N, G, C//G, H, W))

    Dx_mean_0 = Dx_hat * (inv_x_std)
    Dinv_x_std = np.sum(Dx_hat * (x_mean_0), axis=(2, 3, 4), keepdims=True)
    Dx_std = Dinv_x_std * (- x_std ** (-2))
    Dx_var = Dx_std * (0.5 * (x_var + eps) ** (-0.5))
    Dx_mean_0_sqr = Dx_var * (1 / (C//G*H*W) * np.ones_like(x_mean_0_sqr))
    Dx_mean_0 += Dx_mean_0_sqr * (2 * x_mean_0)
    Dx = Dx_mean_0 * (1)
    Dx_mean = np.sum(Dx_mean_0 * (-1), axis=(2, 3, 4), keepdims=True)
    Dx += Dx_mean * (1 / (C//G*H*W) * np.ones_like(Dx_hat))

    dx = Dx.reshape((N, C, H, W))
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
    
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    out, cache, mask = None, None, None
    
    p = dropout_param['p']
    mode = dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        cache = (mask, dropout_param)
    elif mode == 'test':
        out = x
    else:
        raise ValueError('Unrecognized mode: %s' %mode)
        
    cache = (mask, mode)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_backward(dout, cache):
    (mask, mode) = cache
    
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    else:
        raise ValueError('Unrecognized mode: %s' %mode)    
    
    return dx

def conv_forward_naive(x, w, b, conv_param):
    s = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    
    N, in_c, in_h, in_w = x.shape
    K, in_c, f_h, f_w = w.shape
    K = b.shape[0]
    
    assert (in_h - f_h + 2 * pad) % s == 0, 'Filter height mismatch.' 
    assert (in_w - f_w + 2 * pad) % s == 0, 'Filter width mismatch.'
    
    out_h = np.uint8((in_h - f_h + 2 * pad) / s + 1)
    out_w = np.uint8((in_w - f_w + 2 * pad) / s + 1)
    out = np.zeros((N, K, out_h, out_w))
    cache = None
    
    pad_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    for i in range(N):
        for j in range(K):
            for ww in range(out_w):
                for hh in range(out_h):
                    out[i, j, hh, ww] = np.sum(pad_x[i, :, (s*hh):(s*hh+f_h), (s*ww):(s*ww+f_w)] * w[j, :, :, :]) + b[j]

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    (x, w, b, conv_param) = cache
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = conv_param['stride']
    p = conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
    dx_pad = np.zeros_like(x_pad)

    N, in_c, in_h, in_w = x.shape
    N, K, out_h, out_w = dout.shape
    K, in_c, f_h, f_w = w.shape

    for i in range(N):
        for oc in range(K):
            for ww in range(out_w):
                for hh in range(out_h):
                    dx_pad[i, :, (s*hh):(s*hh+f_h), (s*ww):(s*ww+f_w)] += dout[i, oc, hh, ww] * w[oc, ...]
                    dw[oc, ...] += dout[i, oc, hh, ww] * x_pad[i, :, (s*hh):(s*hh+f_h), (s*ww):(s*ww+f_w)]

    dx = dx_pad[:, :, p:(in_h+p), p:(in_w+p)]
    db = np.sum(dout, axis=(0, 2, 3))
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    s = pool_param['stride']
    
    N, C, in_h, in_w = x.shape
    
    assert (in_h - pool_height) % s == 0, 'Pool height mismatch.' 
    assert (in_w - pool_width) % s == 0, 'Pool width mismatch.'
    
    out_h = np.uint8((in_h - pool_height) / s + 1)
    out_w = np.uint8((in_w - pool_width) / s + 1)
    
    out = np.zeros((N, C, out_h, out_w))
    cache = None
    
    for ww in range(out_w):
        for hh in range(out_h):
            out[:, :, hh, ww] = np.max(x[:, :, s*hh:s*hh+pool_height, s*ww:s*ww+pool_width], axis=(2, 3))
    
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    (x, pool_param) = cache
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    s = pool_param['stride']
    
    _, _, out_h, out_w = dout.shape
    dx = np.zeros_like(x)
    
    for ww in range(out_w):
        for hh in range(out_h):
            _max = np.max(x[:, :, s*hh:s*hh+pool_height, s*ww:s*ww+pool_width], axis=(2, 3), keepdims=True)
            dx_idx = (x[:, :, s*hh:s*hh+pool_height, s*ww:s*ww+pool_width] == _max)
            dout_block = dout[:, :, hh, ww]
            dx[:, :, s*hh:s*hh+pool_height, s*ww:s*ww+pool_width] = dx_idx * dout_slice[:, :, np.newaxis, np.newaxis]
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    

