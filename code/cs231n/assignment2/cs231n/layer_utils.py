from cs231n.layers import *
from cs231n.fast_layers import *

def affine_relu_forward(x, w, b):
    affine_out, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(affine_out)
    cache = (affine_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    affine_cache, relu_cache = cache
    daffine = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(daffine, affine_cache)
    return dx, dw, db

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param, norm_type):
    aff_out, aff_cache = affine_forward(x, w, b)
    
    if norm_type == 'batch_norm':
        norm_out, norm_cache = batchnorm_forward(aff_out, gamma, beta, bn_param)
    if norm_type == 'layer_norm':
        norm_out, norm_cache = layernorm_forward(aff_out, gamma, beta, bn_param)

    out, relu_cache = relu_forward(norm_out)
    cache = (aff_cache, norm_cache, relu_cache)
    return out, cache

def affine_norm_relu_backward(dout, cache, norm_type):
    (aff_cache, norm_cache, relu_cache) = cache
    Dnorm_out = relu_backward(dout, relu_cache)
    
    if norm_type == 'batch_norm':       
        Daff_out, Dgamma, Dbeta = batchnorm_backward(Dnorm_out, norm_cache)
    if norm_type == 'layer_norm':
        Daff_out, Dgamma, Dbeta = layernorm_backward(Dnorm_out, norm_cache)

    Dx, Dw, Db = affine_backward(Daff_out, aff_cache)
    return Dx, Dw, Db, Dgamma, Dbeta

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

