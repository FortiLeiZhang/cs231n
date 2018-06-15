from __future__ import print_function
import numpy as np
try:
    from cs231n.im2col_cython import col2im_cython, im2col_cython
    from cs231n.im2col_cython import col2im_6d_cython
except ImportError:
    print('run the following from the cs231n directory and try again:')
    print('python setup.py build_ext --inplace')
    print('You may also need to restart your iPython kernel')

from cs231n.im2col import *

def conv_forward_im2col(x, w, b, conv_param):
    pass

def conv_forward_strides(x, w, b, conv_param):
    pass

def conv_backward_strides(dout, cache):
    pass

def conv_backward_im2col(dout, cache):
    pass

def max_pool_forward_fast(x, pool_param):
    pass

def max_pool_backward_fast(dout, cache):
    pass

def max_pool_forward_reshape(x, pool_param):
    pass

def max_pool_backward_reshape(dout, cache):
    pass

def max_pool_forward_im2col(x, pool_param):
    pass

def max_pool_backward_im2col(dout, cache):
    pass
    
    
    
    
    
    
    