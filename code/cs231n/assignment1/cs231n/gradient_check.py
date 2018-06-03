from __future__ import print_function

import numpy as np
from random import randrange

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])
        
        old_val = x[ix]
        x[ix] = old_val + h
        fx_p = f(x)
        x[ix] = old_val - h
        fx_m = f(x)
        x[ix] = old_val
        
        grad_numerical = (fx_p - fx_m) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
        
def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + h
        fx_p = f(x)
        x[ix] = old_val - h
        fx_m = f(x)
        x[ix] = old_val
        grad[ix] = (fx_p - fx_m) / (2 * h)
        
        if verbose:
            print(ix, grad[ix])
        it.iternext()
        
    return grad
        
        
        
        
        
        
        
    