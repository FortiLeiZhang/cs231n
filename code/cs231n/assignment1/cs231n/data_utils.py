from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' %b)
        X, y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y
    
def load_pickle(filename):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(filename)
    elif version[0] == '3':
        return pickle.load(filename, encoding='latin1')
    raise ValueError('Invalid python version: {}'.format(version))
        
    
    
    
    
    
    
    