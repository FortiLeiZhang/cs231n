import numpy as np

def sgd(w, dw, config=None):
    if config is None:
        config = {}
        
    config.setdefault('learning_rate', 1e-3)
    learning_rate = config['learning_rate']
    
    next_w = w - learning_rate * dw
    
    return next_w, config

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
        
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('velocity', 0.0)
    config.setdefault('mu', 0.9)
    
    learning_rate = config['learning_rate']
    velocity = config['velocity']
    mu = config['mu']
    
    velocity = mu * velocity - learning_rate * dw
    
    w += velocity
    config['velocity'] = velocity
    return w, config

def rmsprop(w, dw, config=None):
    pass

def adam(w, dw, config=None):
    pass



