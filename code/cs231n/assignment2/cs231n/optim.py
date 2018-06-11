import numpy as np

def sgd(w, dw, config=None):
    if config is None:
        config = {}
        
    learning_rate = config.get('learning_rate', 1e-3)
    
    next_w = w - learning_rate * dw

    return next_w, config

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
        
    learning_rate = config.get('learning_rate', 1e-3)
    velocity = config.get('velocity', 0.0)
    mu = config.get('mu', 0.9)
    
    velocity = mu * velocity - learning_rate * dw
    
    next_w = w + velocity
    config['velocity'] = velocity
    return next_w, config

def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
        
    learning_rate = config.get('learning_rate', 1e-3)
    cache = config.get('cache', np.zeros_like(w))
    decay_rate = config.get('decay_rate', 0.99)
    eps = config.get('eps', 1e-8)
        
    cache = decay_rate * cache + (1 - decay_rate) * dw ** 2
    next_w = w -learning_rate * dw / (np.sqrt(cache) + eps)
    config['cache'] = cache
    
    return next_w, config

def adam(w, dw, config=None):
    if config is None:
        config = {}
        
    learning_rate = config.get('learning_rate', 1e-3)
    m = config.get('m', np.zeros_like(w))
    v = config.get('v', np.zeros_like(w))
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.999)
    t = config.get('t', np.uint32(1))
    eps = config.get('eps', 1e-8)
    
    t = t + 1
    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    vt = v / (1 - beta2 ** t)
    
    new_w = w - learning_rate * mt / (np.sqrt(vt) + eps)
    config['v'] = v
    config['m'] = m
    config['t'] = t
    
    return new_w, config
    



