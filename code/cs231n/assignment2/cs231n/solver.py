from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim

class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.mode = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.num_epoch = kwargs.pop('num_epoch', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.verbose = kwargs.pop('verbose', True)
        print_every = kwargs.pop('optim_config', 100)
        
        self.num_train_samples = self.X_train.shape[0]
        self.num_val_samples = self.X_val.shape[0]
        
        if len(kwargs) != 0:
            extra = ', '.join('"%s"' %(k for k in list(kwargs.keys())))
            raise ValueError('Unrecognized arguments %s' % extra)
            
        if not hasattr(optim, self.update_rule):
            raise ValueError('Unrecognized update rule: "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        
        self._reset()
        
    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
    
    def _step():
        idx = np.random.choice(self.num_train_samples, self.batch_size, replace=False)
        X_batch = self.X_train[idx]
        y_batch = self.y_train[idx]
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
            
    def train(self):
        iterations_per_epoch = max(self.num_train_samples // self.batch_size, 1)
        num_iteration = self.num_epoch * iterations_per_epoch
        
        for i in range(num_iteration):
            self._step()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        