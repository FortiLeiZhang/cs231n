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
        self.model = model
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
        self.print_every = kwargs.pop('print_every', 100)
        
        self.num_train_samples = self.X_train.shape[0]
        self.num_val_samples = self.X_val.shape[0]
        
        if len(kwargs) != 0:
            extra = ', '.join('"%s"' %k for k in list(kwargs.keys()))
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
    
    def _step(self):
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

    def check_acc(self, X, y):
        N = X.shape[0]
        
        if N > self.num_val_samples:
            mask = np.random.choice(N, self.num_val_samples)
            X = X[mask]
            y = y[mask]
            
        scores = self.model.loss(X)
        y_pred = np.argmax(scores, axis=1)
        acc = np.mean(y_pred == y)
        
        return acc
 
    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def train(self):
        iterations_per_epoch = max(self.num_train_samples // self.batch_size, 1)
        num_iterations = self.num_epoch * iterations_per_epoch
        
        for i in range(num_iterations):
            self._step()
            
            if self.verbose and i % self.print_every == 0:
                print('Iteration %d / %d - loss: %f' % (i + 1, num_iterations, self.loss_history[-1]))
            
            epoch_end = ((i + 1) % iterations_per_epoch == 0)
            
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
                    
            first_it = (i == 0)
            last_it = (i == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_acc(self.X_train, self.y_train)
                val_acc = self.check_acc(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()
                
                if self.verbose:
                    print('Epoch %d / %d - train acc: %f; val_acc: %f' % (self.epoch, self.num_epoch, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params
        print('Epoch %d / %d - best_val_acc: %f' % (self.epoch, self.num_epoch, self.best_val_acc))