from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):
    
    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        N, dim = X.shape
        C = np.max(y) + 1
        if self.W == None:
            self.W = np.random.randn(dim, C) * 0.001
            
        loss_his = []
        for it in range(num_iters):
            X_batch, y_batch = None, None
            
            idx = np.random.choice(N, batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_his.append(loss)
            self.W = self.W - learning_rate * grad
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_his
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    
    def loss(self, X, y, reg):
        pass
    
class LinearSVM(LinearClassifier):
    
    def __init__(self):
        super().__init__()
        
    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.W, X, y, reg)
    
class Softmax(LinearClassifier):
    
    def __init__(self):
        super().__init__()
        
    def loss(self, X, y, reg):
        return softmax_loss_vectorized(self.W, X, y, reg)
