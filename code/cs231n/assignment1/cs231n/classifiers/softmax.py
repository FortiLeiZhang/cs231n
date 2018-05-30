import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    loss, grads = 0.0, None
    N = X.shape[0]
    C = W.shape[1]
    
    scores = X.dot(W)
    stable_scores = scores - np.max(scores, axis=1, keepdims=True)
    stable_scores = np.exp(stable_scores)

    for i in range(N):
        correct_score = stable_scores[i, y[i]]
        total_score = 0.0
        for j in range(C):
            total_score += stable_scores[i, j]
        loss -= np.log(correct_score/total_score)
        
    loss = loss/N + reg*np.sum(W*W)
        
    return loss, grads

def softmax_loss_vectorized(W, X, y, reg):
    loss, grads = 0.0, None
    N = X.shape[0]
    C = W.shape[1]
    
    scores = X.dot(W)
    stable_scores = scores - np.max(scores, axis=1, keepdims=True)
    stable_scores = np.exp(stable_scores)
    
    correct_score = stable_scores[np.arange(N), y]

    loss = -np.sum(np.log(correct_score/np.sum(stable_scores, axis=1)))
    loss = loss/N + reg*np.sum(W*W)
        
    return loss, grads