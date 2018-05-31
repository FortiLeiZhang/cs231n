import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    loss, dW = 0.0, None
    N = X.shape[0]
    C = W.shape[1]
    dW = np.zeros_like(W)
    
    scores = X.dot(W)
    dScores = np.zeros_like(scores)
    stable_scores = scores - np.max(scores, axis=1, keepdims=True)
    stable_scores = np.exp(stable_scores)

    for i in range(N):
        correct_score = stable_scores[i, y[i]]
        total_score = 0.0
        dScores[i, :] += stable_scores[i, :] / np.sum(stable_scores[i, :])
        dScores[i, y[i]] -= 1
        for j in range(C):
            total_score += stable_scores[i, j]
        loss -= np.log(correct_score/total_score)
    
    loss = loss/N + reg*np.sum(W*W)
    dW = X.T.dot(dScores)
    dW = dW/N + 2*reg*W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    loss, dW = 0.0, None
    N = X.shape[0]
    C = W.shape[1]
    
    scores = X.dot(W)
    dScores = np.zeros_like(scores)
    stable_scores = scores - np.max(scores, axis=1, keepdims=True)
    stable_scores = np.exp(stable_scores)
    dScores = stable_scores / np.sum(stable_scores, axis=1, keepdims=True)
    dScores[np.arange(N), y] -= 1
    
    correct_score = stable_scores[np.arange(N), y]

    loss = -np.sum(np.log(correct_score/np.sum(stable_scores, axis=1)))
    loss = loss/N + reg*np.sum(W*W)
    dW = X.T.dot(dScores)
    dW = dW/N + 2*reg*W        
    return loss, dW