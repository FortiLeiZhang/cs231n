import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    loss, dW = None, None
    # X: (N, 3073)
    # W: (3073, 10)
    N = X.shape[0]
    C = W.shape[1]
    dW = np.zeros_like(W)
    scores = X.dot(W)

    for i in range(N):
        correct_score = scores[i, y[i]]
        cnt = 0
        for j in range(C):
            if j == y[i]:
                scores[i, j] = 0
            else:
                t = scores[i, j] - correct_score + 1.0
                if t > 0:
                    cnt += 1
                    scores[i, j] = t
                    dW[:, j] += X[i, :]
                else:
                    scores[i, j] = 0
        dW[:, y[i]] -= cnt * X[i, :]
    
    loss = np.sum(scores) / N + reg * np.sum(W * W)
    dW = dW / N + 2 * reg * W
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    loss, dW = None, None
    N = X.shape[0]
    C = W.shape[1]
    dW = np.zeros_like(W)
    
    scores = X.dot(W)
    correct_score = scores[np.arange(N), y]
    scores = np.maximum(0, scores - correct_score[:, np.newaxis] + 1.0)
    scores[np.arange(N), y] = 0
    dScore = (scores > 0).astype(np.float)
    dScore[np.arange(N), y] = -np.sum(dScore, axis=1)
    
    dW = X.T.dot(dScore)
    loss = np.sum(scores) / N + reg * np.sum(W * W)
    dW = dW / N + 2 * reg * W
    return loss, dW