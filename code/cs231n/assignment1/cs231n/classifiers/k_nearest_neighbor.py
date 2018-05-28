import numpy as np

class KNearestNeighbor(object):
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loops(X)
        elif num_loops == 2:
            dists = self.computee_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
            
        return self.predict_labels(dists, k=k)
    
    def compute_distances_two_loops(self, X):
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j, :] - X[i, :])))
        return dists
        
    def compute_distances_one_loops(self, X):
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))
        return dists
        
    def compute_distances_no_loops(self, X):
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train))
        X_train_square = np.sum(np.square(self.X_train), axis=1)
        X_square = np.sum(np.square(X), axis=1)
        dists = np.sqrt(X_train_square + X_square[:, np.newaxis] - 2*X.dot(self.X_train.T))

        return dists
        
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        min_dist = np.zeros((num_test, k))
        sort_dist = np.argsort(dists, axis=1)
        min_dist_idx = sort_dist[:, 0:k]
        min_dist = self.y_train[min_dist_idx]
        y_pred = [np.argmax(np.bincount(min_dist[i, :])) for i in range(num_test)]
        
        return y_pred