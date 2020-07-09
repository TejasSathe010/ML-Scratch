import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # init Parameters 
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        # print(self.weights)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # print(y_predicted)
            # print(X.T.shape)
            # print((np.dot(X.T, (y_predicted - y))).shape)
            # print((y_predicted - y).shape)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # print(dw)
            db = (1/n_samples) * np.sum(y_predicted - y)
            # print(db)


            # Regularization
            # dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) + (0.001/n_samples * np.sum(self.weights ** 2))
            # # print(dw)
            # db = (1/n_samples) * np.sum(y_predicted - y) + (0.001/n_samples * np.sum(self.weights ** 2))
            # # print(db)

            self.weights -= self.lr * dw 
            # print(self.weights)
            self.bias -= self.lr * db
            # print(self.bias)

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
