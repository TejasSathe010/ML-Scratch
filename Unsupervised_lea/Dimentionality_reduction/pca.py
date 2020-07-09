import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components 
        self.components = None # Eigenvectors
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # print(X)

        # covariance
        # row = 1 sample, 1 column = 1 feature
        cov = np.cov(X.T)
        # print(cov)
        # [[ 0.68569351 -0.042434    1.27431544  0.51627069]
        #  [-0.042434    0.18997942 -0.32965638 -0.12163937]
        #  [ 1.27431544 -0.32965638  3.11627785  1.2956094 ]
        #  [ 0.51627069 -0.12163937  1.2956094   0.58100626]]

        # eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        print(eigenvectors)
        print(eigenvalues)
        # 1 col = 1 eigenvectors
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # projection
        X = X - self.mean
        return np.dot(X, self.components.T)
