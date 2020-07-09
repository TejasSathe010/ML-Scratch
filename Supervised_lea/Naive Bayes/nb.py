import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        # print(y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        # print(self._classes)
        n_classes = len(self._classes)

        # init mean, var, prior
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        # print(self._mean)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        # print(self._priors)

        for c in self._classes:
            # print(c)
            # print(c == y)
            X_c = X[c == y]
            # print(X_c)
            self._mean[c,:] = X_c.mean(axis=0)
            # print(self._mean)
            self._var[c,:] = X_c.var(axis=0)
            # print(self._var)
            self._priors[c] = X_c.shape[0] / float(n_samples)
            # print(self._priors)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self ,x):
        posteriors = []
        for idx, _ in enumerate(self._classes):
            # print(idx)
            prior = np.log(self._priors[idx])
            # print(prior)
            # print(self._pdf(idx, x))
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        # print(posteriors)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        # print(mean)
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
