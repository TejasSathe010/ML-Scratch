import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=3, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train, y_train)
from linear_regression import LinearRegression

reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)
# print(y_predicted)

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

mse_value = mse(y_test, y_predicted)
print(mse_value)


fig = plt.figure(figsize=[8, 6])
plt.scatter(X[:,0], y, color='b', marker='o', s=30)
plt.plot(X_train, reg.predict(X_train), color='r', linewidth=5)
plt.show()