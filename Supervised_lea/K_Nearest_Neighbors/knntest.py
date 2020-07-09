import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris['data'], iris['target']
# print(iris['target_names']) ['setosa' 'versicolor' 'virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from knn import kNN

clf = kNN(k=5)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
acc = np.sum(prediction == y_test) / len(y_test)
print(acc)

# plt.figure()
# plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()