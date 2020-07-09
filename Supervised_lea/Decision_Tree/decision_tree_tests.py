import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data['data']
# print(X.shape)
y = data['target']
feature_cols = data['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy: ", acc*100, "%")
# clf.display_tree(feature_cols)