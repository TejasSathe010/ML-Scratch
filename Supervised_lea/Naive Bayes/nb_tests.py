import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from nb import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# def golf_datasets():
#     X = pd.DataFrame({
#         'Outlook':['Rainy', 'Rainy', 'Outcast', 'Sunny', 'Sunny'],
#         'Temp':['Hot', 'Hot', 'Hot', 'Mild', 'Cool'],
#         'Humidity':['High', 'High', 'High', 'High', 'Normal'],
#         'Windy':['False', 'True', 'False', 'False', 'False']
#     })
#     y = pd.DataFrame({
#         'Play':['No', 'No', 'Yes', 'Yes', 'Yes',]
#     })
#     return X, y

# X, y = golf_datasets()
# data = pd.concat([X, y], axis=1)
# # print(data.head())

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()

# for col in X.columns:
#     X[col] = label_encoder.fit_transform(X[col])

# y['Play'] = label_encoder.fit_transform(y['Play'])
# print(X.head())
# print(y)

# X = np.array(X, dtype=np.float64)
# y = np.array(y, dtype=np.float64)

# X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X, y = datasets.make_classification(n_samples=100, n_features=4, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# print(len(y[y == 0]) / len(y))
# print(len(y[y == 1]) / len(y))

nb = NaiveBayes()
# print(X_test)
nb.fit(X_train, y_train)

# predictions = nb.predict(X_test)
# print(predictions)

# predictions = nb.predict([[ 0.83617024,  0.47576265, 0.76693704,  1.54433392]])
# print(predictions)

# print('Naive Bayes Classification Accuracy: ', accuracy(y_test, predictions))
