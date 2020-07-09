import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


#Mammal: 0, Reptile: 1

data = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "hair":["True","True","False","True","True","True","False","False","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":[1, 1, 0, 1, 1, 1, 0, 0, 1, 0]}, 
                    columns=["toothed","hair","breathes","legs","species"])

features = data[["toothed","hair","breathes","legs"]]
target = data["species"]

X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy: ", acc*100, "%")

col_names = ["toothed" ,"hair", "breathes", "legs", "species"]
clf.display_tree(col_names)