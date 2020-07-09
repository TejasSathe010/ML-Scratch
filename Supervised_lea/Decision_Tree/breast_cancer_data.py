import pandas as pd
from sklearn import datasets

data = datasets.load_breast_cancer()
#print(data.keys())
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
X = data['data']
y = data['target']
features = data['feature_names']
X = pd.DataFrame(X, columns= features)
y = pd.DataFrame(y, columns= ['Target'])
df = pd.concat([X, y], axis=1)
print(df.head())