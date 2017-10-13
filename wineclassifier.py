import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm,ensemble
import pandas as pd

df = pd.read_csv('wine.data.txt')
X = np.array(df.drop(['id'],1))
y = np.array(df['id'])

X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)
