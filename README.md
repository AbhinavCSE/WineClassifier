# WineClassifier
Classifying Wine based on the given attribute values

## The Data
We have been given a dataset about wines with the following attributes(14 including the ID)
* ID
* Alcohol
* Malic Acid
* Ash
* Alcalinity of ash
* Magnesium
* Total pheonls
* Flavanoids
* Nonflavanoid phenols
* Proanthocyanins
* Color intensity
* Hue
* OD280/OD315 of diluted wines
* Proline

The dataset can be downloaded from [UCI ML Repository Wine Data Set](https://archive.ics.uci.edu/ml/datasets/wine)

## Prerequisites
You need to have the following in python3
 * Numpy
 * Sklearn
 * Pandas
 
## Classifying
### Importing
```
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm,ensemble
import pandas as pd
```
### Loading the dataset
```
df = pd.read_csv('wine.data.txt')
```

### Deciding the features and labels (X and y)
```
X = np.array(df.drop(['id'],1))
y = np.array(df['id'])
```

### Splitting the dataset into Training and Testing
```
X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
```

### Training with RandomForestClassifier
```
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
```

Based on the classifier, we will obtain different accuracy scores, for example:
* RandomForestClassifier gives 96-99% accuraccy

### Calculating the accuracy using Test data
```
accuracy = clf.score(X_test,y_test)
print(accuracy)
```
