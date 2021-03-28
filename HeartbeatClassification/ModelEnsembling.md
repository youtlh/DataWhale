# Model Ensembling

## Simple weighting ensemble

### Regression

- Arithmetic mean: for numerical prediction results from basic models, we can take average or advanced weighted average. The weighting value can be determined by sequence of accuracy.
- Geometric mean

### Classification

- Voting: for binary problems, 3 basic models can vote and decide the final result

### General

- Rank averaging
- Log ensemble

## Boosting / bagging

Used in xgboost, adaboost and GBDT, refine the accuracy with multiple tree.

### Bagging

Sample with replacement to construct and train child model. Can be run in parallel. Robust against outliers or noisy data. Repeat this process multiple times and combine.

- Steps:
    - Repeat the sampling and training K times
    - Combine the K child models with voting for classification and average for regression.
- Sample: random forest.

### Boosting

Compared to bagging that can be parallel, boosting is an iterative method, such that each training is more focused on the mistakes in the previous attempt, given them more weighting. Not robust against outliers or noisy data. It is flexible to be used with any loss function.

## Stacking / blending

Construct multiple models as the first-level (base) and make the final prediction (second level) by fitting their prediction result.

- Steps:
    1. Train the model M1 on training set with cross validation (split training set into subsets, train on one of the subset and predict on another) to prevent overfitting, and generate a prediction result for each row in training P1 and testing set T1
    2. Repeat this model for the other M2, M3... basic models

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2546225a-01c0-4c2d-abf6-1733f6fa1fcd/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2546225a-01c0-4c2d-abf6-1733f6fa1fcd/Untitled.png)

    3. Combine the P1, P2 and P3 as the new training, and T1, T2 and T3 as the new testing set.
    4. Train another model M4 with train2 data and predict on test2.
- Code

```python
# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
```

Write a class SklearnHelper that allows one to extend the inbuilt methods (such as train, predict and fit) common to all the Sklearn classifiers

```python
# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer
```

To avoid overfitting

```python
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
```
