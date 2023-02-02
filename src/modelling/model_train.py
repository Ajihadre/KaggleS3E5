# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:33:47 2023

@author: tcaron
"""
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def model_training(X_train,X_val,y_train,y_val,X_test):
    y_train = y_train.values
    y_val = y_val.values
    pipe_clf = Pipeline([('scaler', preprocessing.StandardScaler())
                         , ('svc', SVC())])
    param_grid = {
        'svc__C': [0.1, 1, 10, 100, 1000], 
        'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'svc__kernel': ['rbf']
        }
    grid = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,n_jobs=-1,verbose=1)
    grid.fit(X_train,y_train)
    # print best parameter after tuning
    print(grid.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_val)
    print(classification_report(y_val, grid_predictions))
    return grid.predict(X_test)
