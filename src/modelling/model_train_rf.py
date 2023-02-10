# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:33:47 2023

@author: tcaron
"""
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
def model_training(X_train,X_val,y_train,y_val,X_test):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    
    # Number of features to consider at every split
    max_features = ['auto']
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num = 5)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8]
    
    # Method of selecting samples for training each tree
    bootstrap = [True]

    pipe_clf = Pipeline([('scaler', preprocessing.StandardScaler())
                         , ('rf', RandomForestClassifier(random_state=42))])
    param_grid = {
        'rf__n_estimators': n_estimators, 
        'rf__max_features': max_features,
        'rf__max_depth': max_depth,
        'rf__bootstrap': bootstrap,
        'rf__min_samples_split':min_samples_split,
        'rf__min_samples_leaf':min_samples_leaf
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
