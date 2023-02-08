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

def model_training(X_train,X_val,y_train,y_val,X_test):

    pipe_clf = Pipeline([('scaler', preprocessing.StandardScaler())
                         , ('rf', RandomForestClassifier(random_state=42))])
    param_grid = {
        'rf__n_estimators': [300,400, 500, 600,700,800,100,200], 
        'rf__max_features': ['auto'],
        'rf__max_depth': [25,20,30,10,15,10,5,None],
        'rf__bootstrap': [True,False]
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
