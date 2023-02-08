# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:27:22 2023

@author: tcaron
"""
# Metric import
from sklearn.metrics import cohen_kappa_score

# Modeling imports
from xgboost  import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import numpy as np

def objective(trial):
    target_clases = train["quality"].value_counts()
    n_classes = target_clases.nunique()
    params_optuna = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 1.0),
            'objective' : " multi:softmax",
            'num_class': n_classes,
        }
    
    n=trial.suggest_int('n_cv', 3, 10)
    cv = StratifiedKFold(n,shuffle=True, random_state=42)
    fold_scores = []
    model = XGBClassifier(**params_optuna)
    model.fit(X_train,
              y_train,
              eval_set= [(X_val,y_val)],
              early_stopping_rounds = 50,
              verbose=500)

    pred_val = model.predict(X_val)

    score = cohen_kappa_score(y_val,pred_val, weights='quadratic')
    fold_scores.append(score)
    return np.mean(fold_scores)

def run_study(func=objective,train=train,X_train=X_train,X_val=X_val,y_train=y_train,y_val=y_val):
    study = optuna.create_study(direction='maximize', sampler = TPESampler())
    study.optimize(func=func, n_trials=500)
    print(study.best_params)
    return study
