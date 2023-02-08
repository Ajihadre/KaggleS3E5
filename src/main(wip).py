# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:44:37 2023

@author: tcaron
"""

import sys
sys.path.insert(0, './src')
from modelling import model_prep as mp
#from modelling import model_train as mt
#from modelling import model_train_rf as mtrf
from modelling import model_train_xgb as mt
from preprocessing import feature_engineering as fe
import pandas as pd
import warnings

if __name__ == '__main__':
    warnings.simplefilter("ignore", UserWarning)
    """
    CHEMIN A MODIFIER :
    """
    path = "/Users/tcaron/Documents/Python Scripts/KaggleS3E5/data/"
    train = pd.read_csv(path+"train.csv")
    test = pd.read_csv(path+"test.csv")
    origin = pd.read_csv(path+"WineQT.csv")
    Id = test[["Id"]]
    train = pd.concat([train,origin],ignore_index=True)
    train,test= fe.pca(train, test)
    X_test = test.drop(columns = "Id")
    X_train,X_val,y_train,y_val = mp.SelfSplitTrain(train)
    score1 = mt.objective()
    #y_pred = mt.model_training(tupleXY[0], tupleXY[1], tupleXY[2], tupleXY[3], X_test)
    #y_pred = fe.sortie_prep(y_pred)
    #Id["quality"]=y_pred
    #print(Id.head(3))
    #Id.to_csv("sample_submission.csv",index=False)