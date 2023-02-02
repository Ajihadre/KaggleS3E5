# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:22:36 2023

@author: tcaron
"""
import sys
sys.path.insert(0, './src')
from modelling import model_prep as mp
from modelling import model_train as mt
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
    Id = test[["Id"]]
    X_test = test.drop(columns = "Id")
    tupleXY = mp.SelfSplitTrain(train)
    y_pred = mt.model_training(tupleXY[0], tupleXY[1], tupleXY[2], tupleXY[3], X_test)
    Id["quality"]=y_pred
    print(Id.head(3))
    Id.to_csv("sample_submission.csv",index=False)