# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:04:26 2023

@author: tcaron
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def SelfSplitTrain(df,test_size=0.2):
    X = df.drop(columns=["Id","quality"])
    y = df[["quality"]]
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=test_size,random_state=42)
    return (X_train, X_val, y_train, y_val)