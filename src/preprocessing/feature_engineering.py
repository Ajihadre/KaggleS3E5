# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:16:23 2023

@author: tcaron
"""
from sklearn.decomposition import PCA
import pandas as pd

def pca(train,test,columns=["pH","fixed acidity"]):
    target = "quality"
    df_trn = train.copy(deep = True)
    df_tst = test.copy(deep = True)
    df_trn[target] = df_trn[target].map({3:0,
                    4:1,
                    5:2,
                    6:3,
                    7:4,
                    8:5})
    pca_ = PCA(n_components=1 ,whiten= False)
    df_trn["pca_1"] = pca_.fit_transform(df_trn[columns])
    df_tst["pca_1"] = pca_.fit_transform(df_tst[columns])
    
    for cols in columns:
        for df in [df_trn,df_tst]:
            df.drop(cols, axis =1, inplace = True)
    return (df_trn,df_tst)

def sortie_prep(y_pred):
    return y_pred +3