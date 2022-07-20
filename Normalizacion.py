# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:05:05 2022

@author: Holger
"""

import numpy as np
import featureNormalize as fn

#%%featureNormalize
def Normalizacion(A,ind):
#Esta función permite normalizar las variables de entrada
    X=np.zeros((ind.shape[1],A.shape[2]))
    for n in np.arange(0,A.shape[2]):
        M=A[:,:,n]
        MV=np.transpose(M).reshape(-1); #MV=MV.reshape(MV.shape[0],1);
        #temp=MV[ind]
        #temp=temp.reshape(-1);
        X[:,n]=MV[ind]
#%%Normalizacion    (Esta funcionando excelente)

    (X_norm, mu, sigma)=fn.featureNormalize(X)
    nd=1
    X_train=X_norm[::nd,:]
    X_train=np.transpose(X_norm)
    return(X_train)