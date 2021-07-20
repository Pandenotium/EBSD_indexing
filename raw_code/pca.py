# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:47:25 2020

@author: Chapman Guan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PCA(object):
    
    def __init__(self,x):
        self.x = x-np.mean(x)
    
    def cptCov(self):
        x_T = np.transpose(self.x)
        return np.cov(x_T)

    def cptFeature(self):
        cov = self.cptCov()
        lamda, X = np.linalg.eig(cov)
        index = lamda.shape[0]
        dt = np.hstack((lamda.reshape((index,1)), X))
        df = pd.DataFrame(dt)
        df_sort = df.sort(columns=0,ascending=False)
        return df_sort
    
    def redDim(self,dim):
        mat_sort = self.cptFeature()
        mat_red = mat_sort.values[0:dim,1:]
        reduced = np.dot(mat_red, np.transpose(self.x))
        return np.transpose(reduced)

'''
data_p = np.load("phi1.npy")
print(data_p.shape)
for i in range(data_p.shape[1]):
    res = data_p[:,i]
    res = np.reshape(res,[16,16])
    plt.imshow(res)
    plt.colorbar()
    plt.show()
'''
'''
data = np.load("phi1_after.npy")
for i in range(data.shape[1]):
    plt.plot(data[:,i])
    plt.show()
'''
