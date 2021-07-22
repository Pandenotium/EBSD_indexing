# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:10:49 2020

@author: Chapman Guan
"""

import h5py as h
import numpy as np

def test_read(begin,n):
    im_result = np.zeros((n,60,60,1))
    y_result = np.zeros((n,6))
    file = h.File("../data/highlin.h5","r")
    for i in range(begin,begin+n):
        if i % 20 == 0:
            print('\r' + "Loading progress:%d/%d"%(i-begin,n),end = '',flush=True)
        im = file['/images'][i]
        im_result[i-begin] = im.reshape((60,60,1))
        im_result = im_result/255
        
        y = file['/lables'][i]
        y_result[i-begin] = y.reshape((6))
        
    return im_result, y_result 
        
def standard_read():
    im_result = np.zeros((1,60,60,1))
    y_result = np.zeros((1,6))
    file = h.File("standard.h5","r")
    im = file['/images'][0]
    im_result[0] = im.reshape((60,60,1))
    im_result = im_result/255
        
    y = file['/lables'][0]
    y_result[0] = y.reshape((6))
        
    return im_result, y_result