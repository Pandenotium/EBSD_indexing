import cv2
import numpy as np


def train_read(begin, n):
    im_result = np.zeros((n,60,60,1))
    y_result = np.zeros((n,6))
    for i in range(begin,begin+n):
        if i % 20 == 0:
            print('\r' + "Loading progress:%d/%d"%(i-begin,n),end = '',flush=True)
        im = cv2.imread('./image2/%i.png' %(i))
        im_result[i-begin] = im[:,:,0].reshape((60,60,1))
        #im_result = im_result/255
        #im_result = im_result.T

        file_object = open('./image2/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        """
        for j in range(6):
            file_context[j] = int(file_context[j])
        """
        """
        for k in range(3):
            y_result[i-begin,k] = file_context[k]/180
        y_result[i-begin,3] = (file_context[-1]-450)/100
        """
        
        y_result[i-begin] = file_context
    return im_result, y_result

def train_read2(begin, n):
    im_result = np.zeros((n,60,60,1))
    y_result = np.zeros((n,6))
    for i in range(begin,begin+n):
        if i % 20 == 0:
            print('\r' + "Loading progress:%d/%d"%(i-begin,n),end = '',flush=True)
        im = cv2.imread('./imagetest/%i.png' %(i))
        im_result[i-begin] = im[:,:,0].reshape((60,60,1))
        im_result = im_result/255
        im_result = im_result.T

        file_object = open('./imagetest/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        """
        for j in range(6):
            file_context[j] = int(file_context[j])
        """
        """
        for k in range(3):
            y_result[i-begin,k] = file_context[k]/180
        y_result[i-begin,3] = (file_context[-1]-450)/100
        """
        
        y_result[i-begin] = file_context
    return im_result, y_result

def fc_read(begin, n):
    im_result = np.zeros((n,60,60,1))
    y_result = np.zeros((n,6))
    for i in range(begin,begin+n):
        if i % 20 == 0:
            print('\r' + "Loading progress:%d/%d"%(i-begin,n),end = '',flush=True)
        im = cv2.imread('./imagee/%i.png' %(i))
        im_result[i-begin] = im[:,:,0].reshape((60,60,1))
        im_result = im_result/255
        im_result = im_result.T

        file_object = open('./imagee/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        """
        for j in range(6):
            file_context[j] = int(file_context[j])
        """
        """
        for k in range(3):
            y_result[i-begin,k] = file_context[k]/180
        y_result[i-begin,3] = (file_context[-1]-450)/100
        """
        
        y_result[i-begin] = file_context
    return im_result, y_result


def train_read1(begin, n):
    im_result = np.zeros((n,60,60,1))
    y_result = np.zeros((n,6))
    for i in range(begin,begin+n):
        if i % 20 == 0:
            print('\r' + "Loading progress:%d/%d"%(i-begin,n),end = '',flush=True)
        im = cv2.imread('./imagetest/%i.png' %(i))
        im_result[i-begin] = im[:,:,0].reshape((60,60,1))/255
        im_result = im_result.T

        file_object = open('./imagetest/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        """
        for j in range(6):
            file_context[j] = int(file_context[j])
        """
        """
        for k in range(3):
            y_result[i-begin,k] = file_context[k]/180
        y_result[i-begin,3] = (file_context[-1]-450)/100
        """
        
        y_result[i-begin] = file_context
    return im_result, y_result


def test_read(begin, n):
    im_result = np.zeros((n,128,143,1))
    y_result = np.zeros((n,6))
    for i in range(begin,begin+n):
        im = cv2.imread('./image2/%i.png' %(i))
        im_result[i-begin] = im[:,:,0].reshape((128,143,1))

        file_object = open('./image2/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        for j in range(6):
            file_context[j] = int(file_context[j])

        y_result[i-begin,:3] = file_context[:3]
        y_result[i-begin,3] = file_context[-1]
    return im_result, y_result


def im_read2(n):
    im_result = np.zeros((n,128,143,1))
    y_result = np.zeros((n,4))
    for i in range(n):
        im = cv2.imread('./resultimage/%i.png' %(i))
        im_result[i] = im[:,:,0].reshape((128,143,1))
        
        file_object = open('./resultimage/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        for j in range(6):
            file_context[j] = int(file_context[j])
        
        y_result[i,:3] = file_context[:3]
        y_result[i,3] = file_context[-1]

    return im_result, y_result

def im_read3(n, begin):
    im_result = np.zeros((n,128,143,1))
    y_result = np.zeros((n,6))
    for i in range(n):
        print("Loading progress:%d/%d\r"%(i-begin,n),end = '\r')
        im = cv2.imread('./resultimage/%i.png' %(i))
        im_result[i-begin] = im[:,:,0].reshape((128,143,1))

        file_object = open('./resultimage/%i.txt'%(i))
        try:
            file_context = file_object.read()
        finally:
            file_object.close()
        file_context = file_context.strip().split(',')
        
        y_result[i-begin] = file_context

    return im_result, y_result

#print(train_read(1,1)[0])