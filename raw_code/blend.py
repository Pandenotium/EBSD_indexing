# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:38:08 2020

@author: Chapman Guan
"""

import PIL.Image as PI
from cv2 import pyrUp
import cv2
import numpy as np
from read_h5 import test_read
'''
img = PI.open("./image2/1.png")
img = img.transpose(PI.TRANSPOSE)
img = img.convert('RGBA')
'''
'''
for i in range(256):
    imgt = PI.open("./mask0.5%/conv1_mask_" + str(i) + "_1.png")
    imgt = imgt.convert('RGBA')
    imgf = PI.blend(img,imgt,0.6)
    imgf.save("./masked0.5%/conv1_masked_" + str(i) + ".png")
'''

'''
for i in range(256):
    imgt = PI.open("./mask0.5%abs/conv1_mask_" + str(i) + "_1.png")
    imgt = imgt.convert('RGBA')
    imgf = PI.blend(img,imgt,0.6)
    imgf.save("./masked0.5%abs/conv1_masked_" + str(i) + ".png")
'''
'''
for i in range(96):
    imgt = PI.open("./results9/conv6_Relu_" + str(i) + "_1.png")
    imgt = pyrUp(np.asarray(imgt))
    imgt = PI.fromarray(imgt)
    imgt = imgt.convert('RGBA')
    imgf = PI.blend(img,imgt,0.6)
    imgf.save("./masked9/conv1_masked9_" + str(i) + "_1.png")
''' 
for j in range(10):
    inadrf = "modified gradcam/ly_" + str(j) + ".png" #此处修改输入文件
    outadr = "modified gradcam/blend_ly_" + str(j) + ".png" #此处修改输出文件
    img = test_read(j+40,1)[0] #此处修改背景
    img = img.reshape(60,60)
    img = img*255
    img = PI.fromarray(img.astype(np.uint8))
    img = img.convert('RGBA')
    imgt = PI.open(inadrf)
    imgt = pyrUp(np.asarray(imgt))
    imgt = cv2.resize(imgt,(60,60),interpolation=cv2.INTER_AREA)
    imgt = PI.fromarray(imgt)
    imgt = imgt.convert('RGBA')
    imgf = PI.blend(img,imgt,0.6)
    imgf.save(outadr)
    