# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:23:05 2020

@author: Chapman Guan
"""


import sys, traceback, os
import re, math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim, time


from tensorflow.python import debug as tf_debug

import matplotlib.pyplot as plt

from img3 import train_read1


from PIL import Image

batch_size = 64
architecture= 'infile'
reg_W = 0.

patience = 100
filter_size = 3
IMAGE_SIZE = 60
NUM_CHANNELS = 1
stride = 1

save_dir = os.getcwd()
#os.system('mkdir -p '+log_folder)

#save_dir = os.path.join(log_folder, logfile.split('.')[0])
#os.system('mkdir -p '+save_dir)

conffile = None

#checkpoint_fn = os.path.join(log_folder,
#							 'checkpoint_'+logfile+'.h5')
#log_fn = os.path.join(log_folder, 'nn_'+logfile)
#pp_file = os.path.join(log_folder, 'pp_'+logfile)
#rr_file = os.path.join(log_folder, 'rr_'+logfile)

if architecture == 'infile':
    architecture = [{'layer_type':'conv', 'num_filters':256, 'input_channels':1, 'filter_size':filter_size, 'border_mode':'same', 'init':'glorot_uniform', 'stride':stride,'activation':'relu', 'reg_W':reg_W},
                    {'layer_type': 'conv', 'num_filters': 256,'stride':stride, 'input_channels': 256, 'filter_size': filter_size, 'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    #{'layer_type': 'maxpool2D', 'pool_size':2},
                    #{'layer_type':'dropout', 'value':0.5},
                    {'layer_type': 'conv', 'num_filters': 512,'stride':stride, 'input_channels': 256, 'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    {'layer_type': 'conv', 'num_filters': 512,'stride':stride, 'input_channels': 512, 'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},
                    {'layer_type': 'maxpool2D', 'pool_size': 2},
                    #{'layer_type': 'dropout', 'value': 0.5},
                    {'layer_type': 'conv', 'num_filters': 256, 'stride': stride, 'input_channels': 256,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    {'layer_type': 'conv', 'num_filters': 256, 'stride': stride, 'input_channels': 256,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},

                    {'layer_type': 'conv', 'num_filters': 96, 'stride': stride, 'input_channels': 256,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    {'layer_type': 'conv', 'num_filters': 96, 'stride': stride, 'input_channels': 96,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},
                    {'layer_type': 'maxpool2D', 'pool_size': 2},
                    {'layer_type': 'flatten'},
                    {'layer_type': 'fully_connected','num_outputs': 8192,'num_inputs':96 * 15 * 15,'activation':'relu', 'reg_W':reg_W, 'init':'glorot_uniform'},
                    {'layer_type': 'fully_connected', 'num_outputs': 2048, 'num_inputs': 8192,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},

                    {'layer_type': 'fully_connected', 'num_outputs': 1024, 'num_inputs': 2048,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform','branch':True},
                    {'layer_type': 'fully_connected', 'num_outputs': 256, 'num_inputs': 1024,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},

                    {'layer_type':'fully_connected','num_outputs': 1, 'num_inputs':256, 'activation':'linear', 'reg_W':reg_W, 'init':'glorot_uniform'}
    ]


SEED = 66478
'''
def model_slim(data, architecture, train=True):
    i=0
    branch = False
    start_branch = False
    if train:
        reuse = None
    else:
        reuse = True
    nets = {}
    nets[0] = data
    for arch in architecture:
        i +=1
        layer_type = arch['layer_type']
        if  ('branch' in arch.keys()) and arch['branch']:
            if not start_branch: start_branch = True
            else: start_branch = False
            branch = True
        else:
            start_branch = False
        if layer_type == 'conv':
            print ('adding cnn layer..', i)
            num_filters = arch['num_filters']
            filter_size = arch['filter_size']
            border_mode = 'SAME'
            activation = tf.nn.relu
            if 'border_mode' in arch.keys():
                border_mode = arch['border_mode']
            padding=border_mode
            if 'padding' in arch.keys():
                padding = arch['padding']
            if 'activation' in arch.keys():
                if arch['activation'] == 'sigmoid':
                    activation = tf.nn.sigmoid
            stride = 1
            if 'stride' in arch.keys():
                stride = arch['stride']
            weights_initializer = tf.truncated_normal_initializer(stddev=0.05)
            if not branch:
                print ('not branch')
                nets[i] = slim.layers.conv2d(nets[i-1], num_outputs=num_filters,kernel_size=[filter_size, filter_size], weights_initializer=weights_initializer, padding=padding, scope='conv'+str(i), stride=stride, weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, activation_fn=activation)
            elif branch:
                print ('branch')
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        print ('start branch...',j)
                        nets[i][j] = slim.layers.conv2d(nets[i - 1], num_outputs=num_filters,
                                             kernel_size=[filter_size, filter_size],
                                             weights_initializer=weights_initializer, padding=padding, weights_regularizer=slim.l2_regularizer(0.001),
                                             scope='conv' + str(i)+str(j), stride=stride, reuse=reuse, activation_fn=activation)
                    else:
                        print ('not start branch')
                        nets[i][j] = slim.layers.conv2d(nets[i - 1][j], num_outputs=num_filters,
                                                        kernel_size=[filter_size, filter_size],
                                                        weights_initializer=weights_initializer, padding=padding, weights_regularizer=slim.l2_regularizer(0.001),
                                                        scope='conv' + str(i)+str(j), stride=stride, reuse=reuse, activation_fn=activation)

        elif layer_type == 'fully_connected':
            num_outputs = arch['num_outputs']
            activation == tf.nn.relu
            if arch['activation'] == 'sigmoid':
                activation = tf.nn.sigmoid
            elif arch['activation'] =='linear':
                activation = None
            print ('adding fully connected layer...', i, ' with ', num_outputs, ' branching is ', branch, 'start branch is : ', str(start_branch))

            if not branch:
                print ('not branch')
                nets[i] = slim.layers.fully_connected(nets[i-1], num_outputs=num_outputs, scope='fc'+str(i),activation_fn=activation, reuse=reuse)
            elif branch:
                print ('branch')
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        print ('start branch..')
                        nets[i][j] = slim.layers.fully_connected(nets[i-1], num_outputs=num_outputs, scope='fc'+str(i)+str(j),activation_fn=activation, reuse=reuse)
                    else:
                        print ('not start branch')
                        nets[i][j] = slim.layers.fully_connected(nets[i-1][j], num_outputs=num_outputs,
                                                                 scope='fc' + str(i)+str(j), activation_fn=activation,
                                                                 reuse=reuse)

        elif layer_type == 'AvgPool2D':
            if not branch:
                nets[i] = slim.layers.avg_pool2d(nets[i-1], [arch['pool_size'], arch['pool_size']])
            elif branch:
                nets[i] = [None, None , None, None, None, None]
                for j in range(6):
                    if start_branch:
                        nets[i][j] = slim.layers.avg_pool2d(nets[i-1], [arch['pool_size'], arch['pool_size']])
                    else:
                        nets[i][j] = slim.layers.avg_pool2d(nets[i-1][j], [arch['pool_size'], arch['pool_size']])

        elif layer_type == 'maxpool2D':
            print ('adding maxpoo2D...', i)
            if not branch:
                nets[i] = slim.layers.max_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
            elif branch:
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        nets[i][j] = slim.layers.max_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
                    else:
                        nets[i][j] = slim.layers.max_pool2d(nets[i - 1][j], [arch['pool_size'], arch['pool_size']])


        elif layer_type == 'flatten':
            if not branch:
                nets[i] = slim.layers.flatten(nets[i-1], scope='flatten'+str(i))
            elif branch:
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        nets[i][j] = slim.layers.flatten(nets[i-1], scope='flatten'+str(i)+str(j))
                    else:
                        nets[i][j] = slim.layers.flatten(nets[i-1][j], scope='flatten' + str(i)+str(j))
        elif layer_type == 'dropout':
            if not branch:
                nets[i] = tf.nn.dropout(nets[i-1], arch['value'], seed=SEED)
            elif branch:
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    nets[i][j] = tf.nn.dropout(nets[i-1][j], arch['value'], seed=SEED)

    return nets[i]
    '''
    
def model_slim(data, architecture, train=True):
    i=0
    branch = False
    start_branch = False
    if train:
        reuse = None
    else:
        reuse = None
    nets = {}
    nets[0] = data
    for arch in architecture:
        i +=1
        layer_type = arch['layer_type']
        if  ('branch' in arch.keys()) and arch['branch']:
            if not start_branch: start_branch = True
            else: start_branch = False
            branch = True
        else:
            start_branch = False
        if layer_type == 'conv':
            print ('adding cnn layer..', i)
            num_filters = arch['num_filters']
            filter_size = arch['filter_size']
            border_mode = 'SAME'
            activation = tf.nn.relu
            if 'border_mode' in arch.keys():
                border_mode = arch['border_mode']
            padding=border_mode
            if 'padding' in arch.keys():
                padding = arch['padding']
            if 'activation' in arch.keys():
                if arch['activation'] == 'sigmoid':
                    activation = tf.nn.sigmoid
            stride = 1
            if 'stride' in arch.keys():
                stride = arch['stride']
            weights_initializer = tf.truncated_normal_initializer(stddev=0.05)
            if not branch:
                print ('not branch')
                nets[i] = slim.layers.conv2d(nets[i-1], num_outputs=num_filters,kernel_size=[filter_size, filter_size], weights_initializer=weights_initializer, padding=padding, scope='conv'+str(i), stride=stride, weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, activation_fn=activation)
            elif branch:
                print ('branch')
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        print ('start branch...',j)
                        nets[i][j] = slim.layers.conv2d(nets[i - 1], num_outputs=num_filters,
                                             kernel_size=[filter_size, filter_size],
                                             weights_initializer=weights_initializer, padding=padding, weights_regularizer=slim.l2_regularizer(0.001),
                                             scope='conv' + str(i)+str(j), stride=stride, reuse=reuse, activation_fn=activation)
                    else:
                        print ('not start branch')
                        nets[i][j] = slim.layers.conv2d(nets[i - 1][j], num_outputs=num_filters,
                                                        kernel_size=[filter_size, filter_size],
                                                        weights_initializer=weights_initializer, padding=padding, weights_regularizer=slim.l2_regularizer(0.001),
                                                        scope='conv' + str(i)+str(j), stride=stride, reuse=reuse, activation_fn=activation)

        elif layer_type == 'fully_connected':
            num_outputs = arch['num_outputs']
            activation == tf.nn.relu
            if arch['activation'] == 'sigmoid':
                activation = tf.nn.sigmoid
            elif arch['activation'] =='linear':
                activation = None
            print ('adding fully connected layer...', i, ' with ', num_outputs, ' branching is ', branch, 'start branch is : ', str(start_branch))

            if not branch:
                print ('not branch')
                nets[i] = slim.layers.fully_connected(nets[i-1], num_outputs=num_outputs, scope='fc'+str(i),activation_fn=activation, reuse=reuse)
            elif branch:
                print ('branch')
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        print ('start branch..')
                        nets[i][j] = slim.layers.fully_connected(nets[i-1], num_outputs=num_outputs, scope='fc'+str(i)+str(j),activation_fn=activation, reuse=reuse)
                    else:
                        print ('not start branch')
                        nets[i][j] = slim.layers.fully_connected(nets[i-1][j], num_outputs=num_outputs,
                                                                 scope='fc' + str(i)+str(j), activation_fn=activation,
                                                                 reuse=reuse)

        elif layer_type == 'AvgPool2D':
            if not branch:
                nets[i] = slim.layers.avg_pool2d(nets[i-1], [arch['pool_size'], arch['pool_size']])
            elif branch:
                nets[i] = [None, None , None, None, None, None]
                for j in range(6):
                    if start_branch:
                        nets[i][j] = slim.layers.avg_pool2d(nets[i-1], [arch['pool_size'], arch['pool_size']])
                    else:
                        nets[i][j] = slim.layers.avg_pool2d(nets[i-1][j], [arch['pool_size'], arch['pool_size']])

        elif layer_type == 'maxpool2D':
            print ('adding maxpoo2D...', i)
            if not branch:
                nets[i] = slim.layers.max_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
            elif branch:
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        nets[i][j] = slim.layers.max_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
                    else:
                        nets[i][j] = slim.layers.max_pool2d(nets[i - 1][j], [arch['pool_size'], arch['pool_size']])


        elif layer_type == 'flatten':
            if not branch:
                nets[i] = slim.layers.flatten(nets[i-1], scope='flatten'+str(i))
            elif branch:
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    if start_branch:
                        nets[i][j] = slim.layers.flatten(nets[i-1], scope='flatten'+str(i)+str(j))
                    else:
                        nets[i][j] = slim.layers.flatten(nets[i-1][j], scope='flatten' + str(i)+str(j))
        elif layer_type == 'dropout':
            if not branch:
                nets[i] = tf.nn.dropout(nets[i-1], arch['value'], seed=SEED)
            elif branch:
                nets[i] = [None, None, None, None, None, None]
                for j in range(6):
                    nets[i][j] = tf.nn.dropout(nets[i-1][j], arch['value'], seed=SEED)

    return nets[i]

data_path = 'training-data'
#train_files = [train_file1]

def run_model():
    tf.compat.v1.reset_default_graph()

    global batch_size, learning_rate, architecture, num_epochs, IMAGE_SIZE

    eval_data = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    print ('building evaluation graph')
    eval_prediction = model_slim(eval_data, architecture,train=False)


    saver = tf.train.Saver()

    save_path_ = os.path.join(save_dir, 'model.ckpt')
    
    with tf.Session() as sess:
        
        ckpt = tf.train.get_checkpoint_state(save_path_)#读取训练好的参数
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('loading sucess,global-step is %s' % global_step)
        else:
            print('No checkpoint file found')
        saver.restore(sess,save_path_)
        
        for i in range(60):
            image_input = train_read1(i,1)[0]
            conv1_g = sess.run(tf.get_default_graph().get_tensor_by_name('conv1/Relu:0'),feed_dict={eval_data: image_input})
            #result = sess.run(eval_prediction,feed_dict={eval_data: image_input})
            for j in range(2):
                temp = conv1_g[:,:,:,j]
                temp = np.reshape(temp,[60,60])
                #im = Image.fromarray(temp)
                #im = im.convert('L')
                #im.save("./filters/conv1_weights_" + str(i) + ".png")
                #im.save("./real1/conv1_Relu_" + str(i) + "_2.png")
                #plt.imsave("./results6/conv6_weights_" + str(i) + ".png",temp,cmap = 'Greys')
                plt.imsave("./compare/conv1_Relu_" + str(j) + "_" + str(i) + ".png",temp)
            #print(result)

    return


run_model()
