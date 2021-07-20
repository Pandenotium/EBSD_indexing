
import sys, traceback, os
import re, math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim, time


from tensorflow.python import debug as tf_debug

import matplotlib.pyplot as plt

from img3 import train_read,fc_read


from PIL import Image

from pca import PCA

batch_size = 64
architecture= 'infile'
reg_W = 0.

patience = 100
filter_size = 3
IMAGE_SIZE = 60
NUM_CHANNELS = 1
stride = 1

save_dir = os.getcwd()
#save_dir = './temp_200503201834/temp_200503201834'
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
        
        grp = 20
        num = 10
        data = np.zeros(grp)
        err = np.zeros(grp)
        grt = np.zeros(grp)
        
        lbst = np.zeros(num)
        datat = np.zeros(grp)
        
        for i in range(grp):
            for j in range(num):
                image_input,label = fc_read(i*10+j,1)
                pr = sess.run(eval_prediction,feed_dict={eval_data: image_input})
                lbst[j] = label[0][3]
                datat[j] = pr[3][0]
            grt[i] = label[0][3]
        
            sqe = np.zeros(num)
            for k in range(num):
                sqe[k] = (datat[k] - lbst[k])**2
            
            mse = np.sqrt(sum(sqe)/num)
            avg = sum(datat)/num
            
            data[i] = avg
            err[i] = mse
            
        plt.plot(grt,grt,color='green')      
        plt.errorbar(grt, data, yerr=err,fmt = 'o:', ecolor='red',elinewidth=1,ms = 3,capsize=2)

        plt.xlabel('ground truth')
        plt.ylabel('prediction')
        plt.show()
        
    return

run_model()

