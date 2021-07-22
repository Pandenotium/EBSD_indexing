import os

import tensorflow as tf
import tensorflow.contrib.slim as slim


class EBSDIndexingModel:
    batch_size = 64
    architecture = 'infile'
    reg_W = 0.

    patience = 100
    filter_size = 3
    IMAGE_SIZE = 60
    NUM_CHANNELS = 1
    stride = 1

    save_dir = os.getcwd()

    conffile = None

    if architecture == 'infile':
        architecture = [{'layer_type': 'conv', 'num_filters': 256, 'input_channels': 1, 'filter_size': filter_size,
                         'border_mode': 'same', 'init': 'glorot_uniform', 'stride': stride, 'activation': 'relu',
                         'reg_W': reg_W},
                        {'layer_type': 'conv', 'num_filters': 256, 'stride': stride, 'input_channels': 256,
                         'filter_size': filter_size, 'border_mode': 'same', 'init': 'glorot_uniform',
                         'activation': 'relu',
                         'reg_W': reg_W},
                        # {'layer_type': 'maxpool2D', 'pool_size':2},
                        # {'layer_type':'dropout', 'value':0.5},
                        {'layer_type': 'conv', 'num_filters': 512, 'stride': stride, 'input_channels': 256,
                         'filter_size': filter_size,
                         'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                        {'layer_type': 'conv', 'num_filters': 512, 'stride': stride, 'input_channels': 512,
                         'filter_size': filter_size,
                         'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},
                        {'layer_type': 'maxpool2D', 'pool_size': 2},
                        # {'layer_type': 'dropout', 'value': 0.5},
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
                        {'layer_type': 'fully_connected', 'num_outputs': 8192, 'num_inputs': 96 * 15 * 15,
                         'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},
                        {'layer_type': 'fully_connected', 'num_outputs': 2048, 'num_inputs': 8192,
                         'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},

                        {'layer_type': 'fully_connected', 'num_outputs': 1024, 'num_inputs': 2048,
                         'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'branch': True},
                        {'layer_type': 'fully_connected', 'num_outputs': 256, 'num_inputs': 1024,
                         'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},

                        {'layer_type': 'fully_connected', 'num_outputs': 1, 'num_inputs': 256, 'activation': 'linear',
                         'reg_W': reg_W, 'init': 'glorot_uniform'}
                        ]

    SEED = 66478

    def model_slim(self, data, train=False):
        i = 0
        branch = False
        start_branch = False
        if train:
            reuse = None
        else:
            reuse = None
        nets = {}
        nets[0] = data
        for arch in self.architecture:
            i += 1
            layer_type = arch['layer_type']
            if ('branch' in arch.keys()) and arch['branch']:
                if not start_branch:
                    start_branch = True
                else:
                    start_branch = False
                branch = True
            else:
                start_branch = False
            if layer_type == 'conv':
                print('adding cnn layer..', i)
                num_filters = arch['num_filters']
                filter_size = arch['filter_size']
                border_mode = 'SAME'
                activation = tf.nn.relu
                if 'border_mode' in arch.keys():
                    border_mode = arch['border_mode']
                padding = border_mode
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
                    print('not branch')
                    nets[i] = slim.layers.conv2d(nets[i - 1], num_outputs=num_filters,
                                                 kernel_size=[filter_size, filter_size],
                                                 weights_initializer=weights_initializer, padding=padding,
                                                 scope='conv' + str(i), stride=stride,
                                                 weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse,
                                                 activation_fn=activation)
                elif branch:
                    print('branch')
                    nets[i] = [None, None, None, None, None, None]
                    for j in range(6):
                        if start_branch:
                            print('start branch...', j)
                            nets[i][j] = slim.layers.conv2d(nets[i - 1], num_outputs=num_filters,
                                                            kernel_size=[filter_size, filter_size],
                                                            weights_initializer=weights_initializer, padding=padding,
                                                            weights_regularizer=slim.l2_regularizer(0.001),
                                                            scope='conv' + str(i) + str(j), stride=stride, reuse=reuse,
                                                            activation_fn=activation)
                        else:
                            print('not start branch')
                            nets[i][j] = slim.layers.conv2d(nets[i - 1][j], num_outputs=num_filters,
                                                            kernel_size=[filter_size, filter_size],
                                                            weights_initializer=weights_initializer, padding=padding,
                                                            weights_regularizer=slim.l2_regularizer(0.001),
                                                            scope='conv' + str(i) + str(j), stride=stride, reuse=reuse,
                                                            activation_fn=activation)

            elif layer_type == 'fully_connected':
                num_outputs = arch['num_outputs']
                activation == tf.nn.relu
                if arch['activation'] == 'sigmoid':
                    activation = tf.nn.sigmoid
                elif arch['activation'] == 'linear':
                    activation = None
                print('adding fully connected layer...', i, ' with ', num_outputs, ' branching is ', branch,
                      'start branch is : ', str(start_branch))

                if not branch:
                    print('not branch')
                    nets[i] = slim.layers.fully_connected(nets[i - 1], num_outputs=num_outputs, scope='fc' + str(i),
                                                          activation_fn=activation, reuse=reuse)
                elif branch:
                    print('branch')
                    nets[i] = [None, None, None, None, None, None]
                    for j in range(6):
                        if start_branch:
                            print('start branch..')
                            nets[i][j] = slim.layers.fully_connected(nets[i - 1], num_outputs=num_outputs,
                                                                     scope='fc' + str(i) + str(j),
                                                                     activation_fn=activation,
                                                                     reuse=reuse)
                        else:
                            print('not start branch')
                            nets[i][j] = slim.layers.fully_connected(nets[i - 1][j], num_outputs=num_outputs,
                                                                     scope='fc' + str(i) + str(j),
                                                                     activation_fn=activation,
                                                                     reuse=reuse)

            elif layer_type == 'AvgPool2D':
                if not branch:
                    nets[i] = slim.layers.avg_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
                elif branch:
                    nets[i] = [None, None, None, None, None, None]
                    for j in range(6):
                        if start_branch:
                            nets[i][j] = slim.layers.avg_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
                        else:
                            nets[i][j] = slim.layers.avg_pool2d(nets[i - 1][j], [arch['pool_size'], arch['pool_size']])

            elif layer_type == 'maxpool2D':
                print('adding maxpool2D...', i)
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
                    nets[i] = slim.layers.flatten(nets[i - 1], scope='flatten' + str(i))
                elif branch:
                    nets[i] = [None, None, None, None, None, None]
                    for j in range(6):
                        if start_branch:
                            nets[i][j] = slim.layers.flatten(nets[i - 1], scope='flatten' + str(i) + str(j))
                        else:
                            nets[i][j] = slim.layers.flatten(nets[i - 1][j], scope='flatten' + str(i) + str(j))
            elif layer_type == 'dropout':
                if not branch:
                    nets[i] = tf.nn.dropout(nets[i - 1], arch['value'], seed=SEED)
                elif branch:
                    nets[i] = [None, None, None, None, None, None]
                    for j in range(6):
                        nets[i][j] = tf.nn.dropout(nets[i - 1][j], arch['value'], seed=SEED)

        return nets[i]

    data_path = 'training-data'

    def initialize(self):
        tf.compat.v1.reset_default_graph()

        eval_data = tf.placeholder(tf.float32, shape=(1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))
        print('building evaluation graph')
        eval_prediction = self.model_slim(eval_data, train=False)

        saver = tf.train.Saver()

        save_path_ = os.path.join(self.save_dir, 'model.ckpt')

        return eval_data, eval_prediction, saver, save_path_
