import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from model import EBSDIndexingModel as model

import copy

import PIL.Image as PI
from cv2 import pyrUp
import cv2


def predict(image_input):
    '''
    This function outputs the predicted euler angles and pc values of a EBSD pattern.

    :param image_input: input images of the size (n,60,60,1) in format np.array, which
     can be generated by file read_h5. n = 1 is recommended.

    :return: n vectors (needs to be proved) of length 6, the first 3 elements are phi1,
     psi, phi2 in degrees and the last 3 elements are pcx, pcy, pcz.
    '''
    model_i = model()
    eval_data, eval_prediction, saver, save_path_ = model_i.initialize()

    with tf.Session() as sess:

        saver.restore(sess, save_path_)

        result = sess.run(eval_prediction, feed_dict={eval_data: image_input})
        for j in range(6):
            result[j] = result[j].tolist()
            result[j] = result[j][0][0]
            if j <= 2:
                result[j] = result[j] * 180 / np.pi

    return result


def feature_map(image_input, layer, filter, save=False, draw = True, grey=True, format='png'):
    '''
    This function print the feature map of certain layer and filter of
    certain EBSD pattern to the console.

    :param image_input: input image of the size (1,60,60,1) in format
     np.array, which can be generated by file read_h5.

    :param layer: integer, indicates the convolutional layer. range: 1-4, 6-9

    :param filter: integer, indicates the filter. range: for layer 1-2: 256,
     for layer 3-4: 512, for layer 6-7: 256, for layer 8-9: 96

    :param save: boolean, whether to save the feature map.

    :param draw: boolean, indicates whether to draw the maps.

    :param grey: boolean, whether to view and save the map in greys.

    :param format: string, indicates the format which you want to save the map

    :return: np.array, the feature map matrix. the size depends on layer
    '''
    model_i = model()
    eval_data, eval_prediction, saver, save_path_ = model_i.initialize()

    with tf.Session() as sess:

        saver.restore(sess, save_path_)

        f_maps = sess.run(tf.get_default_graph().get_tensor_by_name('conv' + str(layer) + '/Relu:0'),
                          feed_dict={eval_data: image_input})

        f_map = f_maps[:, :, :, filter]

        if layer in range(1, 5):
            f_size = 60
        if layer in range(6, 10):
            f_size = 30

        f_map = np.reshape(f_map, [f_size, f_size])

        im = Image.fromarray(f_map)

        cmap = 'viridis'
        if grey:
            im = im.convert('L')
            cmap ='Greys'

        if draw:
            plt.imshow(f_map, cmap = cmap)

        if save:
            im.save("conv" + str(layer) + "_" + str(filter) + "." + format)

    return f_map

def heatmap(image_input, m_size, d_type, save = False, draw = True, format = 'png'):
    '''
    This function calculates the difference heatmap of 6 parameters of certain
    EBSD pattern.

    :param image_input: input image of the size (1,60,60,1) in format np.array,
     which can be generated by file read_h5.

    :param m_size: integer of value 3 or 5, indicates the size of the mask

    :param d_type: 'abs' or 'sqr', whether to use absolute or square difference

    :param save: boolean, indicates whether to save the maps. saved maps are
     named as "param_maskSize_differenceType.format"

    :param draw: boolean, indicates whether to draw the maps.

    :param format: string, indicates how to save the maps.

    :return: a list contains 6 maps of 6 parameters. the order is phi1, psi,
     phi2, pcx, pcy, pcz.
    '''
    assert m_size in [3, 5]
    assert d_type in ['abs', 'sqr']

    names = ['phi1', 'psi', 'phi2', 'pcx', 'pcy', 'pcz']
    ims = []

    model_i = model()
    eval_data, eval_prediction, saver, save_path_ = model_i.initialize()

    with tf.Session() as sess:
        saver.restore(sess, save_path_)

        true = sess.run(eval_prediction, feed_dict={eval_data: image_input})
        avg = np.average(image_input)
        im = [[0 for col in range(60)] for row in range(60)]

        half_size = 1 if m_size == 3 else 2

        for p in range(6):
            for i in range(60):
                for j in range(60):
                    temp = copy.deepcopy(image_input)
                    temp = np.reshape(temp, [60, 60])

                    for n in range(max(0, i - half_size), min(60, i + 1 + half_size)):
                        for m in range(max(0, j - half_size), min(60, j + 1 + half_size)):
                            temp[n, m] = avg

                    temp = np.reshape(temp, [1, 60, 60, 1])
                    result = sess.run(eval_prediction, feed_dict={eval_data: temp})

                    if d_type == 'abs':
                        im[i][j] = abs(result[p][0][0] - true[p][0][0])
                    elif d_type == 'sqr':
                        im[i][j] = (result[p][0][0] - true[p][0][0]) ** 2

            ims.append(im)

            if save:
                plt.imsave(names[p] + '_' + m_size*2 + '_' + d_type + '.' + format, im, cmap = 'hot')

    if draw:
        for p in range(6):
            plt.subplot(3,2,p)
            plt.imshow(ims[p], cmap = 'hot')
            plt.xlabel(names[p])
        plt.title("difference heatmap of mask size " + m_size + '*' + m_size + " and difference type " + d_type)
        plt.show()

    return ims

def m_grad_cam(image_input, param, save = False, draw = True, format = 'png'):
    '''
    This function calculates the modified Grad-CAM heat map of certain parameter
    as it was mentioned in the paper.

    :param image_input: input image of the size (1,60,60,1) in format np.array,
     which can be generated by file read_h5.

    :param param: integer of 0 to 5, which indicates the 6 parameters.

    :param save: boolean, indicates whether to save the map.

    :param draw: boolean, indicates whether to print the map on the console.

    :param format: string, indicates how to save the map

    :return: np.array of size 15*15, which is the map.
    '''
    assert param in range(6)
    names = ['phi1', 'psi', 'phi2', 'pcx', 'pcy', 'pcz']

    model_i = model()
    eval_data, eval_prediction, saver, save_path_ = model_i.initialize()

    with tf.Session() as sess:
        saver.restore(sess, save_path_)

        conv_g = sess.run(tf.get_default_graph().get_tensor_by_name('MaxPool2D_1/MaxPool:0'),
                          feed_dict={eval_data: image_input})
        grads = tf.gradients(tf.get_default_graph().get_tensor_by_name('fc16' + str(param) + '/BiasAdd:0'),
                             tf.get_default_graph().get_tensor_by_name('MaxPool2D_1/MaxPool:0'))
        grads_g = sess.run(grads, feed_dict={eval_data: image_input})

        final = np.zeros([15, 15])
        grads_g = np.asarray(grads_g)

        for i in range(conv_g.shape[3]):
            temp = conv_g[:, :, :, i]
            temp = np.reshape(temp, [15, 15])
            temp_g = np.array(grads_g[:, :, :, :, i])
            final = final + np.reshape(temp_g * temp, [15, 15])

        if draw:
            plt.imshow(final)

        if save:
            plt.imsave("m_grad-cam_" + names[param] + '.' + format, final)

    return final

def masked_f_map(image_input, layer, filter, r_prop = 0.95, save=False, draw = True, format='png'):
    '''
    This function apply a mask to a feature map, reserve the most activated pixels and
    blend the mask and the input image.

    :param image_input: input image of the size (1,60,60,1) in format np.array,
     which can be generated by file read_h5.

    :param layer: integer, indicates the convolutional layer. range: 1-4, 6-9

    :param filter: integer, indicates the filter. range: for layer 1-2: 256,
     for layer 3-4: 512, for layer 6-7: 256, for layer 8-9: 96

    :param r_prop: float, between 0 and 1, indicates the proportion of the removed
     pixels. (example, if r_prop = 0.95, then the 5% most activated pixels will be
     of value 1 and the others 0)

    :param save: boolean, whether to save the feature map.

    :param draw: boolean, indicates whether to draw the maps.

    :param format: string, indicates the format which you want to save the map

    :return: the masked input image.
    '''
    assert r_prop >= 0 and r_prop <= 1

    map = feature_map(image_input, layer, filter, save = False, draw = False, grey= False, format = format)

    temp = copy.deepcopy(map)

    s = copy.deepcopy(map)
    s = np.reshape(s,[map.shape[0]*map.shape[1]])
    th = s[int(s.size*r_prop)]

    for j in range(map.shape[0]):
        for k in range(map.shape[1]):
            if abs(temp[j, k]) >= th:
                temp[j, k] = 1
            else:
                temp[j, k] = 0

    img = image_input * 255
    img = PI.fromarray(img.astype(np.uint8))
    img = img.convert('RGBA')

    temp = pyrUp(np.asarray(temp))
    temp = cv2.resize(temp, (60, 60), interpolation=cv2.INTER_AREA)
    temp = PI.fromarray(temp)
    temp = temp.convert('RGBA')
    final = PI.blend(img, temp, 0.6)

    if save:
        final.save("m_conv_" + str(r_prop*100) + "_" + str(layer) + "_" + str(filter) + "." + format)

    if draw:
        plt.imshow(final)

    return final