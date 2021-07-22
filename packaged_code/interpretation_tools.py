import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from model import EBSDIndexingModel as model

import copy


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