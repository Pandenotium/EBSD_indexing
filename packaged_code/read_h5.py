import h5py as h
import numpy as np

def imread(begin, n, file = "data.h5"):
    '''
    This function reads n EBSD patterns from the indicated file.

    :param begin: integer, indicates from which image the function reads
    the images. this parameter begins with 0.

    :param n: integer, indicates how many images the function reads

    :param file: string, indicates from which file the function reads
    the images. this string should contain ".h5"

    :return: im_result: the read n images in format np.array of the size
    (n,60,60,1)

    :return: y_result: true parameters (eul and pcs) of the read n images
    in format np.array of the size (n,6)
    '''
    im_result = np.zeros((n,60,60,1))
    y_result = np.zeros((n,6))
    file = h.File("../data/" + file,"r")
    for i in range(begin,begin+n):
        if i % 20 == 0:
            print('\r' + "Loading progress:%d/%d"%(i-begin,n),end = '',flush=True)
        im = file['/images'][i]
        im_result[i-begin] = im.reshape((60,60,1))
        im_result = im_result/255
        
        y = file['/lables'][i]
        y_result[i-begin] = y.reshape((6))
        
    return im_result, y_result