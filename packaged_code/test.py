from interpretation_tools import predict

from read_h5 import imread

image_input, ground_truth = imread("hignlin.h5", 0, 1)

predict(image_input)