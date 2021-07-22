from interpretation_tools import feature_map

from read_h5 import imread

image_input, ground_truth = imread(0, 1, "highlin.h5")

feature_map(image_input, 1, 0, grey=False)