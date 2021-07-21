from interpretation_tools import predict

from read_h5 import test_read

image_input = test_read(0, 1)

predict(image_input)