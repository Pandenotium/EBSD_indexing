# Directions for the organized codes

This directory contains the well organized codes.
They are packaged and can run under simple calls of functions.

##Introduction to the files

`model.py` is the very key file that contains the definition of our CNN
model and a corresponding function to load it. The other codes are all
based on this file.

`read_h5.py` contains the functions that load the EBSD patterns from the
`.h5` files and transform them into proper formats that the codes can
process.

`intepretation_tools.py` (not complete yet) contains functions that are used to interpret 
our model, such as making predictions, draw the feature map of certain
convolutional layer, etc.

`sample.py` (not ready yet) contains several samples of how to use the codes.

## Plan
### functions to add
masked feature maps, modified grad-cam, difference heat map, error test,
feature maps of a whole layer (optional)

### other notations
model parameters are not ready yet (too big to load), test data sets
are not ready yet