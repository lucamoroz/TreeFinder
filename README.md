Project from "Computer Vision" course - University of Padua.

The assignment required to develop an application capable of detecting trees inside an image by drawing a rectangle around them.

The following project is based on: 
- Bag of Visual Words: to extract a normalized histogram descriptor of an image using a trained dictionary based on SIFT features (the dictionary is obtained by running kmeans on the full set of features extracted from the training dataset)
- SVM (RBF kernel): a binary classifier that takes in input the Bag of Visual Words descriptor and decides if the descriptor is a tree.
- Sliding Window: to detect trees at different scales.
