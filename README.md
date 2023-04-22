# Classifier Training with Python
This repository contains Python code for training two different classifiers, Random Forest and ANN classifiers, and testing their accuracy using three different feature extraction tools: HOG, SIFT, and SURF.

# Data
The data used for testing the accuracy of the classifiers consists of 606 images of cats, dogs, and horses, with each category containing 202 images. The data is split into 70% training data and 30% test data.

# Classifiers
The two classifiers used for this project are Random Forest and ANN classifiers. Random Forest is an ensemble learning method that builds a multitude of decision trees and then combines their outputs to make a final prediction. ANN, on the other hand, is a deep learning algorithm inspired by the structure of the human brain.

# Feature Extraction Tools
The accuracy of the classifiers is tested using three different feature extraction tools: HOG, SIFT, and SURF. HOG (Histogram of Oriented Gradients) is a feature extraction method used for object detection in computer vision. SIFT (Scale-Invariant Feature Transform) is a feature extraction algorithm that detects and describes local features in images. SURF (Speeded-Up Robust Features) is a feature detection algorithm that identifies and describes local features in images.

# Results
After testing the accuracy of the classifiers with the different feature extraction tools, the results show that the accuracy was highest when using HOG and lowest when using SURF.

# Conclusion
The code in this repository can be used as a starting point for training classifiers and testing their accuracy using different feature extraction tools. The results can be used to determine which feature extraction tool is best suited for a specific task.
