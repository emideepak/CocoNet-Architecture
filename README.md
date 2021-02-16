# CocoNet Architecture for Coconut tree classification and change detection

A hierarchical Deep Neural Network Model which runs on CNN and trained to Detect coconut trees.

A change detection result was also provided to mark the changes before and after a calamity. 

## Cloning the repository

Clone the git repository by running git bash in your computer and run the following command

`https://github.com/emideepak/CocoNet-Architecture.git`

Or click on the download button and extract the zip file

## Create an Anaconda Environment

Run the following command in your conda prompt

`conda install tensorflow keras opencv`

`import tensorflow`

`import keras`

`import cv2`

## Installing Dependencies

Make sure that you have latest python version and type the following command in your anaconada prompt

`pip install easy_gui os`

`conda install imutils scikit-learn`

Steps to execute the code :

1. Initial Unsupervised KMeans clustering is performed in SNAP Geoprocessing tool.

2. The vegetation area alone extracted with conversionImage.m MATLAB file.

3. Supervised classification with trained CNN is performed with CocoNet.py python file.

4. Change detection is performed with Changedetection.m MATLAB file.

5. Original preprocessed image, KMeans results and other inputs are provided in its corresponding folders seperately. 








