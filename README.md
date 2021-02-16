# CocoNet Architecture 

A two-phase hybrid machine learning framework (K-Means+CNN) for for Coconut tree identification and cyclonic damage assessment.

## Cloning the repository

Clone the git repository by running git bash in your computer and run the following command

`git clone https://github.com/emideepak/CocoNet-Architecture.git`

or click on the download button and extract the zip file

## Installing dependencies

On a PC running Windows,

Install `SNAP Geoprocessing tool` with Sentinel toolbox from  `https://step.esa.int/main/download/snap-download/`.

Install `MATLAB` with Image Processing toolbox from `https://www.mathworks.com/downloads/`.

Install `Anaconda` from `https://www.anaconda.com/products/individual` and run the following commands in your conda prompt.

`conda install tensorflow keras opencv`

`import tensorflow`

`import keras`

`import cv2`

`pip install easy_gui os`

`conda install imutils scikit-learn`

## Execution of codes 

1. Preprocess SAR images using SNAP Geoprocessing tool.

2. Perform unsupervised K-Means clustering using SNAP Geoprocessing tool.

3. Run `kmeansTreesSegmentation.m` on MATLAB.

4. Run `CocoNetCNN.py` python on Anaconda.

6. Run `Testing_CocoNet.py` python on Anaconda.

5. The above steps are executed for both before and after cyclone SAR images.

6. Run `ChangeDetection.m` on MATLAB.

The final change map is stored in `ChangeMap.png`








