##################################################################################
# Program: CNN for coconut tree 
#          identification
# Inputs:  6 X 6 SAR image patches, KMeans Tree Mask(kmeans_Trees_Mask.jpg) 
# Output: Identified coconut (CNNOutput.jpg)
###################################################################################

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
class Coco:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        #First layer
        model.add(Conv2D(4, kernel_size=(3, 3), padding="same", activation='relu', input_shape=inputShape))
        #Second layer
        model.add(Conv2D(8, kernel_size=(3, 3), padding="same", activation='relu', strides=1))
        #Third layer
        model.add(Conv2D(16,kernel_size=(3, 3), padding="same",activation='relu', strides=1))
        #Fourth layer
        model.add(Conv2D(32, kernel_size=(3, 3), padding="same",activation='relu', strides=1))
        #Fifth layer
        model.add(Flatten())
        model.add(Dense(1152))
        model.add(Activation('relu')) 
        #Output layer 
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

# set the matplotlib backend for saving figures
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
EPOCHS = 100
INIT_LR = 0.01
BS = 10


# initialize the data and labels
print("[INFO] loading images...")
XTrain = []
Trlabels = []


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("TrainData")))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = img_to_array(image)
    XTrain.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label=='Coconut':
       Trlabels.append(0) 
    elif label=='Othertrees':
       Trlabels.append(1)
    Telabels = []


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("Test")))
random.seed(42)
random.shuffle(imagePaths)
Xtest = np.zeros((len(imagePaths),6,6,2))


# loop over the input images
aa=0
for imagePath in range(1,len(imagePaths)):
    image = cv2.imread("Test/"+str(imagePath)+".jpg")
    image = img_to_array(image)
    Xtest[aa,:,:,:]=image
    aa=aa+1


# scale the raw pixel intensities to the range [0, 1]
XTrain = np.array(XTrain, dtype="float") / 255.0
Trlabels = np.array(Trlabels)
Xtest = np.array(Xtest, dtype="float") / 255.0
Trlabels1 = to_categorical(Trlabels, num_classes=2)

# initialize the model
print("[INFO] compiling model...")
model = Coco.build(width=6, height=6, depth=2, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


imgg=cv2.imread("kmeans_Trees_Mask.jpg")
wholeImg1=cv2.resize(imgg,(1572,1992))
final_img=np.zeros((wholeImg1.shape[0],wholeImg1.shape[1],3))
import tensorflow as tf 
model = tf.keras.models.load_model('CocoNet.h5')
pred_test = model.predict(Xtest)
y_classes = pred_test.argmax(axis=-1)
ii=0;
dd=0;
while ii<=wholeImg1.shape[0]-1:
    jj=0;
    while jj<=wholeImg1.shape[1]-1:
        imD1=cv2.imread('Test/'+str(dd+1)+'.jpg');
        if y_classes[dd]==0:
            final_img[ii:ii+6,jj:jj+6,0]=255;
            final_img[ii:ii+6,jj:jj+6,1]=0;
            final_img[ii:ii+6,jj:jj+6,2]=0;
        else:
            final_img[ii:ii+6,jj:jj+6,0]=0
            final_img[ii:ii+6,jj:jj+6,1]=0
            final_img[ii:ii+6,jj:jj+6,2]=0
            
        dd=dd+1;
        jj=jj+6;
        ii=ii+6;


# Identified coconut farms stored in a file
cv2.imwrite('CNNOutput.jpg', final_img) 



