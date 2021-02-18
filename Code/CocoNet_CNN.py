###############################################
# Program: CNN training code for coconut tree 
#          identification
# Inputs:  6 X 6 SAR image patches stored in
#                      'Sample_Dataset' folder 
# Output: Trained model (CocoNet.h5)
###############################################
 

# import necessary packages
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
import matplotlib
matplotlib.use("Agg")
# import necessary packages
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
data = []
labels = []


# grab the image path and randomly shuffle them
imagePaths = sorted(list(paths.list_images("Sample_Dataset")))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = img_to_array(image)
    image = image[:,:,:2]
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label=='COCONUT':
       labels.append(0) 
    elif label=='OTHERTREES':
       labels.append(1)


# Normalize the pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(np.shape(data))


# partition the data: 80% for training, 20% for validation
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)


# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, horizontal_flip=True, 

# initialize the model
print("[INFO] compiling model...")
model = Coco.build(width=6, height=6, depth=2, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)
print("[INFO] serializing network...")
model.save("CocoNet.h5")
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import easygui
import cv2
path1 = easygui.fileopenbox()
image = cv2.imread(path1)
orig = image.copy()
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the CNN model
print("[INFO] loading network...")
model = load_model("CocoNet.h5")
# classify the input image
pred_test = model.predict(image)
y_classes = pred_test.argmax(axis=-1)
print(y_classes)
if y_classes==0:
   print('COCONUT')
elif y_classes==1:
   print('OTHERTREES')
pred_test = model.predict(trainX)
y_classes = pred_test.argmax(axis=-1)
# Performance Measures
from sklearn.metrics import classification_report,confusion_matrix
def perf_measure(y_actual, y_Prediction):
    TP,TN,FP,FN=1,1,1,1
    for i in range(y_Prediction.shape[0]): 
        if y_actual[i]==1 and y_Prediction[i]==1:
           TP += 1
        if y_Prediction[i]==0 and y_actual[i]==0:
           TN += 1
        if y_actual[i]==1 and y_Prediction[i]==0:
           FP += 1
        if y_Prediction[i]==0 and y_actual[i]==1:
           FN += 1
    return(TP, FP, TN, FN)
labels1=labels
labels1=np.where(labels1==0, 1, labels1)
labels1=np.where(labels1==1, 0, labels1) 
y_classes1=y_classes
y_classes1=np.where(y_classes1==0, 1, y_classes1)
y_classes1=np.where(y_classes1==1, 0, y_classes1) 
[TP, FP, TN, FN]=perf_measure(labels1,y_classes1)
print('TP',TP)
print('TN',TN)
print('FP',FP)
print('FN',FN)


# Sensitivity, hit rate, recall, or true positive rate
SEN = TP/(TP+FN)*100
print('Sensitivity',SEN)
SPE = TN/(TN+FP)*100
print('Specificity',SPE)


# Precision or positive predictive value
prec = TP/(TP+FP)*100
print('Precision',prec)
print('Recall',SEN)


# Negative predictive value
NPV = TN/(TN+FN)
print('NPV',NPV)


# Fall out or false positive rate
FPR = FP/(FP+TN)
print('FPR',FPR)


# False negative rate
FNR = FN/(TP+FN)
print('FNR',FNR)


# False discovery rate
FDR = FP/(TP+FP)
print('FDR',FDR)


# Overall accuracy
ACC = ((TP+TN)/(TP+FP+FN+TN))*100
print('Accuracy',ACC)
Coeff_a = ((TP + FN) * (TP + FP)) / (TP + FN + FP + TN)
Coeff_b = ((FP + TN) * (FN + TN)) / (TP + FN + FP + TN)
expec_agree = (Coeff_a + Coeff_b) / (TP + FN + FP + TN)
obs_agree = (TP + TN) / (TP + FN + FP + TN)
KAPPACoeff = (obs_agree - expec_agree) / (1 - expec_agree)  
print('Kappa Coeffient',KAPPACoeff)
Jaccard=TP/(TP+FP+FN)
print('Jaccard Coeffient',Jaccard)
FAR=100-SPE
print('False Acceptance Rate',FAR)
FRR=100-SEN
print('False Rejection Rate',FRR)
FScore=(2*TP)/((2*TP)+FN+FP)
print('F-Measure',FScore)
Overall=[SEN,SPE,prec,NPV,FPR,FNR,FDR,ACC,KAPPACoeff,Jaccard,FAR,FRR,FScore]

print("[INFO] serializing network...")
model.save("Coconet.h5")
import pandas as pd
hist_df = pd.DataFrame(H.history) 


# save to csv:  
hist_csv_file = 'Coconet.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
Hist_file=np.array(pd.read_csv('CoconetHistory.csv'))
plt.figure(figsize=(9, 3))




