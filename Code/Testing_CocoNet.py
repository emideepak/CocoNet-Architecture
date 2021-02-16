############################################################
# Program: Testing phase
# Inputs: K-Means masked SAR image 
#                    (kmeansMaskedSAR.jpg)
# Output: Coconut trees shown in green colour 
#                    (CNNOutput.jpg) 
############################################################
# import necessary packages
from Code import CocoNet_CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
imgg=cv2.imread("kmeansMaskedSAR.jpg")
wholeImg1=cv2.resize(imgg,(1572,1992))
final_img=wholeImg1
pred_test = model.predict(Xtest)
y_classes = pred_test.argmax(axis=-1)
ii=0;
dd=0;
while ii<=wholeImg1.shape[0]-1:
    jj=0;
    while jj<=wholeImg1.shape[1]-1:
        imD1=cv2.imread('Test/'+str(dd+1)+'.jpg');
        if y_classes[dd]==0:
            final_img[ii:ii+6,jj:jj+6,0]=0;
            final_img[ii:ii+6,jj:jj+6,1]=255;
            final_img[ii:ii+6,jj:jj+6,2]=0;
        else:
            final_img[ii:ii+6,jj:jj+6,0]=imD1[:,:,0]
            final_img[ii:ii+6,jj:jj+6,1]=imD1[:,:,1]
            final_img[ii:ii+6,jj:jj+6,2]=imD1[:,:,2]
       
        dd=dd+1;
        jj=jj+6;

    ii=ii+6;


# Identified coconut farms stored in a file
cv2.imwrite('CNNOutput.jpg', final_img) 
