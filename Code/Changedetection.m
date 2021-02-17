%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Program:To identify the changes due to cyclone on coconut trees
Input: CocoNet output before cyclone (Before_Cyclone_CNNOutput.jpg)
       CocoNet output after cyclone (After_Cyclone_CNNOutput.jpg)
Output: Change map of coconut trees (Change_Map.png)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = imread('Before_Cyclone_CNNOutput.jpg');
B = imread('After_Cyclone_CNNOutput.jpg');
imshowpair(A,B,'diff')
imshow(C)
imwrite(C,'Change_Map.png');