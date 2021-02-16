%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Program:To extract the trees and mask out the other terrain classes
Input: KMeans output before cyclone (KMeans_Before_Cyclone.tif)
       KMeans output after cyclone (KMeans_After_Cyclone.tif)
Output: Change map of coconut trees (KMeans_VegetationMask.jpg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
clc;

imgg=imread('KMeans_Before_Cyclone.tif');
figure,imshow(imgg(:,:,1:3));
for ii=1:size(imgg,1)
    for jj=1:size(imgg,2)
        if imgg(ii,jj,1)>230
            img1(ii,jj,1)=1;
            img1(ii,jj,2)=1;
            img1(ii,jj,3)=1;
            
        else
            img1(ii,jj,1)=0;
            img1(ii,jj,2)=0;
            img1(ii,jj,3)=0;
        end
    end
end
figure,imshow(img1);
imwrite(img1,['KMeans_VegetationMask.jpg']);
img1=imresize(img1,[1992,1572]);
dd=1;
for ii=1:6:size(img1,1)
    for jj=1:6:size(img1,2)
        imm=img1(ii:ii+5,jj:jj+5,1:3);
        imwrite(imm,['Test/',num2str(dd),'.jpg']);
        dd=dd+1;
    end
end


        

            
            