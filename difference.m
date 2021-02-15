A = imread('Finalafter.jpg');
B = imread('Finalbefore.jpg');
imshowpair(A,B,'diff')
imshow(C)
imwrite(C,'different.png');