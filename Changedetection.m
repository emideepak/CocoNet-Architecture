% For identifying the changes happened on coconut trees 
A = imread('Beforemap.jpg');
B = imread('Aftermap.jpg');
imshowpair(A,B,'diff')
imshow(C)
imwrite(C,'Final_Change_Map.png');