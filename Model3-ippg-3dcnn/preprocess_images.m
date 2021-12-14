%Matlab script to convert image frames extracted from ubfc videos to gray
%scale and resize them to 25*25
clear all
clc
folder = 'all_video_frames/'; %input the extracted video frames path
file = fullfile(folder, '*.jpg');
filename=dir(file) 
for i = 1:numel(filename)
    f=fullfile(folder,filename(i).name)
    img=imread(f); %read the image
    figure(1),imshow(img)
    gray_image = rgb2gray(img); %convert image to gray scale
    figure(2), imshow(gray_image)
    resized_image=imresize(gray_image,[25,25]); %resize gray scale images to 25*25 dimmenisonal images
    figure(3),imshow(resized_image)
    path=strcat('C:\Users\parm0\out_images\', filename(i).name);
    imwrite(resized_image, path); %to write the images into specified path above
end