# 3D convolutional neural networks for remote pulse rate measurement and mapping from facial video

## Requirements
The codes were tested with Python 3.5/3.6 and Tensorflow + Keras frameworks.

Different packages must be installed to properly run the codes : 

pip install tensorflow (or tensorflow-gpu)
pip install opencv-python
pip install matplotlib
pip install scipy

## Usage

Data Preprocessing: Convert all the image frame extracted from UBFC Dataset to gray scale images with dimensions 25*25. You can use the preprocess_images.m Matlab code to perform this task. 

Make Testing and Training folders with heart rate level and put the images with same heart rate level in the folder and images hsould be named as 1.jpg, 2.jpg and so on.

Also create a distinct labels csv file and put in all the heart rate labels that you are using for training and testing.

Note: Same heart rate levels should be used for training and testing. 
Also number of heart rate levels should be same for training and testing. 

##main.ipynb

main.ipynb is the standalone iternative python notebook file which you can use for training and testing the model.


## Reference

Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, **3D convolutional neural networks for remote pulse rate measurement and mapping from facial video**, *Applied Sciences*, vol. 9, n° 20, 4364 (2019). [Link](https://www.mdpi.com/2076-3417/9/20/4364)


