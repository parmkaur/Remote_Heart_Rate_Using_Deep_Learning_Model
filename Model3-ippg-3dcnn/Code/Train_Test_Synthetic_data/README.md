# 3D convolutional neural networks for remote pulse rate measurement and mapping from facial video

## Requirements
The codes were tested with Python 3.5/3.6 and Tensorflow + Keras frameworks.

Different packages must be installed to properly run the codes : 

pip install tensorflow (or tensorflow-gpu)
pip install opencv-python
pip install matplotlib
pip install scipy

## Usage

train
main.py is a standalone program that creates the network architecture and the synthetic data before launching training. The code saves the architecture (.json) and weights (.h5) as well as statistics (training / validation loss and accuracy) at the end of the procedure.

Variables and hyperparameters

NB_VIDEOS_BY_CLASS_TRAIN and NB_VIDEOS_BY_CLASS_VALIDATION: number of synthetic videos (a video is a tensor of size 25 x 25 x 60) that will be generated and used for training and validation.
EPOCHS: 5000 by default.
CONTINUE_TRAINING: False to start training from scratch (weights are randomly initialized), True to resume training.
SAVE_ALL_MODELS: False to save only the model that presents the highest validation accuracy.


predict
main.py first display an UI that allows selection of the folder that contains the images. The program will load all the files in the selected directory so it must contain only image files. (e.g. 0000.png; 0001.png; 0002.png...).

Variables and hyperparameters

DIR_SAVE: save directory. The predictions will be saved in a Matlab format (.mat) in this path.
USE_RANDOM_PIXEL_LOCATION: 1 by default. 0 to disable shuffle of pixel positions.

## Reference

Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, **3D convolutional neural networks for remote pulse rate measurement and mapping from facial video**, *Applied Sciences*, vol. 9, n° 20, 4364 (2019). [Link](https://www.mdpi.com/2076-3417/9/20/4364)


