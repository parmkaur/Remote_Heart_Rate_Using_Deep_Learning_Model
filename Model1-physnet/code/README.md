Model 1 - Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks

Requirements:

The codes were tested with Python 3.6.

pip install torch
pip install opencv-python
pip install matplotlib
pip install scipy
pip install numpy
pip install pandas

## Usage:

Data Preprocessing: 

1. Make a Training and Testing folder and put all the image frames into the Training/ Testing folder named as 1.jpg, 2.jpg and so on. 
2. Input image frames need to of size 128*128 with three channels.
3. Put all the corresponding labels of the image frames in csv file 

To Train the model:
python train.py

To Test the model:
python test.py

It is just for research purpose, and commercial use is not allowed.

If you use the PhysNet please cite:

@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {Proc. BMVC},
    year = {2019}
}