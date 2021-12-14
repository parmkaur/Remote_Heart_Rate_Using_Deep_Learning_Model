'''
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
By Zitong Yu, 2019/05/05
If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019 
'''

#Import packages
import os
import math
import torch
import scipy.io
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _triple
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from model.model import *
from model.loss import *
from PIL import  Image
import matplotlib.pyplot as plt

# Declare Constants
Image_Height=128
Image_Width=128
Batch_Size=128
Channel=3
EPOCHS = 12 

#Load Training RPPG_data CSV File 
rppg_data_label=pd.read_csv('/content/drive/MyDrive/final_ppg_data.csv', sep=',',header=None) 
rppg_data_label = pd.DataFrame(rppg_data_label).to_numpy()
len_rppg_data_label = rppg_data_label.shape[0]
batch_num_count = math.floor(len_rppg_data_label/Batch_Size)
print('Input Image Dimensions Requirement= ' + str(Image_Width) + '*' + str(Image_Height) + '*' + str(Channel))
print('Number of rPPG labels loaded = ' + str(rppg_data_label.shape[0]))
print('batch_size= ' + str(Batch_Size))
print('no_of_iterations= ' + str(batch_num_count))

# Initialize frames_train and labels_train numpy array 
frames_train= np.zeros([batch_num_count,Channel,Batch_Size,Image_Height,Image_Width])
labels_train = np.zeros([batch_num_count, Batch_Size])
print('Initialized Input Video Frames Shape = ' + str(frames_train.shape))
print('Initialized rPPG labels Shape = ' + str(labels_train.shape))

# Load Training Video Frames
images_path = '/content/drive/MyDrive/final_input_images' #path to the video_frames directory images named as 1.jpg, 2.jpg, and so on
img_counter=0 
for i in range(batch_num_count):
          images = np.zeros([Batch_Size,Image_Height,Image_Width,Channel])
          for j in range(Batch_Size):
                  img_counter+=1
                  image = images_path + '/' + str(img_counter) + '.jpg'                
                  image = np.asarray(Image.open(image))                 
                  images[i,:,:,:] = image[:,:,:]
          frames_train[i,:,:,:,:] = np.rollaxis(images,axis=-1)
          labels_train[i,:] = np.reshape(rppg_data_label[i*Batch_Size:i*Batch_Size + Batch_Size],[1,Batch_Size])

print('Frames Train Generated')
print('RPPG Labels Train Generated')

#Call the defined model and loss function
model = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
criterion = Neg_Pearson()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #define optimizer with learning rate 0.001

#Training Loop
print('**********Start Training**************')
for epoch_num in range(EPOCHS):
  print('Epoch = ' + str(epoch_num+1))
  for i in range(batch_num_count):
      Input_Frames =np.reshape(frames_train[i,:,:,:,:], [1,Channel,Batch_Size,Image_Height,Image_Width])
      Target_rppg_labels = np.reshape(labels_train[i,:],[1,Batch_Size])
      Input_Frames = torch.from_numpy(Input_Frames).float()
      Target_rppg_labels = torch.from_numpy(Target_rppg_labels).float()
      #1. Inference the model
      rPPG, x_visual, x_visual3232, x_visual1616 = model(Input_Frames)
      #2. Normalized the Predicted rPPG signal and GroundTruth BVP signal
      rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize   
      Target_rppg_labels = (Target_rppg_labels-torch.mean(Target_rppg_labels)) /torch.std(Target_rppg_labels)    
      #3. Calculate the loss
      loss = criterion(rPPG,Target_rppg_labels)     
      loss_print = loss.detach().numpy() 
  print('Training loss=' + str(loss_print))
  #Update Paramaters
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  #Save the Model after every epoch
  torch.save(model.state_dict(), '/content/drive/My Drive/physnetmodel.pt')
  print('A model has been saved')
  print('Epoch=' + str(epoch_num+1) + '/' + str(EPOCHS) +" done")
   
