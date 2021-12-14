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
from numpy import asarray
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _triple
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from models.model import *
from models.loss import *
from PIL import  Image
import matplotlib.pyplot as plt

# Declare Constants
Image_Height=128
Image_Width=128
Mask_Height=64
Mask_Width=64
Channel=3
Batch_Size=64

#Load Testing RPPG_data CSV File 
rppg_data_label=pd.read_csv('/content/drive/MyDrive/final_ppg_data.csv', sep=',',header=None) 
rppg_data_label = pd.DataFrame(rppg_data_label).to_numpy()
len_rppg_data_label = rppg_data_label.shape[0]
batch_num_count = math.floor(len_rppg_data_label/Batch_Size)
print('Input Image Dimensions Requirement= ' + str(Image_Width) + '*' + str(Image_Height) + '*' + str(Channel))
print('Number of rPPG labels loaded = ' + str(rppg_data_label.shape[0]))
print('batch_size= ' + str(Batch_Size))
print('no_of_iterations= ' + str(batch_num_count))

# Initialize frames_train, mask_train and labels_train numpy array 
frames_test= np.zeros([batch_num_count,Channel,Batch_Size,Image_Height,Image_Width])
labels_test = np.zeros([batch_num_count, Batch_Size])
mask_test= np.zeros([batch_num_count,Batch_Size,Mask_Height,Mask_Width])
print('Initialized Input Video Frames Shape = ' + str(frames_test.shape))
print('Initialized Mask Images Shape = ' + str(mask_test.shape))
print('Initialized rPPG labels Shape = ' + str(labels_test.shape))


# Load Testing Video Frames and Mask Images
images_path = '/content/drive/MyDrive/final_input_images/' #path to the video_frames directory images named as 1.jpg, 2.jpg, and so on
Skin_mask_images_path = '/content/drive/MyDrive/Models/Model1-3dcnn/final_mask_images/' #path to the mask_images directory images named as 1.jpg, 2.jpg, and so on
img_counter=0 
images = np.zeros([Batch_Size,Image_Height,Image_Width,Channel])
mask_images = np.zeros([Batch_Size,Mask_Height,Mask_Width])

for batch_num in range(batch_num_count):
        for i in range(Batch_Size):
            img_counter = img_counter+1
            cropped_image = images_path +  str(img_counter) + '.jpg' 
            mask_image = Skin_mask_images_path + str(img_counter) + '.jpg'                
            cropped_images = asarray(Image.open(cropped_image))   
            mask_image = asarray(Image.open(mask_image))               
            images[i,:,:,:] = cropped_images[:,:,:]
            mask_images[i,:,:] = mask_image[:,:]
            print(img_counter)
        frames_test[batch_num,:,:,:,:] = np.rollaxis(images,axis=-1)
        labels_test[batch_num,:] = np.reshape(rppg_data_label[batch_num*Batch_Size:batch_num*Batch_Size + Batch_Size],[1,Batch_Size])
        mask_test[batch_num,:,:,:] = np.rollaxis(mask_images,axis=-1)
      
print('Frames Test Generated')
print('Mask images Test Generated')
print('RPPG Labels Test Generated')

#Call rPPGNet Model & two loss functions
loaded_model = rPPGNet()
criterion_Binary = nn.BCELoss()  # binary segmentation
criterion_Pearson = Neg_Pearson()   # rPPG singal 

#Load the model and put model in evaluation model
loaded_model.load_state_dict(torch.load('/content/drive/MyDrive/stevenrppgmodel.pt'))
loaded_model.eval()

print('**********Start Testing**************')

for i in range(batch_num_count):
    Input_Frames =np.reshape(frames_test[i,:,:,:,:], [1,Channel,Batch_Size,Image_Height,Image_Width])
    Target_rppg_labels = np.reshape(labels_test[i,:],[1,Batch_Size])
    Skin_Mask_Images = np.reshape(mask_test[batch_num,:],[1,Batch_Size, Mask_Height,Mask_Width])
    Input_Frames = torch.from_numpy(Input_Frames).float()
    Skin_Mask_Images = torch.from_numpy(Skin_Mask_Images).float()
    Target_rppg_labels = torch.from_numpy(Target_rppg_labels).float()

    #1.Forward the model, get the predicted skin maps and rPPG signals
    skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232  = loaded_model(Input_Frames)
    #2. Calculate the loss between predicted skin maps and binary skin labels (loss_binary)
    loss_binary = criterion_Binary(skin_map, Skin_Mask_Images)  
    #3. Calculate the loss between predicted rPPG signals and groundtruth smoothed ecg signals (loss_ecg, loss_ecg1, loss_ecg2, loss_ecg3,## loss_ecg4, loss_ecg_aux)
    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize2
    rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) /torch.std(rPPG_SA1)	 	# normalize2
    rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) /torch.std(rPPG_SA2)	 	# normalize2
    rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) /torch.std(rPPG_SA3)	 	# normalize2
    rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) /torch.std(rPPG_SA4)	 	# normalize2
    rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) /torch.std(rPPG_aux)	 	# normalize2
    loss_ecg = criterion_Pearson(rPPG, Target_rppg_labels)
    loss_ecg1 = criterion_Pearson(rPPG_SA1, Target_rppg_labels)
    loss_ecg2 = criterion_Pearson(rPPG_SA2, Target_rppg_labels)
    loss_ecg3 = criterion_Pearson(rPPG_SA3, Target_rppg_labels)
    loss_ecg4 = criterion_Pearson(rPPG_SA4, Target_rppg_labels)
    loss_ecg_aux = criterion_Pearson(rPPG_aux, Target_rppg_labels)
    loss = 0.1*loss_binary +  0.5*(loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg   
    loss_print = loss.detach().numpy() 
print('Testing loss=' + str(loss_print))


