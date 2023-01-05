# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:05:54 2021

@author: gbellitto
"""

import os
import numpy as np

from dataset.FFRDatasetCV import FFRDataset
import matplotlib.pyplot as plt
import monai
import copy
from torch.utils.data import DataLoader
from monai.transforms import MapTransform, Randomizable
from monai.config import KeysCollection
from monai.transforms import (
    DivisiblePadD,  
    LoadImageD, 
    NormalizeIntensityD,
    AddChannelD,
    AsDiscrete,
    Activations,
    Compose,
    RandFlipD,
    RandRotate90D,
    ToTensorD,
    AsChannelFirstD,
    ScaleIntensityD
    )
from utils.transforms import ResizeWithRatio, Delete4Ch,Convert1Ch, NewMergedImage, RandPatchedImageLateFusion,CenterPatchedImageAndEarlyFusion, RandPatchedImage3Channels, RandPatchedImage, RandPatchedImageAndEarlyFusion, RandDepthCrop, NDITKtoNumpy


data_dir = os.path.join('data')
data_path =  os.path.join('data','5BalancedFoldValSplit_complete.json')

num_fold = 0
inner_loop = 0
image_key = "image"
KEYS = (image_key, "label")

img_size = 224

train_transforms = Compose([
        LoadImageD(KEYS[0]),
        Delete4Ch(KEYS[0]),
        ResizeWithRatio(KEYS[0], image_size = img_size),
        AsChannelFirstD(KEYS[0]),
        ScaleIntensityD(KEYS[0]),
        DivisiblePadD(KEYS[0], k = img_size, mode = 'constant'),
        RandFlipD(KEYS[0], prob = 0.5, spatial_axis=0),
        RandRotate90D(KEYS[0], prob=0.5, spatial_axes=(0,1)),  
        ToTensorD(KEYS[0])
        ])


val_transforms = Compose([
    LoadImageD(KEYS[0]),
    Delete4Ch(KEYS[0]),
    ResizeWithRatio(KEYS[0], image_size = img_size),
    AsChannelFirstD(KEYS[0]),
    ScaleIntensityD(KEYS[0]),
    DivisiblePadD(KEYS[0], k = img_size, mode = 'constant'),
    ToTensorD(KEYS[0]),
    ])


#dataset_train = Dataset(root_dir = data_dir, num_fold=num_fold, data_path = data_path, section = 'training', transforms = train_transforms)
dataset_val = FFRDataset(root_dir = data_dir, split_path = data_path, num_fold=num_fold, section = 'validation', transforms = train_transforms)
# dataset_test = Dataset(root_dir = data_dir, data_path = data_path, num_fold=num_fold, section = 'test', transforms = train_transforms)
#dataset_show_aug = PanMRIDataset(data_dir, 'training', train_transforms)


'''
train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
'''

# from pylab import rcParams
# rcParams['figure.figsize'] = 20,5
# for i in range(len(dataset_val)): 

#     f, axarr = plt.subplots(1,9)
#     data = dataset_val[i]
#     print(f"name: {data['image_meta_dict']['filename_or_obj'].split()[-1]}")
#     for j in range(8):        
#         axarr[j].imshow(data['image'].numpy()[0,:,:])
#         axarr[j].set_title(f"Orig {j}")
#     plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 20,5
f, axarr = plt.subplots(1,9)
for i in range(8): 
    data = dataset_val[i]
    print(f"name: {data['image_meta_dict']['filename_or_obj'].split()[-1]}")
    axarr[i].imshow(data['image'].numpy()[0,:,:])
    axarr[i].set_title(f"Orig {i}")
plt.show()
