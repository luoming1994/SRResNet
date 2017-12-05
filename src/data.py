# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:45:06 2017

@author: LM
"""
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch



def loadImg2Numpy(filepath):
    """
    load a rgb image to numpy(channel * H * W)
    dtype = np.uint8 (make the data easy to Storage)
    """
    img = Image.open(filepath)  
    img = np.array(img).transpose(2, 0, 1)  # Image=>numpy.array
    
    return img

def numpy2Tensor(img_np):
    """
    np.uint8 => torch.Tensor
    """
    img_np = torch.from_numpy(img_np)
    return img_np.float().div(255)

def cut2normal(img_np,cut_size = 24):
    """
    cut a numpy(channel * H * W ) to normal size
    """
    shape = img_np.shape
    assert len(shape) == 3,"img_np is not 3 dim"
    nH,nW = shape[-2]//cut_size, shape[-1]//cut_size
    c = shape[0]    # channels
    img = np.empty((nH*nW*c,cut_size,cut_size),dtype=img_np.dtype)
    index = 0
    for i in range(nH):
        for j in range(nW):
            img[index*c:(index+1)*c,:,:] = img_np[:,i*cut_size:(i+1)*cut_size,j*cut_size:(j+1)*cut_size]
            index += 1 
            
    return img
 
img2 = np.concatenate((img,img),axis=0)    

class DIV2K(data.Dataset):
    """
    load DIV2K data set to train the SRResNet
    """
    def __init__(self, data_dir,scale = 4,image_num = 800,hr_size = 96):
        assert hr_size % scale == 0,"hr_size can not " 
        self.hr_size = hr_size
        self.lr_size = hr_size//scale
        
    
    def __getitem__(self, index):
        
    def __len__(self):
        


def main(imgDir):
    """
    convert images into data
    """

if __name__ == "__main__":
    main()