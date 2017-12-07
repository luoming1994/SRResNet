# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:45:06 2017

@author: LM
"""
import os
import platform
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch

def isImage(filename):
    """
    a file is a image? via extension
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def loadImg2Numpy(filepath,scale = None):
    """
    load a rgb image to numpy(channel * H * W)
    dtype = np.uint8 (make the data easy to Storage)
    """
    img = Image.open(filepath)  
    if scale is not None:
        W,H = img.size
        img.resize((W*scale,H*scale),Image.BICUBIC)
    img = np.array(img).transpose(2, 0, 1)  # Image=>numpy.array
    
    return img


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

def numpy2Tensor(img_np):
    """
    np.uint8 => torch.Tensor
    """
    img_np = torch.from_numpy(img_np)
    return img_np.float().div(255)
 
 

class img2data(object):
    """
    transform images as numpy(dtype = np.uint8) into data storage in disk
    """
    def __init__(self,hr_dir, lr_dir = None,upscale = 4,img_num = 800,hr_size = 96):
        assert hr_size % upscale == 0,"hr_size can not div scale" 
        self.hr_size    = hr_size
        self.lr_size    = hr_size//upscale
        self.upscale      = upscale
        self.hr_paths   = [os.path.join(hr_dir, x) for x in os.listdir(hr_dir) if isImage(x)]
        if lr_dir == None:
            self.lr_paths = self.hr_paths
            self.down_scale = 1/upscale
        else:
            self.lr_paths   = [os.path.join(lr_dir, x) for x in os.listdir(lr_dir) if isImage(x)]
            self.down_scale = 1 # the lr image has been down sacled ,so down_scale = 1(no need to down scale agian)
        assert len(self.hr_paths) == len(self.lr_paths),"hr_dir,lr_dir have the image num is not the same"
        # get the first img_num images
        if img_num < len(self.hr_paths):
            self.lr_paths = self.lr_paths[0:img_num]
            self.hr_paths = self.hr_paths[0:img_num]
            self.img_num    = img_num
        else:
            self.img_num = len(self.hr_paths)
        
        self.lr = np.array([],dtype = np.uint8).reshape(-1,self.lr_size,self.lr_size)
        self.hr = np.array([],dtype = np.uint8).reshape(-1,self.hr_size,self.hr_size)
    
    def imgLoad(self):
        for hr_path in self.hr_paths:
            imgs = cut2normal(loadImg2Numpy(hr_path),cut_size = self.hr_size)
            self.hr = np.concatenate((self.hr,imgs),axis=0) # concat
        for lr_path in self.lr_paths:
            imgs = cut2normal(loadImg2Numpy(lr_path,scale = self.down_scale ),cut_size = self.lr_size)
            self.lr = np.concatenate((self.lr,imgs),axis=0) 
    
    def imgSave(self,save_path):
        np.savez(save_path,lr = self.lr, hr = self.hr)
        
        
        


class DIV2K(data.Dataset):
    """
    load DIV2K data set to train the SRResNet
    """
    def __init__(self,dataPath,channels =3):
        super(DIV2K,self).__init__()
        
        dt = np.load(dataPath)
        self.lr = dt['lr']
        self.hr = dt['hr']
        self.ch = channels
    
    def __getitem__(self, index):
        """
        get the index item
        """
        # np.uint8(0~255) => folatTensor (0.0~1.0)
        lr = numpy2Tensor(self.lr[index*self.ch:(index+1)*self.ch,:,:])
        hr = numpy2Tensor(self.hr[index*self.ch:(index+1)*self.ch,:,:])
        
        return lr,hr
        
    def __len__(self):
        """
        get the data lens
        """
        
        return self.lr.shape[0]//self.ch
    
        


def main():
    """
    convert images into data
    """
    sysstr = platform.system()
    if(sysstr =="Windows"): # Windows
        hr_dir = r'E:\Data\DIV2K\DIV2K_train_HR'
        lr_dir = r'E:\Data\DIV2K\DIV2K_train_LR_bicubic\X4'
        dataPath = r'E:\Data\DIV2K\DIV2K_SRResNet.npz'
    elif(sysstr == "Linux"): # Linux
        hr_dir = r'/home/we/devsda1/lm/DIV2K/DIV2K_train_HR'
        lr_dir = r'/home/we/devsda1/lm/DIV2K/DIV2K_train_LR_bicubic/X4'
        dataPath = r'/home/we/devsda1/lm/DIV2K/DIV2K_SRResNet.npz'
    else:
        print ("don't support the system")
    
    dt = img2data(hr_dir, lr_dir ,upscale = 4,img_num = 20,hr_size = 96)
    dt.imgLoad()
    dt.imgSave(dataPath)
    
    

if __name__ == "__main__":
    main()