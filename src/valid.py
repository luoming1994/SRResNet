# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:43:41 2017

@author: LM
"""

import argparse
import torch
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SRResNet import SRResNet
from PIL import Image
from tool import Normalize,deNormalize
from data import loadImgYCbCr2Numpy,loadImgRGB2Numpy
from data import numpy2Tensor,tensor2Numpy
from data import numpyYCbCr2Image,numpyRGB2Image
from tool import PSNR

parser = argparse.ArgumentParser(description="PyTorch SRResNet Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="../model/model_rgb_epoch_500.pth", type=str, help="model path")
parser.add_argument("--image", default="../Set5/butterfly_GT.bmp", type=str, help="image name")
parser.add_argument("--image_save", default="../result/sr.bmp", type=str, help="image save name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=3, type=int, help="network processing channel,rgb:3,y channel:1, Default: 1")


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opt.channel == 1:    # Y model
    # the image should can be divided by scale
    img_lr = loadImgYCbCr2Numpy(opt.image,down_scale = 1/opt.scale)
    img_bicubic = loadImgYCbCr2Numpy(opt.image,down_scale = 1/opt.scale,up_scale = opt.scale)
    img_hr = loadImgYCbCr2Numpy(opt.image)
    lr = numpy2Tensor(img_lr[0:1])
    hr = numpy2Tensor(img_hr[0:1])
elif opt.channel == 3:
    img_lr = loadImgRGB2Numpy(opt.image,down_scale = 1/opt.scale)
    img_bicubic = loadImgRGB2Numpy(opt.image,down_scale = 1/opt.scale,up_scale = opt.scale)
    img_hr = loadImgRGB2Numpy(opt.image)
    lr = numpy2Tensor(img_lr)
    hr = numpy2Tensor(img_hr)
else:
    raise Exception("channel param error")
# 3dim => 4dim
lr = torch.unsqueeze(lr,0)  # imput data, lr image
hr = torch.unsqueeze(hr,0)  # ground truth label, hr image
lr = Variable(lr)


model = SRResNet(io_channels = opt.channel,bn = True)
weight = torch.load(opt.model)
model.load_state_dict(weight)
if cuda:
    model = model.cuda()
    lr = lr.cuda()
else:
    model = model.cpu()
    
out = model(lr)
out = out.data[0]   # 4dim => 3dims
sr = tensor2Numpy(out)  # 3dim numpy

if opt.channel == 1:    # y channel
    psnr_sr = PSNR(sr[0],img_hr[0],shave_border = opt.scale)
    psnr_bicubic = PSNR(img_bicubic[0], img_hr[0], shave_border = opt.scale)
    img_sr = img_bicubic
    img_sr[0] = sr[0]
    img_save = numpyYCbCr2Image(img_sr)
    img_save.save(opt.image_save) 
elif opt.channel == 3:  # rgb
    psnr = PSNR(sr,img_hr,opt.scale)
    psnr_bicubic = PSNR(img_bicubic, img_hr, shave_border = opt.scale)
    img_save = numpyRGB2Image(sr)
    img_save.save(opt.image_save)
print('SR:',psnr_sr,' ; ','BICUBIC:',psnr_bicubic)
    
