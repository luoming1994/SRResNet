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
from SRResNet import SRResNet,SRResNet_RGBY
from PIL import Image
from tool import Normalize,deNormalize
from data import loadImgYCbCr2Numpy,loadImgRGB2Numpy
from data import numpy2Tensor,tensor2Numpy
from data import numpyYCbCr2Image,numpyRGB2Image
from tool import PSNR

parser = argparse.ArgumentParser(description="PyTorch SRResNet Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="../model/model_ch[RGB][RGB]_epoch_2.pth", type=str, help="model path")
parser.add_argument("--image", default="../Set5/butterfly_GT.bmp", type=str, help="image name")
parser.add_argument("--image_save", default="../result/sr_ch[RGB][RGB]_epoch_1.bmp", type=str, help="image save name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--in_channel", default=3, type=int, help="network processing channel,rgb:3,y channel:1, Default: 1")
parser.add_argument("--out_channel", default=3, type=int, help="network processing channel,rgb:3,y channel:1, Default: 1")


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
    

    
img_bi_rgb = loadImgRGB2Numpy(opt.image,down_scale = 1/opt.scale,up_scale = opt.scale)
img_lr_rgb = loadImgRGB2Numpy(opt.image,down_scale = 1/opt.scale)
img_hr_rgb = loadImgRGB2Numpy(opt.image)

img_bi_YCbCr = loadImgYCbCr2Numpy(opt.image,down_scale = 1/opt.scale,up_scale = opt.scale)
img_lr_YCbCr = loadImgYCbCr2Numpy(opt.image,down_scale = 1/opt.scale)
img_hr_YCbCr = loadImgYCbCr2Numpy(opt.image)

if opt.in_channel == 1 and opt.out_channel == 1:    # Y model
    lr = numpy2Tensor(img_lr_YCbCr[0:1])
    hr = numpy2Tensor(img_hr_YCbCr[0:1])
elif opt.in_channel == 3 and opt.out_channel == 3:
    lr = numpy2Tensor(img_lr_rgb)
    hr = numpy2Tensor(img_hr_rgb)
elif opt.in_channel == 4 and opt.out_channel == 4:
    lr = numpy2Tensor(np.concatenate((img_lr_rgb,img_lr_YCbCr[0:1]),0))
    hr = numpy2Tensor(np.concatenate((img_hr_rgb,img_hr_YCbCr[0:1]),0))
elif opt.in_channel == 3 and opt.out_channel == 1:    # RGB => Y model:
    lr = numpy2Tensor(img_lr_rgb)
    hr = numpy2Tensor(img_hr_YCbCr[0:1])
else:
    raise Exception("channel param error")
# 3dim => 4dim
lr = torch.unsqueeze(lr,0)  # imput data, lr image
hr = torch.unsqueeze(hr,0)  # ground truth label, hr image
lr = Variable(lr)


model = SRResNet(in_channels = opt.in_channel,out_channels = opt.out_channel,bn = False)
#model = SRResNet_RGBY(in_channels = 4,out1_channels = 3,out2_channels =1,bn = False)
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

if opt.in_channel == 1 and opt.out_channel == 1:    # Y model
    psnr_sr = PSNR(sr[0],img_hr_YCbCr[0],shave_border = opt.scale)
    psnr_bicubic = PSNR(img_bi_YCbCr[0], img_hr_YCbCr[0], shave_border = opt.scale)
    img_sr_YCbCr = img_bi_YCbCr.copy()
    img_sr_YCbCr[0,:,:] = sr[0,:,:]
    img_save = numpyYCbCr2Image(img_sr_YCbCr)
    img_save.save(opt.image_save)
elif opt.in_channel == 3 and opt.out_channel == 3:
    psnr_sr = PSNR(sr,img_hr_rgb,opt.scale)
    psnr_bicubic = PSNR(img_bi_rgb, img_hr_rgb, shave_border = opt.scale)
    img_sr_rgb = sr.copy()
    img_save = numpyRGB2Image(img_sr_rgb)
    img_save.save(opt.image_save)
elif opt.in_channel == 4 and opt.out_channel == 4:
    psnr_sr = PSNR(sr[0:3],img_hr_rgb,opt.scale),PSNR(sr[3:4],img_hr_YCbCr[0:1],opt.scale)
    psnr_bicubic = PSNR(img_bi_rgb, img_hr_rgb, shave_border = opt.scale),PSNR(img_bi_YCbCr[0:1],img_hr_YCbCr[0:1],opt.scale)
    img_sr_rgb = sr[0:3].copy()
    img_save = numpyRGB2Image(img_sr_rgb)
    img_save.save(opt.image_save)
    
    img_sr_YCbCr = img_bi_YCbCr.copy()
    img_sr_YCbCr[0,:,:] = sr[3,:,:]
    img_save = numpyYCbCr2Image(img_sr_YCbCr)
    img_save.save(opt.image_save + '.bmp')
elif opt.in_channel == 3 and opt.out_channel == 1:    # RGB => Y model:
    psnr_sr = PSNR(sr[0],img_hr_YCbCr[0],shave_border = opt.scale)
    psnr_bicubic = PSNR(img_bi_YCbCr[0], img_hr_YCbCr[0], shave_border = opt.scale)
    img_sr_YCbCr = img_bi_YCbCr.copy()
    img_sr_YCbCr[0,:,:] = sr[0,:,:]
    img_save = numpyYCbCr2Image(img_sr_YCbCr)
    img_save.save(opt.image_save)

print('SR:',psnr_sr,' ; ','BICUBIC:',psnr_bicubic)
    
