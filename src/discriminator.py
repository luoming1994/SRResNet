# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:27:52 2017

@author: LM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 96*96
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        # 48*48
        
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        # 24*24
        
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)
        # 12*12
        
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.LeakyReLU(0.2, inplace=True)
        # 6*6
        self.pool =  nn.AvgPool2d(6, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        
        self.fc1 = nn.Linear(512,1024)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024,1)
        #self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.conv1(x))

        out = self.relu2(self.bn2(self.conv2(out)) )
        out = self.relu3(self.bn3(self.conv3(out)) )
        out = self.relu4(self.bn4(self.conv4(out)) )
        out = self.relu5(self.bn5(self.conv5(out)) )
        out = self.relu6(self.bn6(self.conv6(out)) )
        out = self.relu7(self.bn7(self.conv7(out)) )
        out = self.relu8(self.bn8(self.conv8(out)) )

        out = self.pool(out)
        out = out.view(out.size(0),-1)  # batchsize * fc_in
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.softmax(out)
        out = self.sigmoid(out)
        return out