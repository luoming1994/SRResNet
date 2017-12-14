# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:21:10 2017

@author: LM
"""

import torch
import torch.nn as nn
import math
import torch.nn.init as init

class _Residual_Block(nn.Module):
    def __init__(self,bn = True):
        super(_Residual_Block, self).__init__()
        
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(num_parameters=1,init=0.2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        output = self.conv1(x)
        if self.bn:
            output =self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        if self.bn:
            output = self.bn2(output)
        output = torch.add(output,x)
        return output 

class SRResNet(nn.Module):
    def __init__(self,in_channels = 3,out_channels = 3,bn = True):
        """
        in and out channel = io_channels;RGB mode io_channels =3 ,Y channel io_channels = 1
        """
        super(SRResNet, self).__init__()
        
        self.bn = bn
        self.conv_input = nn.Conv2d(in_channels = in_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        #self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.PReLU(num_parameters=1,init=0.2)
        
        self.residual = self.make_layer(_Residual_Block, bn, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.bn:
            self.bn_mid = nn.BatchNorm2d(64)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.PReLU(num_parameters=1,init=0.2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.PReLU(num_parameters=1,init=0.2),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels = out_channels, kernel_size=9, stride=1, padding=4, bias=False)
        
        # init the weight of conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, bn, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(bn = bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.relu(self.conv_input(x))
        #residual = out.clone()
        out = self.residual(out1)
        out = self.conv_mid(out)
        if self.bn:
            out = self.bn_mid(out)
        out = torch.add(out,out1)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out
    
    
class SRResNet_RGBY(nn.Module):
    def __init__(self,in_channels = 4,out1_channels = 3,out2_channels = 1,bn = True):
        """
        in and out channel = io_channels;RGB mode io_channels =3 ,Y channel io_channels = 1
        """
        super(SRResNet_RGBY, self).__init__()
        
        self.bn = bn
        self.conv_input = nn.Conv2d(in_channels = in_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        #self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.PReLU(num_parameters=1,init=0.2)
        
        self.residual = self.make_layer(_Residual_Block, bn, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.bn:
            self.bn_mid = nn.BatchNorm2d(64)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.PReLU(num_parameters=1,init=0.2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.PReLU(num_parameters=1,init=0.2),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels = out1_channels, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv_output2 = nn.Conv2d(in_channels = out1_channels, out_channels = out2_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # init the weight of conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, bn, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(bn = bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.relu(self.conv_input(x))
        #residual = out.clone()
        out = self.residual(out1)
        out = self.conv_mid(out)
        if self.bn:
            out = self.bn_mid(out)
        out = torch.add(out,out1)
        out = self.upscale4x(out)
        rgb = self.conv_output(out)
        y = self.conv_output2(rgb)
        print(y.size())
        out = torch.cat((rgb,y), dim = 1)   # channel
        return out