# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:29:52 2017

@author: LM
"""

import argparse, os,re
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SRResNet import SRResNet
from data import DIV2K
from torchvision import models
import torch.utils.model_zoo as model_zoo
from tool import Normalize
from tool import deNormalize

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpu_num", default="1", type=str, help="which gpu(0 or 1) to use for train")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")




normal = Normalize(mean = [0.485, 0.456, 0.406],
                   std = [0.229, 0.224, 0.225])
deNormal = deNormalize(mean = [0.485, 0.456, 0.406],
                       std = [0.229, 0.224, 0.225])


def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    dataPath = r'/home/we/devsda1/lm/DIV2K/DIV2K_SRResNet200.npz'
    train_set = DIV2K(dataPath)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    model = SRResNet()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda() 

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            #checkpoint = torch.load(opt.resume)
            #opt.start_epoch = checkpoint["epoch"] + 1
            #model.load_state_dict(checkpoint["model"].state_dict())
            # model_epoch_20.pth => 20
            opt.start_epoch = int(re.split('_',re.split('\.',opt.resume)[-2])[-1]) + 1
            model.load_state_dict(torch.load(opt.resume))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
            
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    with open(os.path.join('../log','train.log'),'w') as f:
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            train(training_data_loader, optimizer, model, criterion, epoch, f)
            save_checkpoint(model, epoch)
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(training_data_loader, optimizer, model, criterion, epoch, file):
    # adjust learnnig rate,every step reduce to 0.1*lr
    lrate = adjust_learning_rate(optimizer, epoch-1)    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lrate 

    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        for j in range(batch[0].shape[0]):  # batchsize
            batch[0] = normal(batch[0])
            batch[1] = normal(batch[1])
        lr, hr = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        # lr = (lr -mean)/std
        #lr = normal(lr)
        #hr = normal(hr)
        if opt.cuda:
            lr = lr.cuda()
            hr = hr.cuda()        
        sr = model(lr)
        loss = criterion(sr, hr)

        if opt.vgg_loss:
            content_input = netContent(sr)
            content_target = netContent(hr)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)
        
        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_variables=True)
        
        loss.backward()

        optimizer.step()
        if opt.vgg_loss:
            write_str = '%f\t%f\n' % (loss.data[0],content_loss.data[0])
        else:
            write_str = '%f\n' % (loss.data[0])
        file.write(write_str)
        if iteration%100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f} Content_loss {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0], content_loss.data[0]))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    
def save_checkpoint(model, epoch):
    model_out_path = "../model/" + "model_epoch_{}.pth".format(epoch)
    #state = {"epoch": epoch ,"model": model}
    if not os.path.exists("../model/"):
        os.makedirs("../model/")

    #torch.save(state, model_out_path)
    torch.save(model.state_dict(), model_out_path)   
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()