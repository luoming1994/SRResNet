import numpy as np
import torch


def PSNR(im,gt,shave_border=0):
    """
    im: image with noise,value in [0,255]
    gt: GroundTurth image,value in [0,255]
    shave_border: the border width need to shave
    """   
    im_shape = im.shape
    gt_shape = gt.shape
    if gt_shape != im_shape:
        return -1
    im = np.array(im,dtype = np.float32)
    gt = np.array(gt,dtype = np.float32)
    if len(im_shape) == 3:
        c,h,w = im_shape
        im = im[:,shave_border:h - shave_border,shave_border:w - shave_border]
        gt = gt[:,shave_border:h - shave_border,shave_border:w - shave_border]
    elif len(im_shape) == 2:
        h,w = im_shape
        im = im[shave_border:h - shave_border,shave_border:w - shave_border]
        gt = gt[shave_border:h - shave_border,shave_border:w - shave_border]
    mse = np.mean((gt - im)**2)
    if mse == 0:
        return 100
    psnr = 10*np.log10(255**2/mse)
    return psnr

def SSIM(im,gt):
    im_shape = im.shape
    gt_shape = gt.shape
    if gt_shape != im_shape:
        return -1   
    
    # C1=(K1*L)^2, 
    # C2=(K2*L)^2
    # C3=C2/2,     1=0.01, K2=0.03, L=255
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    C3 = C2/2.0
    
    mean_x = im.mean() # mean of im
    mean_y = gt.mean() # mean of gt
    cov = np.cov([gt.flatten(),im.flatten()])
    cov_xx = cov[0,0]
    cov_x = np.sqrt(cov_xx)
    cov_yy= cov[1,1]
    cov_y = np.sqrt(cov_yy) 
    cov_xy = cov[0,1]
    
    l_xy = (2*mean_x*mean_y + C1) / (mean_x**2 + mean_y**2 + C1)
    c_xy = (2*cov_x*cov_y + C2) / (cov_xx + cov_yy + C2)
    s_xy = (cov_xy + C3) / (cov_x*cov_y + C3)
    ssim = l_xy*c_xy*s_xy
    
    return ssim


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            #t.sub_(m).div_(s)
            t = t.mul(s).add(m)
        return tensor

class deNormalize(object):
    """ de Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel*std + mean) 

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            #t.sub_(m).div_(s)
            t.mul_(s).add_(m)
        return tensor