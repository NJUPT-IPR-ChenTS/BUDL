import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid

def unnormalize(tensor, mean, std, inplace = False):
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor
    

def BDSP_Face(img_1):
    img_1 = unnormalize(img_1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])*255 #反归一化 从-1到1变为0到255
    img = torch.log(img_1 + 1)
    (a, b) = (img.shape[2], img.shape[3]) #图片长和宽
    in_one_dim = 1 
    #beta = 0.5 
    beta = 0.8

    #对图片四周进行填充
    img1 = nn.functional.pad(img, 
                             (in_one_dim, in_one_dim, in_one_dim, in_one_dim), 
                              "replicate")

    #实现：采用2*2 block实现 
    temp1 = img1[:, 1, 0:a, 0:b] #UpLeft
    temp2 = img1[:, 1, 0:a, 1:b+1]#UpRight
    temp3 = img1[:, 1, 1:a+1,0:b]#DownLeft
    temp4 = img1[:, 1,1:a+1, 1:b+1 ]#DownRight

    delete1 = (temp1 - temp4)#UpLeft - DownRight
    delete2 = (temp2 - temp3)#UpRight - DownLeft

    result  = 2 * (beta - 0.5) * (abs(delete1) + abs(delete2)) + (delete1 + delete2) 
    result = torch.atan(4*result)
    result = result/max(abs(float(result.max())), abs(float(result.min())))
    return Variable(result)