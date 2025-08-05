import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import os

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        same_seeds(0)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)



#参考https://zhuanlan.zhihu.com/p/109166845
#upsample可能会带来不一致
class UpsampleDeterministic(nn.Module):
    def __init__(self,upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        '''
        return x[:, :, :, None, :, None]\
        .expand(-1, -1, -1, self.upscale, -1, self.upscale)\
        .reshape(x.size(0), x.size(1), x.size(2)\
                 *self.upscale, x.size(3)*self.upscale)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        same_seeds(0)

        channels = input_shape[0]#channels = 3

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features#in_features = 64

        # Downsampling
        for _ in range(2):
            out_features *= 2#out_features=128,256
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features


        # Residual blocks  （256， 32， 32）
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                UpsampleDeterministic(2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

        

        #+		input_shape	(3, 128, 128)	tuple
        #+		00	ReflectionPad2d((3, 3, 3, 3))	ReflectionPad2d
        #+		01	Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))	Conv2d
        #+		02	InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)	InstanceNorm2d
        #+		03	ReLU(inplace=True)	ReLU
        #+		04	Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))	Conv2d
        #+		05	InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)	InstanceNorm2d
        #+		06	ReLU(inplace=True)	ReLU
        #+		07	Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))	Conv2d
        #+		08	InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)	InstanceNorm2d
        #+		09	ReLU(inplace=True)	ReLU
        #+		10	ResidualBlock(
        #      (block): Sequential(
        #        (0): ReflectionPad2d((1, 1, 1, 1))
        #        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        #        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        #        (3): ReLU(inplace=True)
        #        (4): ReflectionPad2d((1, 1, 1, 1))
        #        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        #        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        #  )
        #)	ResidualBlock


    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        same_seeds(0)
        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
