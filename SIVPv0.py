import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda')


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]
######## EnlightenGAN Start #######################
class Generator(nn.Module):
    def __init__(self,norm_layer=nn.InstanceNorm2d):
        super(Generator, self).__init__()
        p = 1

        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)

        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = norm_layer(32)

        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = norm_layer(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = norm_layer(64)

        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = norm_layer(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = norm_layer(128)

        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = norm_layer(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = norm_layer(256)

        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = norm_layer(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = norm_layer(512)

        #self.res1 = ResnetBlock(512)
        #self.res2 = ResnetBlock(512)
        #self.res3 = ResnetBlock(512)
        #self.res4 = ResnetBlock(512)

        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = norm_layer(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)

        self.bn6_1 = norm_layer(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)

        self.bn6_2 = norm_layer(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)

        self.bn7_1 = norm_layer(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)

        self.bn7_2 = norm_layer(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)

        self.bn8_1 = norm_layer(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)

        self.bn8_2 = norm_layer(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = norm_layer(32)

        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv10 = nn.Conv2d(32, 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, input):

        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))

        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))  # 128 128 32
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))  # 64 64 64
        x = self.max_pool2(conv2)

        x3 = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x3)))  # 32 32 128
        x = self.max_pool3(conv3)

        x4 = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 32 32 256
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x4)))  # 16 16 256
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))  # 16,16,512

        #x = self.res1(x)
        #x = self.res2(x)
        #x = self.res3(x)
        #x = self.res4(x)

        x = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

        conv5 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [512,32,32]
        up6 = torch.cat([self.deconv5(conv5), conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        x = F.dropout2d(x)
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6 = F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=True)
        up7 = torch.cat([self.deconv6(conv6), conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        x = F.dropout2d(x)
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=True)
        up8 = torch.cat([self.deconv7(conv7), conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        x = F.dropout2d(x)
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=True)
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))
        latent = self.conv10(conv9)

        output = latent + input
        output = self.tanh(output)

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        return output

#######################################################################################
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.LeakyReLU(0.2, True),
                       nn.InstanceNorm2d(dim)]

        conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.LeakyReLU(0.2, True),
                       nn.InstanceNorm2d(dim)]

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.model(x)
        return out
#######################################################################################

class Discriminator(nn.Module):
    def __init__(self,n_layers):
        super(Discriminator, self).__init__()
        input_nc = 3
        ndf = 64
        kw = 4
        padw = 2
        norm_layer = nn.InstanceNorm2d
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.3)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)

if __name__ == '__main__':
    g = Generator()
    x = torch.randn([2,3,320,320])
    gen = g(x)
    print(g)
    print(gen.shape)

    d = Discriminator(4)
    dis = d(x)
    print(d)
    print(dis.shape)

    print("Parameters of full network %.4f " % (sum([m.numel() for m in g.parameters()]) / 1e6))

