""" Fully-convolutional neural network with bits taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from deepsentinel.models.encoders import encoders


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            #self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2) # think the break is here, in_channels//2!=out_channels
            #self.conv = DoubleConv(in_channels, out_channels)
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2) # think the break is here, in_channels//2!=out_channels
            self.conv = DoubleConv(out_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        #print ('x forward up', x.shape)
        x = self.conv(x)
        #print ('x forward conv', x.shape)
        # input is CHW
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimpleFCNN(nn.Module):
    def __init__(self, encoder, encoder_params, n_classes, activation, bilinear):
        super(SimpleFCNN, self).__init__()
        self.encoder = encoders[encoder](**encoder_params)
        
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        if activation=='sigmoid':
            self.activation=lambda x: torch.softmax(x, dim=1)
        else:
            self.activation=lambda x: x
        
    def forward(self, x):
        x = self.encoder(x)
        #print ('encoder', x.shape)
        x = self.up1(x)
        #print ('up1', x.shape)
        x = self.up2(x)
        #print ('up2', x.shape)
        x = self.up3(x)
        #print ('up3', x.shape)
        x = self.up4(x)
        #print ('up4', x.shape)
        x = self.outc(x)
        #print ('outc',x.shape)
        return self.activation(x)
    
class SimpleCNN(nn.Module):
    def __init__(self, encoder, encoder_params, activation):
        super(SimpleCNN, self).__init__()
        self.encoder = encoders[encoder](**encoder_params)
        if activation=='softmax':
            self.activation=lambda x: torch.softmax(x, dim=1)
        elif activation=='relu':
            self.activation=lambda x: torch.relu(x)
        else:
            self.activation=lambda x: x
        
    def forward(self, x):
        x = self.encoder(x)
        return self.activation(x)