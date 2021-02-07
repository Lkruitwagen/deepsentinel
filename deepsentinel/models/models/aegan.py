""" Fully-convolutional neural network with bits taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from deepsentinel.models.encoders import encoders

def build_aegan(encoder, encoder_params, n_classes, activation, bilinear):
    
    model = {
        'encoder':encoders[encoder](**encoder_params),
        'decoder':Generator(),
        'discriminator_image':DiscriminatorImage(),
        'discriminator_latent':DiscriminatorLatent(),
    }
    return model
    
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

    def __init__(self, in_channels, out_channels,simple=False):
        super().__init__()
        if simple:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        else:
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
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2) # think the break is here, in_channels//2!=out_channels
            self.conv = DoubleConv(out_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class DiscriminatorLatent(nn.Module):
    def __init__(self, n_classes):
        self.down = nn.Sequential(
            nn.MaxPool2d(2), #->512x4x4
            nn.Conv2d(512, 512, stride=2, kernel_size=2), #->512x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1024,stride=2,kernel_size=2) # ->1024x1x1
        ) 
        
    def forward(self,x):
        x = self.down(x)
        return torch.sigmoid(torch.sqeeze(x))
    
class DiscriminatorImage(nn.Module):
    def __init__(self, n_classes):
        
        self.down = nn.Sequential(
            Down(n_classes,32, simple=True), #64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Down(n_classes,32, simple=True), #32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Down(n_classes,32, simple=True), #16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Down(n_classes,32, simple=True), #8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Down(n_classes,32, simple=True), #4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(32,64,stride=2,kernel_size=2)
        self.dense = nn.Linear(64,1)
        
    def forward(self,x):
        x = self.down(x)
        x = self.final(x)
        x = self.dense(x)
        return torch.sigmoid(torch.sqeeze(x))
        

class Generator(nn.Module):
    def __init__(self, n_classes, activation, bilinear):
        super(Generator, self).__init__()
        
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        if activation=='sigmoid':
            self.activation=lambda x: torch.softmax(x, dim=1)
        elif activation=='tanh':
            self.activation=lambda x: torch.tanh(x)
        else:
            self.activation=lambda x: x
        
    def forward(self, z):
        #print ('encoder', x.shape)
        z = self.up1(z)
        #print ('up1', x.shape)
        z = self.up2(z)
        #print ('up2', x.shape)
        z = self.up3(z)
        #print ('up3', x.shape)
        z = self.up4(z)
        #print ('up4', x.shape)
        return self.outc(z)