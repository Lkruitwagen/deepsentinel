""" A CNN variational autoencoder from https://github.com/sksq96/pytorch-vae/blob/master/vae.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from deepsentinel.models.encoders import encoders


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self,shape):
        super(UnFlatten, self).__init__()
        self.shape=shape
        
    def forward(self, input):
        return input.view(input.size(0), *self.shape[1:])
    
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
        x = self.conv(x)
        return x # maybe int he future we rescale to [-1,1] and tanh activtion



class VAE(nn.Module):
    def __init__(self, encoder, encoder_params, bilinear=False, image_channels=5,c_dim=64, h_dim=64*8*8, z_dim=64*8*8):
        super(VAE, self).__init__()
        self.encoder = encoders[encoder](**encoder_params)
        
        
        
        self.conv1 = nn.Conv2d(512,c_dim,kernel_size=1) #512-> 64
        self.conv2 = nn.Conv2d(512,c_dim,kernel_size=1) # 512->64
        
        self.flatten=Flatten() # 64x8x8
        
        ### let's not do linear, let's do 1x1 conv.
        #self.fc1 = nn.Linear(h_dim, z_dim) # mean
        #self.fc2 = nn.Linear(h_dim, z_dim) # logvar
        #self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.unflatten=UnFlatten(shape=(-1,c_dim,8,8)) # 64*8*8 -> 64,8,8
        
        self.conv3 = nn.Conv2d(c_dim,512,kernel_size=1)
        #
        
        self.decoder = nn.Sequential(
            Up(512, 256, bilinear),
            Up(256, 128, bilinear),
            Up(128, 64, bilinear),
            Up(64, 32, bilinear),
            OutConv(32, image_channels)
        )
        
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        #esp = torch.randn(*mu.size())
        esp = torch.cuda.FloatTensor(*mu.size()).normal_()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.conv1(h), self.conv2(h)
        mu = self.flatten(mu)
        logvar = self.flatten(logvar)
        #print ('mu',mu)
        #print ('logvar',logvar)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        #h = self.conv1(h)
        #print ('conv1',h.shape)
        #h = self.flatten(h)
        #print ('flatten',h.shape)
        z, mu, logvar = self.bottleneck(h)
        z = self.unflatten(z)
        z = self.conv3(z)
        return self.decoder(z), mu, logvar