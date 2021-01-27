""" A CNN variational autoencoder from https://github.com/sksq96/pytorch-vae/blob/master/vae.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from deepsentinel.models.encoders import encoders

class SyntheticRGB(nn.Module):
    def __init__(self, encoder, encoder_params, image_channels=5, h_dim=1024, z_dim=32):
        super(SyntheticRGB, self).__init__()
        self.encoder = encoders[encoder](**encoder_params)
        
        self.decoder = nn.Sequential(
            #UnFlatten(h_dim),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ZeroPad2d((0,-1,0,-1)),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ZeroPad2d((0,-1,0,-1)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2),
            nn.ZeroPad2d((0,-1,0,-1)),
            #nn.Sigmoid(),
        )
        
    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h) #, mu, logvar