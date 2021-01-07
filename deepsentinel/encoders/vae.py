""" A CNN variational autoencoder from https://github.com/sksq96/pytorch-vae/blob/master/vae.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self,size):
        super(UnFlatten, self).__init__()
        self.size=size
        
    def forward(self, input):
        return input.view(input.size(0), 256, 6, 6)



class VAE(nn.Module):
    def __init__(self, image_channels=5, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        self.flatten=Flatten()
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.unflatten=UnFlatten(size=9216)
        
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
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2),
            nn.ZeroPad2d((0,-1,0,-1)),
            #nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        #esp = torch.randn(*mu.size())
        esp = torch.cuda.FloatTensor(*mu.size()).normal_()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        #print ('mu',mu)
        #print ('logvar',logvar)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        z = self.unflatten(z)
        return self.decoder(z) #, mu, logvar