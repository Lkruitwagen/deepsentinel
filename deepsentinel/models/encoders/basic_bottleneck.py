import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBottleneck(nn.Module):
    """Conventional CNN bottleneck."""
    
    def __init__(self, input_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.net(x)