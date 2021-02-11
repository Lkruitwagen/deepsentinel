""" tile2vec from https://github.com/ermongroup/tile2vec/blob/master/src/training.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from deepsentinel.models.encoders import encoders


class TileNet(nn.Module):
    def __init__(self, encoder, encoder_params, bilinear=False, image_channels=5):
        super(TileNet, self).__init__()
        self.encoder = encoders[encoder](**encoder_params)

    def forward(self, x):
        return self.encoder(x)
