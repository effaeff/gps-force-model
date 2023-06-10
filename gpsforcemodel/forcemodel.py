"""Custom network as force model for GPS"""

from functools import reduce
import numpy as np

from pytorchutils.globals import nn, torch, DEVICE
from pytorchutils.layers import Attention, LinearAttention, Residual, PreNorm
from pytorchutils.basic_model import BasicModel

class ForceModel(BasicModel):
    """Class for model"""
    def __init__(self, config):
        BasicModel.__init__(self, config)

        self.kernel_size = config.get('kernel_size', 3)
        self.padding = config.get('padding', 1)

        channels = [32, 64, 128]

        layers = []
        in_chn = 1
        for chn in channels:
            layers += [
                nn.Conv2d(in_chn, chn, self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(chn),
                nn.ReLU(inplace=True),
                Residual(nn.Conv2d(chn, chn, self.kernel_size, padding=self.padding)),
                nn.BatchNorm2d(chn),
                nn.ReLU(inplace=True),
                Residual(PreNorm(chn, Attention(chn))),
                # Reduce last dimension of image by 1
                # The other dimension is left untouched
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
            ]
            in_chn = chn

        self.encoder = nn.Sequential(*layers)

        self.fc_layer = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, inp):
        """Forward pass"""
        pred_out = reduce(lambda x, y: y(x), self.encoder, inp)
        pred_out = self.fc_layer(pred_out)
        return torch.squeeze(pred_out)