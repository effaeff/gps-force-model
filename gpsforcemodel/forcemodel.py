"""Custom network as force model for GPS"""

from functools import reduce
import numpy as np

from pytorchutils.globals import nn, torch, DEVICE
from pytorchutils.layers import (
    Attention,
    LinearAttention,
    LinearAttention3d,
    Residual,
    PreNorm,
    PreNorm3d
)
from pytorchutils.basic_model import BasicModel

from config import WINDOW

class TemporalPad(nn.Module):
    """Left pad second to last dimension of input with zeros"""
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, inp):
        """Forward pass"""
        return nn.functional.pad(inp, (0, 0, self.pad_size, 0), 'constant', 0)

class ForceModel(BasicModel):
    """Class for model"""
    def __init__(self, config):
        BasicModel.__init__(self, config)

        self.kernel_size = config.get('kernel_size', 3)
        self.padding = config.get('padding', 1)
        self.dilation = config.get('dilation', 1)
        self.channels = config.get('channels', [32, 64, 128])
        self.force_samples = config['force_samples']

        self.drop_rate = config.get('drop_rate', 0)

        self.encoder = self.make_layers(self.input_size if WINDOW else 1, self.channels)
        self.decoder = self.make_layers(
            self.channels[-1],
            (list(reversed(self.channels))[1:] + ([self.output_size] if WINDOW else [1]))
        )

        self.bn = nn.ModuleList(
            [
                nn.BatchNorm2d(chn) for chn in list(reversed(self.channels))[1:]
            ]
        )

        ########################
        #### Simple version ####
        ########################
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)

        # self.fc1 = nn.Linear(self.force_samples * self.output_size * 64, 512)
        # self.fc2 = nn.Linear(512, self.force_samples * self.output_size)

        ########################

    def make_layers(self, in_chn, channels):
        """Outsourced method to build layers of a network block"""
        layers = []
        for chn_idx, chn in enumerate(channels):
            layers += [
                TemporalPad(self.padding[chn_idx]) if WINDOW else nn.Identity(),
                nn.Conv2d(
                    in_chn,
                    chn,
                    kernel_size=(self.kernel_size, (1 if WINDOW else self.kernel_size)),
                    dilation=(self.dilation[chn_idx], 1),
                    padding=(0 if WINDOW else self.padding)
                ),
                nn.BatchNorm2d(chn),
                getattr(nn, self.config.get('activation', 'ReLU'))(),
                nn.Dropout2d(self.drop_rate),
                TemporalPad(self.padding[chn_idx]) if WINDOW else nn.Identity(),
                nn.Conv2d(
                    chn,
                    chn,
                    kernel_size=(self.kernel_size, (1 if WINDOW else self.kernel_size)),
                    dilation=(self.dilation[chn_idx], 1),
                    padding=(0 if WINDOW else self.padding)
                ),
                nn.BatchNorm2d(chn),
                getattr(nn, self.config.get('activation', 'ReLU'))(),
                Residual(PreNorm(chn, LinearAttention(chn_idx, chn))),
                nn.Dropout2d(self.drop_rate),
            ]
            in_chn = chn
        return nn.Sequential(*layers)

    def forward(self, inp):
        """Forward pass"""
        # pred_out = reduce(lambda x, y: y(x), self.encoder, inp)
        # pred_out = reduce(lambda x, y: y(x), self.decoder, pred_out)

        # Store results of each block to realize skip connections
        block_size = len(self.encoder) // len(self.channels)
        output = {}
        for idx in range(0, len(self.encoder), block_size):
            for layer in range(idx, idx + block_size):
                inp = self.encoder[layer](inp)
            output[f'x{idx}'] = inp

        for idx in range(0, len(self.decoder), block_size):
            for layer in range(idx, idx + block_size):
                inp = self.decoder[layer](inp)
            skip_idx = len(self.encoder) - idx - (2 * block_size)
            if skip_idx >= 0:
                bn_idx = idx // block_size
                inp = self.bn[bn_idx](inp + output[f'x{skip_idx}'])

        pred_out = torch.squeeze(inp)

        ########################
        #### Simple version ####
        ########################
        # pred_out = self.act(self.conv1(inp))
        # pred_out = self.act(self.conv2(pred_out))
        # pred_out = self.act(self.conv3(pred_out))

        # pred_out = pred_out.view(pred_out.size(0), -1)

        # pred_out = self.act(self.fc1(pred_out))
        # pred_out = self.fc2(pred_out)

        # pred_out = pred_out.view(-1, self.force_samples, self.output_size)
        ########################

        return pred_out
