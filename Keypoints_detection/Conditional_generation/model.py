import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

def conv_block(in_channels, out_channels, kernel_size, stride, dilation=1, bias=True, batch_norm=True, layer_norm=False, activation='ReLU'):
    padding = (dilation*(kernel_size-1)+2-stride)//2
    seq_modules = nn.Sequential()
    seq_modules.add_module('conv', \
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias))
    if batch_norm:
        seq_modules.add_module('norm', nn.BatchNorm2d(out_channels))
    elif layer_norm:
        seq_modules.add_module('norm', LayerNorm())
    if activation is not None:
        seq_modules.add_module('relu', getattr(nn, activation)(inplace=True))
    return seq_modules


class landmark_detection_network(nn.Module):
    
    def __init__(self, in_channels, n_filters, batch_norm=True, layer_norm=False):
        super(landmark_detection_network, self).__init__()
        self.block_layers = nn.ModuleList()

        conv1 = conv_block(in_channels, n_filters, kernel_size=7, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
        conv2 = conv_block(n_filters, n_filters, kernel_size=3, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)

        self.block_layers.append(conv1)
        self.block_layers.append(conv2)

        new_filters = n_filters * 2
        conv3 = conv_block(n_filters, new_filters, kernel_size=3, stride=2, batch_norm=batch_norm, layer_norm=layer_norm)
        conv4 = conv_block(new_filters, new_filters, kernel_size=3, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
        n_filters = new_filters

        self.block_layers.append(conv3)
        self.block_layers.append(conv4)

        new_filters = n_filters * 2
        conv5 = conv_block(n_filters, new_filters, kernel_size=3, stride=2, batch_norm=batch_norm, layer_norm=layer_norm)
        conv6 = conv_block(new_filters, new_filters, kernel_size=3, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
        n_filters = new_filters

        self.block_layers.append(conv5)
        self.block_layers.append(conv6)

        new_filters = n_filters * 2
        conv7 = conv_block(n_filters, new_filters, kernel_size=3, stride=2, batch_norm=batch_norm, layer_norm=layer_norm)
        conv8 = conv_block(new_filters, new_filters, kernel_size=3, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
        n_filters = new_filters

        self.block_layers.append(conv7)
        self.block_layers.append(conv8)

    def forward(self, x):
        block_features = []
        for block in self.block_layers:
            x = block(x)
            block_features.append(x)

        return block_features

