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

def get_coord(x, other_axis, axis_size):
    "get x-y coordinates"
    g_c_prob = torch.mean(x, dim=other_axis)  # B,NMAP,W
    g_c_prob = F.softmax(g_c_prob, dim=2) # B,NMAP,W
    coord_pt = torch.linspace(-1.0, 1.0, axis_size).to(x.device) # W
    coord_pt = coord_pt.view(1, 1, axis_size) # 1,1,W
    g_c = torch.sum(g_c_prob * coord_pt, dim=2) # B,NMAP
    return g_c, g_c_prob

def get_gaussian_maps(mu, shape_hw, inv_std, mode='rot'):
    """
    Generates [B,NMAPS,SHAPE_H,SHAPE_W] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, shape_hw[0]).to(mu.device)

    x = torch.linspace(-1.0, 1.0, shape_hw[1]).to(mu.device)

    if mode in ['rot', 'flat']:
        mu_y, mu_x = torch.unsqueeze(mu_y, dim=-1), torch.unsqueeze(mu_x, dim=-1)

        y = y.view(1, 1, shape_hw[0], 1)
        x = x.view(1, 1, 1, shape_hw[1])

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        if mode == 'rot':
            g_yx = torch.exp(-dist)
        else:
            g_yx = torch.exp(-torch.pow(dist + 1e-5, 0.25))

    elif mode == 'ankush':
        y = y.view(1, 1, shape_hw[0])
        x = x.view(1, 1, shape_hw[1])

        g_y = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))
        g_x = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))

        g_y = torch.unsqueeze(g_y, dim=3)
        g_x = torch.unsqueeze(g_x, dim=2)
        g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    return g_yx
'''
class Encoder(nn.Module):
    
    def __init__(self, in_channels, n_filters, batch_norm=True, layer_norm=False):
        super(Encoder, self).__init__()
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
'''
class Encoder(nn.Module):
    """Phi Net:
    input: target image -- distorted image
    output: confidence maps"""
    def __init__(self, in_channels, n_filters, batch_norm=True, layer_norm=False):
        super(Encoder, self).__init__()
        self.block_layers = nn.ModuleList()
        conv1 = conv_block(in_channels, n_filters, kernel_size=7, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
        conv2 = conv_block(n_filters, n_filters, kernel_size=3, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
        self.block_layers.append(conv1)
        self.block_layers.append(conv2)

        for _ in range(3):
            filters = n_filters*2
            conv_i0 = conv_block(n_filters, filters, kernel_size=3, stride=2, batch_norm=batch_norm, layer_norm=layer_norm)
            conv_i1 = conv_block(filters, filters, kernel_size=3, stride=1, batch_norm=batch_norm, layer_norm=layer_norm)
            self.block_layers.append(conv_i0)
            self.block_layers.append(conv_i1)
            n_filters = filters

    def forward(self, x):
        block_features = []
        for block in self.block_layers:
            x = block(x)
            block_features.append(x)
        return block_features
class psai_encoder(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(psai_encoder, self).__init__()
        self.image_encoder = Encoder(in_channels, n_filters)

    def forward(self, x):
        block_features = self.image_encoder(x)
        block_features = [x] + block_features
        return block_features

class PoseEncoder(nn.Module):
    """Pose_Encoder:
    input: target image (transformed image)
    ouput: gaussian maps of landmarks
    """
    def __init__(self, in_channels, n_filters, n_maps, map_sizes, gauss_std=0.1, gauss_mode='ankush'):
        super(PoseEncoder, self).__init__()
        self.map_sizes = map_sizes
        self.gauss_std = gauss_std
        self.gauss_mode = gauss_mode

        self.image_encoder = Encoder(in_channels, n_filters)
        self.conv = conv_block(n_filters*8, n_maps, kernel_size=1, stride=1, batch_norm=False, activation=None)

    def forward(self, x):
        block_features = self.image_encoder(x)
        x = block_features[-1]

        xshape = x.shape
        x = self.conv(x)

        gauss_y, gauss_y_prob = get_coord(x, 3, xshape[2])  # B,NMAP
        gauss_x, gauss_x_prob = get_coord(x, 2, xshape[3])  # B,NMAP
        gauss_mu = torch.stack([gauss_y, gauss_x], dim=2)

        gauss_xy = []
        for shape_hw in self.map_sizes:
            gauss_xy_hw = \
                get_gaussian_maps(gauss_mu, [shape_hw, shape_hw], 1.0 / self.gauss_std, mode=self.gauss_mode)
            gauss_xy.append(gauss_xy_hw)

        return gauss_mu, gauss_xy

class Decoder(nn.Module):

    def __init__(self, map_size, map_filters, n_filters, n_final_out, n_final_res, batch_norm=True):
        super(Decoder, self).__init__()
        self.seq_renderers = nn.Sequential()
        i = 1
        while map_size[0] <= n_final_res:
            self.seq_renderers.add_module('conv_render{}'.format(i), \
                conv_block(map_filters, n_filters, kernel_size=3, stride=1, batch_norm=batch_norm))

            if map_size[0] == n_final_res:
                self.seq_renderers.add_module('conv_render_final', \
                    conv_block(n_filters, n_final_out, kernel_size=3, stride=1, batch_norm=False, activation=None))
                break
            else:
                self.seq_renderers.add_module('conv_render{}'.format(i+1), \
                    conv_block(n_filters, n_filters, kernel_size=3, stride=1, batch_norm=batch_norm))
                #upsample
                map_size = [2 * s for s in map_size]
                self.seq_renderers.add_module('upsampler_render{}'.format(i+1), nn.Upsample(size=map_size))

            map_filters = n_filters
            if n_filters >= 8:
                n_filters //= 2
            i += 2

    def forward(self, x):
        x = self.seq_renderers(x)
        x = torch.sigmoid(x)
        return x