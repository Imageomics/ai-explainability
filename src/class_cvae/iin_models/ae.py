"""
This file has been adapted from https://raw.githubusercontent.com/CompVis/iin/master/iin/models/ae.py
Based on the CVPR paper: A Disentangling Invertible Interpretation Network for Explaining Latent Representations
Authors: Patrick Esser, Robin Rombach, Björn Ommer
"""

import functools
import torch.nn as nn
import torch
import numpy as np


class ActNorm(nn.Module):
    def __init__(self, num_features, affine=True, logdet=False):
        super().__init__()
        assert affine
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        return output / self.scale - self.loc


_norm_options = {
        "in": nn.InstanceNorm2d,
        "bn": nn.BatchNorm2d,
        "an": ActNorm}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='IN', extra_layers=0):
        super().__init__()
        self.scale = scale
        self.norm = _norm_options[norm.lower()]
        self.extra_layers = extra_layers
        if in_channels is None:
            self.in_channels = 64*min(2**(self.scale-1), 16)
        else:
            self.in_channels = in_channels
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        out_num_feat = 64*min(2**self.scale, 16)
        layers = [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_num_feat,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=out_num_feat),
                Activate()]
        
        for _ in range(self.extra_layers):
            layers += [
                nn.Conv2d(
                    in_channels=out_num_feat,
                    out_channels=out_num_feat,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                Norm(num_features=out_num_feat),
                Activate()
            ]
        self.sub_layers = nn.ModuleList(layers)


class LatentLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LatentLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
                    ])


class DecoderLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='IN', extra_layers=0):
        super().__init__()
        self.scale = scale
        self.norm = _norm_options[norm.lower()]
        self.extra_layers = extra_layers
        if in_channels is not None:
            self.in_channels = in_channels
        else:
            self.in_channels = 64*min(2**(self.scale+1), 16)
        self.build()

    def forward(self, input):
        d = input
        for layer in self.sub_layers:
            d = layer(d)
        return d

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        out_num_feat = 64*min(2**self.scale, 16)
        
        layers = [
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=out_num_feat,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=out_num_feat),
                Activate()]
          
        for _ in range(self.extra_layers):
            layers += [
                nn.Conv2d(
                    in_channels=out_num_feat,
                    out_channels=out_num_feat,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                Norm(num_features=out_num_feat),
                Activate()
            ]

        self.sub_layers = nn.ModuleList(layers)


class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None):
        super().__init__()
        self.scale = scale
        self.in_channels = 64*min(2**(self.scale-1), 16)
        if in_channels is not None:
            self.in_channels = in_channels
        self.out_channels = out_size
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True)])


class DenseDecoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, in_size):
        super().__init__()
        self.scale = scale
        self.in_channels = in_size
        self.out_channels = 64*min(2**self.scale, 16)
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True)])


class ImageLayer(nn.Module):
    def __init__(self, out_channels=3, in_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return (x + 1)/2

    def build(self):
        FinalActivate = lambda: torch.nn.Tanh()
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                FinalActivate()
                ])


class Distribution(object):
    def __init__(self, parameters, deterministic=False, num_att_vars=None):
        self.num_att_vars = num_att_vars
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.deterministic = deterministic
        std = torch.exp(0.5*self.logvar)
        var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(self.mean.get_device())
        else:
            self.var = var
            self.std = std
    def sample(self):
        if self.num_att_vars is not None:
            x_att = self.mean[:, :self.num_att_vars]
            x_var = self.mean[:, self.num_att_vars:] + self.std[:, self.num_att_vars:]*torch.randn(self.mean[:, self.num_att_vars:].shape).to(self.mean.get_device())
            return torch.cat((x_att, x_var), 1)
        x = self.mean + self.std*torch.randn(self.mean.shape).to(self.mean.get_device())
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if self.num_att_vars is not None:
                mean = self.mean[:, :self.num_att_vars]
                var = self.var[:, :self.num_att_vars]
                logvar = self.logvar[:, :self.num_att_vars]
            else:
                mean = self.mean
                var = self.var
                logvar = self.logvar
            if other is None:
                return 0.5*torch.sum(torch.pow(mean, 2)
                        + var - 1.0 - logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(mean - other.mean, 2) / other.var
                        + var / other.var - 1.0 - logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

    def mode(self):
        return self.mean


class IIN_AE(nn.Module):
    def __init__(self, n_down, z_dim, in_size, in_channels, norm, deterministic, extra_layers=0, num_att_vars=None):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.num_att_vars = num_att_vars
        n_down = n_down
        z_dim = z_dim
        in_size = in_size
        bottleneck_size = in_size // 2**n_down
        in_channels = in_channels
        norm = norm
        self.be_deterministic = deterministic

        self.feature_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm, extra_layers=extra_layers))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm, extra_layers=extra_layers))

        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, 2*z_dim)
        self.dense_decode = DenseDecoderLayer(n_down-1, bottleneck_size, z_dim)

        for scale in range(n_down-1):
            self.decoder_layers.append(DecoderLayer(scale, norm=norm, extra_layers=extra_layers))
        self.image_layer = ImageLayer(out_channels=in_channels)

        self.apply(weights_init)

        self.n_down = n_down
        self.z_dim = z_dim
        self.bottleneck_size = bottleneck_size

    def encode(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return Distribution(h, deterministic=self.be_deterministic, num_att_vars=self.num_att_vars)

    def decode(self, input):
        h = input
        h = self.dense_decode(h)
        for layer in reversed(self.decoder_layers):
            h = layer(h)
        h = self.image_layer(h)
        return h

    def get_last_layer(self):
        return self.image_layer.sub_layers[0].weight
