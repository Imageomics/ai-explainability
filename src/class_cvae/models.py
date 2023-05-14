import math

import torch
import torch.nn as nn
from torchvision import models

import functools

from iin_models.ae import IIN_AE, IIN_RESNET_AE

class VAE_Encoder(nn.Module):
    def __init__(self):
        pass

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class VAE_Decoder(nn.Module):
    def __init__(self, z_dim, n_down, ngf, nc):
        super().__init__()

        modules = []

        in_c = z_dim
        c_mul = 32
        for i in range(6):
            pad = 0 if i == 0 else 1
            modules.append(nn.ConvTranspose2d(in_c, ngf * c_mul, 4, 1, pad, bias=False))
            modules.append(nn.BatchNorm2d(ngf * c_mul))
            modules.append(nn.LeakyReLU(0.2))
            in_c = ngf * c_mul
            c_mul /= 2
        
        modules.append(nn.ConvTranspose2d(in_c, nc, 4, 1, 1, bias=False))
        modules.append(nn.Tanh())

        self.gen_net = nn.Sequential(modules)

        self.mapping_net = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim)
        )

    def generate(self, w):
        return self.gen_net(w)

    def forward(self, num, device):
        z = torch.normal(0, 1, size=(num, self.z_dim)).to(device)
        w = self.mapping_net(z)
        return self.generate(w)


# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super().__init__()
        self.net = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class GAN_VAE(nn.Module):
    def __init__(self, z_dim, n_down, num_att_vars=None, add_real_cls_vec=False, inject_z=False):
        super().__init__()
        self.z_dim = z_dim
        self.num_att_vars = num_att_vars
        self.cls_vec = None
        if add_real_cls_vec:
            self.cls_vec = nn.parameter.Parameter(torch.ones(num_att_vars), requires_grad=True)
        
        self.encoder = VAE_Encoder(z_dim, n_down)
        self.decoder = VAE_Decoder(z_dim, n_down)
        self.mapping_net = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim)
        )

        self.discriminator = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.discriminator.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1)
        )


class IIN_AE_Wrapper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_att_vars = configs.num_att_vars
        self.iin_ae = IIN_AE(configs.depth, configs.num_features, configs.img_size, configs.in_channels, \
                             'bn', False, extra_layers=configs.extra_layers, \
                             num_att_vars=configs.num_att_vars, inject_z=configs.inject_z)

        self.cls_vec = None
        if configs.add_real_cls_vec:
            self.cls_vec = nn.parameter.Parameter(torch.ones(configs.num_att_vars), requires_grad=True)

        self.z_dim = configs.num_features
        self.generator = None
        self.discriminator = None
        self.add_gan = configs.add_gan
        if configs.add_gan:
            """
            self.generator = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(z_dim, z_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(z_dim, z_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(z_dim, z_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )
            """

            if configs.use_patch_gan_dis:
                self.discriminator = NLayerDiscriminator(input_nc=configs.in_channels, ndf=64, n_layers=configs.n_disc_layers, norm_layer=nn.BatchNorm2d)
            else:
                weights = models.VGG16_BN_Weights.IMAGENET1K_V1
                self.discriminator = models.vgg16_bn(weights=weights)
                self.discriminator.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 1)
                )
            
    def get_ae_parameters(self):
        params = self.parameters()
        if not self.add_gan: return params
        dis_params = set(self.discriminator.parameters())
        rv = (p for p in params if not p in dis_params)
        return rv

    def encode(self, x):
        self.dist = self.iin_ae.encode(x)
        rv = self.dist.sample()[:, :, 0, 0]
        if self.num_att_vars is not None:
            att_vars = nn.Sigmoid()(rv[:, :self.num_att_vars])
            new_rv = torch.cat((att_vars, rv[:, self.num_att_vars:]), 1)    
            return new_rv
        #return nn.Sigmoid()(rv)
        return rv
    
    def generate(self, num, device):
        z = torch.normal(0, 1, size=(num, self.z_dim)).to(device)
        if self.num_att_vars is not None:
            z_att = torch.randint(0, 2, (num, self.num_att_vars))
            z[:, :self.num_att_vars] = z_att
        
        #w = self.generator(z)
        #w_c = self.replace(w)
        w_c = self.replace(z)

        img = self.decode(w_c)

        return img
    
    #TODO: resnet18?
    def discriminate(self, imgs):
        return nn.Sigmoid()(self.discriminator(imgs))
    
    def replace(self, z):
        if self.cls_vec is None: return z
        att_vars = z[:, :self.num_att_vars]
        feat_vars = att_vars * self.cls_vec
        feat_rv = torch.cat((feat_vars, z[:, self.num_att_vars:]), 1)
        return feat_rv
    
    def decode(self, z):
        return self.iin_ae.decode(z.unsqueeze(2).unsqueeze(3))
    
    def forward(self, x):
        return self.decode(self.encode(x))

    def kl_loss(self):
        loss = self.dist.kl()
        return torch.sum(loss) / loss.shape[0]

class ResNet50(nn.Module):
    def __init__(self, pretrain=True, num_classes=10, img_ch=1):
        super().__init__()
        weights = None
        if pretrain:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        model_resnet = models.resnet50(weights=weights)
        if img_ch != 3:
            self.conv1 = nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.linear = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x = self.get_features(x)
        x = self.linear(x)

        return x
    
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ImageClassifier(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear = nn.Linear(20*7*7, class_num)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        return out


class Encoder(nn.Module):
    def __init__(self, num_features=20, use_sigmoid=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_mean = nn.Linear(20*7*7, num_features)
        self.linear_std = nn.Linear(20*7*7, num_features)
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x, stats=False):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        mu = self.linear_mean(x)
        std = self.linear_std(x)

        z = mu + std * torch.normal(mean=0, std=1, size=std.shape).cuda()

        if self.use_sigmoid:
            z = self.sigmoid(z)
        
        if stats:
            return z, mu, std

        return z

class Decoder(nn.Module):
    def __init__(self, num_features=20):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_features, 20*7*7),
            nn.ReLU()
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=2, stride=2, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 20, -1)
        size = int(math.sqrt(x.shape[2]))
        x = x.view(x.shape[0], x.shape[1], size, size)
        x = self.convT1(x)
        x = self.convT2(x)
        x = ((x + 1) / 2)
        return x


class Classifier(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.linear = nn.Linear(in_c, out_c)

    def forward(self, x):
        return self.linear(x)


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear = nn.Linear(20*7*7, 7)

    def forward(self, x, stats=False):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        return nn.Sigmoid()(self.linear(x))

class HandCraftedMNISTDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch = len(x)
        base = torch.zeros(batch, 1, 28, 28).cuda() # Blank Images
        base[:, 0, 2:4, 6:22] = x[:, 0].view(batch, 1).repeat(1, 2*16).view(batch, 2, 16) # Top Line
        base[:, 0, 13:15, 6:22] = x[:, 1].view(batch, 1).repeat(1, 2*16).view(batch, 2, 16) # Middle Line
        base[:, 0, 24:26, 6:22] = x[:, 2].view(batch, 1).repeat(1, 2*16).view(batch, 2, 16) # Bottom Line
        base[:, 0, 4:13, 4:6] = x[:, 3].view(batch, 1).repeat(1, 2*9).view(batch, 9, 2) # Top Left Line
        base[:, 0, 4:13, 22:24] = x[:, 4].view(batch, 1).repeat(1, 2*9).view(batch, 9, 2) # Top Right Line
        base[:, 0, 15:24, 4:6] = x[:, 5].view(batch, 1).repeat(1, 2*9).view(batch, 9, 2) # Bottom Left Line
        base[:, 0, 15:24, 22:24] = x[:, 6].view(batch, 1).repeat(1, 2*9).view(batch, 9, 2) # Bottom Right Line

        return base
    
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
        --> from
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)