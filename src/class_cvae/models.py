import math

import torch
import torch.nn as nn
from torchvision import models

from iin_models.ae import IIN_AE, IIN_RESNET_AE

class IIN_AE_Wrapper(nn.Module):
    def __init__(self, n_down, z_dim, in_size, in_channels, norm, deterministic, \
                 extra_layers=0, num_att_vars=None, add_real_cls_vec=False, resnet=None, \
                 inject_z=False, add_gan=False):
        super().__init__()
        self.num_att_vars = num_att_vars
        if resnet is None:
            self.iin_ae = IIN_AE(n_down, z_dim, in_size, in_channels, norm, deterministic, \
                                extra_layers=extra_layers, num_att_vars=num_att_vars, inject_z=inject_z)
        else:
            self.iin_ae = IIN_RESNET_AE(resnet, n_down, z_dim, in_size, in_channels, norm, deterministic, \
                                extra_layers=extra_layers, num_att_vars=num_att_vars)
        self.cls_vec = None
        if add_real_cls_vec:
            self.cls_vec = nn.parameter.Parameter(torch.ones(num_att_vars), requires_grad=True)

        self.z_dim = z_dim
        self.generator = None
        self.discriminator = None
        if add_gan:
            self.generator = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim),
                nn.Linear(z_dim, z_dim)
            )
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1
            self.discriminator = models.vgg16_bn(weights=weights)
            self.discriminator.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 1)
            )
            

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
            
        w = self.generator(z)
        w_c = self.replace(w)

        img = self.decode(w_c)

        return img
    
    #TODO: resnet18?
    def discriminate(self, imgs):
        return self.discriminator(imgs)
    
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
        return self.dist.kl().mean()

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