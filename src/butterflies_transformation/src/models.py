import math

import torch
import torch.nn as nn

from torchvision import models

class VGG_Encoder(nn.Module):
    def __init__(self, z_dim=512, pretrain=True):
        super().__init__()
        self.z_dim=z_dim
        model_vgg = models.vgg16(pretrained=pretrain)
        self.layer1 = nn.Sequential(
            model_vgg.features[0], # CONV
            model_vgg.features[1], # RELU
            model_vgg.features[2], # CONV
            model_vgg.features[3], # RELU
            model_vgg.features[4], # MAXPOOL
        )
        self.layer2 = nn.Sequential(
            model_vgg.features[5],
            model_vgg.features[6],
            model_vgg.features[7],
            model_vgg.features[8],
            model_vgg.features[9],
        )

        self.layer3 = nn.Sequential(
            model_vgg.features[10],
            model_vgg.features[11],
            model_vgg.features[12],
            model_vgg.features[13],
            model_vgg.features[14],
            model_vgg.features[15],
            model_vgg.features[16],
        )

        self.layer4 = nn.Sequential(
            model_vgg.features[17],
            model_vgg.features[18],
            model_vgg.features[19],
            model_vgg.features[20],
            model_vgg.features[21],
            model_vgg.features[22],
            model_vgg.features[23],
        )

        self.layer5 = nn.Sequential (
            model_vgg.features[24],
            model_vgg.features[25],
            model_vgg.features[26],
            model_vgg.features[27],
            model_vgg.features[28],
            model_vgg.features[29],
            model_vgg.features[30],
        )

        self.avgpool = model_vgg.avgpool

        self.in_features = model_vgg.classifier[0].in_features
        self.linear = nn.Linear(self.in_features, self.z_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.linear(x)

        return z
    
class VGG_VEncoder(nn.Module):
    def __init__(self, z_dim=512, pretrain=True):
        super().__init__()
        self.z_dim=z_dim
        model_vgg = models.vgg16(pretrained=pretrain)
        self.layer1 = nn.Sequential(
            model_vgg.features[0], # CONV
            model_vgg.features[1], # RELU
            model_vgg.features[2], # CONV
            model_vgg.features[3], # RELU
            model_vgg.features[4], # MAXPOOL
        )
        self.layer2 = nn.Sequential(
            model_vgg.features[5],
            model_vgg.features[6],
            model_vgg.features[7],
            model_vgg.features[8],
            model_vgg.features[9],
        )

        self.layer3 = nn.Sequential(
            model_vgg.features[10],
            model_vgg.features[11],
            model_vgg.features[12],
            model_vgg.features[13],
            model_vgg.features[14],
            model_vgg.features[15],
            model_vgg.features[16],
        )

        self.layer4 = nn.Sequential(
            model_vgg.features[17],
            model_vgg.features[18],
            model_vgg.features[19],
            model_vgg.features[20],
            model_vgg.features[21],
            model_vgg.features[22],
            model_vgg.features[23],
        )

        self.layer5 = nn.Sequential (
            model_vgg.features[24],
            model_vgg.features[25],
            model_vgg.features[26],
            model_vgg.features[27],
            model_vgg.features[28],
            model_vgg.features[29],
            model_vgg.features[30],
        )

        self.avgpool = model_vgg.avgpool

        self.in_features = model_vgg.classifier[0].in_features
        self.mean_linear = nn.Linear(self.in_features, self.z_dim)
        self.std_linear = nn.Linear(self.in_features, self.z_dim)

    def forward(self, x, stats=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        mu = self.mean_linear(x)
        sigma = torch.exp(self.std_linear(x))

        z = mu + sigma * torch.normal(mean=0, std=1, size=sigma.shape).cuda()

        if stats:
            return z, mu, sigma

        return z

class VGG_Classifier(nn.Module):
    def __init__(self, class_num=18, pretrain=True):
        super().__init__()
        model_vgg = models.vgg16(pretrained=pretrain)
        self.layer1 = nn.Sequential(
            model_vgg.features[0], # CONV
            model_vgg.features[1], # RELU
            model_vgg.features[2], # CONV
            model_vgg.features[3], # RELU
            model_vgg.features[4], # MAXPOOL
        )
        self.layer2 = nn.Sequential(
            model_vgg.features[5],
            model_vgg.features[6],
            model_vgg.features[7],
            model_vgg.features[8],
            model_vgg.features[9],
        )

        self.layer3 = nn.Sequential(
            model_vgg.features[10],
            model_vgg.features[11],
            model_vgg.features[12],
            model_vgg.features[13],
            model_vgg.features[14],
            model_vgg.features[15],
            model_vgg.features[16],
        )

        self.layer4 = nn.Sequential(
            model_vgg.features[17],
            model_vgg.features[18],
            model_vgg.features[19],
            model_vgg.features[20],
            model_vgg.features[21],
            model_vgg.features[22],
            model_vgg.features[23],
        )

        self.layer5 = nn.Sequential (
            model_vgg.features[24],
            model_vgg.features[25],
            model_vgg.features[26],
            model_vgg.features[27],
            model_vgg.features[28],
            model_vgg.features[29],
            model_vgg.features[30],
        )

        self.avgpool = model_vgg.avgpool

        self.in_features = model_vgg.classifier[0].in_features
        self.linear = nn.Linear(self.in_features, class_num)

    def forward(self, x, compute_z=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x

class Encoder(nn.Module):
    def __init__(self, z_dim=32) -> None:
        super().__init__()
        self.z_dim=z_dim

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.linear = nn.Linear(16*16*128, z_dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        z = self.linear(out.view(out.shape[0], -1))
        return z

class Decoder(nn.Module):
    def __init__(self, z_dim=32, out_size=128) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.out_size = out_size

        self.linear = nn.Sequential(
            nn.Linear(z_dim, 8*8*256),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 256, -1)
        size = int(math.sqrt(out.shape[2]))
        out = out.view(out.shape[0], out.shape[1], size, size)
        out = self.avg_pool(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        x = self.block4(out)
        x = ((x + 1) / 2)
        return x

class VGG_Decoder(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(z_dim, 8*8*256),
            nn.ReLU()
        )

        self.decov1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decov2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decov3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decov4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decov5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decov6 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        
    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 256, -1)
        size = int(math.sqrt(out.shape[2]))
        x = out.view(out.shape[0], out.shape[1], size, size)
        x = self.relu(self.decov1(x))
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.relu(self.conv1_3(x))
        x = self.relu(self.decov2(x))
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.relu(self.conv2_3(x))
        x = self.relu(self.decov3(x))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.decov4(x))
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.decov5(x))
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.tanh(self.decov6(x))

        x = ((x + 1) / 2)
        return x
    
