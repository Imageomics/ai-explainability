import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, pretrain=True):
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

    def get_activations(self, layer, x):
        a = layer[0](x)
        a = layer[1](a)
        a = a.view(a.size(0), -1)
        return a

    def forward(self, x, compute_z=False):
        #if compute_z:
        #    z = x.view(x.size(0), -1)
        #if compute_z:
        ###    z = self.get_activations(self.layer1, x)
        #    z = torch.cat((z, self.get_activations(self.layer1, x)), axis=1)
        x = self.layer1(x)
        #if compute_z:
        ###    z = self.get_activations(self.layer2, x)
        #    z = torch.cat((z, self.get_activations(self.layer2, x)), axis=1)
        x = self.layer2(x)
        if compute_z:
            z = self.get_activations(self.layer3, x)
            #z = torch.cat((z, self.get_activations(self.layer3, x)), axis=1)
        x = self.layer3(x)
        if compute_z:
            z = torch.cat((z, self.get_activations(self.layer4, x)), axis=1)
        x = self.layer4(x)
        if compute_z:
            z = torch.cat((z, self.get_activations(self.layer5, x)), axis=1)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #if compute_z:
        #    #z = x
        #    z = torch.cat((z, x), axis=1)
        if compute_z:
            return x, z
        return x