import torch
import torch.nn as nn
from torchvision import models

class Res50(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        model_resnet = models.resnet50(pretrained=pretrain)
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

    def get_activations(self, layer, x):
        a = layer[0].conv1(x)
        #a = layer[0].bn1(a)
        #a = layer[0].conv2(a)
        #a = layer[0].bn2(a)
        #a = layer[0].conv3(a)
        #a = layer[0].bn3(a)
        #a = layer[0].relu(a)
        #a = layer[0](x) # Get output of second block
        a = nn.ReLU()(a)
        a = a.view(a.size(0), -1)
        return a

    def forward(self, x, compute_z=False):
        #if compute_z:
        #    z = x.view(x.size(0), -1)
        x = self.conv1(x)
        #if compute_z:
        #    a = nn.ReLU()(x)
        #    a = a.view(a.size(0), -1)
        #    z = a
        #    z = torch.cat((z, a), axis=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ##if compute_z:
            #z = self.get_activations(self.layer1, x)
            #z = torch.cat((z, self.get_activations(self.layer1, x)), axis=1)
        x = self.layer1(x)
        if compute_z:
            z = self.get_activations(self.layer2, x)
            #z = torch.cat((z, self.get_activations(self.layer2, x)), axis=1)
        x = self.layer2(x)
        if compute_z:
        #####    z = self.get_activations(self.layer3, x)
            z = torch.cat((z, self.get_activations(self.layer3, x)), axis=1)
        x = self.layer3(x)
        if compute_z:
        ##    #z = self.get_activations(self.layer4, x)
        ##    z = self.get_activations(self.layer4, x)
            z = torch.cat((z, self.get_activations(self.layer4, x)), axis=1)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #if compute_z:
            #z = x
            #z = torch.cat((z, x), axis=1)

        if compute_z:
            return x, z
        return x

class Res101(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        model_resnet = models.resnet101(pretrained=pretrain)
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

    def forward(self, x):
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