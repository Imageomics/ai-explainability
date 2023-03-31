import torch
import torch.nn as nn
from torchvision import transforms

class SinGAN_GenerationLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layer_size = layer_size


        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x=None):
        z = torch.normal(0, 1, size=(1, 3, self.layer_size, self.layer_size))
        z = z.to(list(self.parameters())[0].get_device())
        if x is not None:
            z += x

        out = self.layers(z)

        out = (out + 1) / 2

        return out

class SinGAN_DiscriminatorLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layer_size = layer_size
        start_num_features = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        num_blocks = 3
        if self.layer_size <= 32:
            num_blocks = 2
        
        if self.layer_size <= 16:
            num_blocks = 2

        blocks = []
        num_feat = start_num_features
        for _ in range(num_blocks):
            blocks.append(self.create_block(num_feat))
            num_feat *= 2
        self.blocks = nn.ModuleList(blocks)
        out_spatial_dim = int(self.layer_size // (2**num_blocks))
        self.fc = nn.Linear(out_spatial_dim * out_spatial_dim * num_feat, 1)

    def create_block(self, in_features):
        return nn.Sequential(
            nn.Conv2d(in_features, in_features*2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features*2, in_features*2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features*2, in_features*2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    
    def forward(self, x):
        feats = self.conv1(x)
        for block in self.blocks:
            feats = block(feats)
        out = self.fc(torch.flatten(feats, 1))
        return out
        



class SinGAN(nn.Module):
    def __init__(self, num_levels=4, img_size=128):
        super().__init__()
        self.num_levels = num_levels
        self.img_size = img_size

        gen_layers = []
        disc_layers = []
        for i in range(num_levels):
            layer_size = int(self.img_size / (2**i))
            gen_layers.append(SinGAN_GenerationLayer(layer_size))
            disc_layers.append(SinGAN_DiscriminatorLayer(layer_size))
        
        self.start_size = int(layer_size)

        self.gen_layers = nn.ModuleList(gen_layers)
        self.disc_layers = nn.ModuleList(disc_layers)

    def forward(self, x):
        fake = None
        layer_outputs = []
        for i in range(self.num_levels):
            cur_size = int(self.start_size * (2**i))
            layer_pos = self.num_levels - (i+1)
            real = transforms.functional.resize(x, (cur_size, ))
            fake = self.gen_layers[layer_pos](fake)
            real_out = self.disc_layers[layer_pos](real)
            fake_out = self.disc_layers[layer_pos](fake)
            layer_outputs.append([real, fake, real_out, fake_out])
            fake = transforms.functional.resize(fake, (cur_size*2, ))

        return layer_outputs

