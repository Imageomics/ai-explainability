import os

import numpy as np

import torch
import torchvision.transforms as T

from PIL import Image

from arch import SinGAN

global_configs = {
    "img_path" : "/home/carlyn.1/ai_explanability/data/dogs.jpg",
    "save_dir" : "/home/carlyn.1/ai_explanability/tmp/"
}

def load_data():
    im = Image.open(global_configs['img_path'])
    img_tran = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    return img_tran(im).unsqueeze(0)

def load_model():
    return SinGAN(img_size=128, num_levels=4)

def train(X, G):
    G = G.cuda()
    X = X.cuda()
    params = G.parameters()
    optimizer = torch.optim.Adam(params, lr=0.0001)
    l1_loss = torch.nn.L1Loss()
    for epoch in range(100000):
        layer_outputs = G(X)
        wgan_gp_loss = torch.zeros((1, 1)).cuda()
        recon_loss = torch.zeros((1, 1)).cuda()
        for i, (real, fake, real_out, fake_out) in enumerate(layer_outputs):
            wgan_gp_loss += torch.abs(fake_out - real_out) # TODO Need to add the regularization on gradient of the input: https://paperswithcode.com/method/wgan-gp-loss

            real_arr = np.array(T.ToPILImage()(real[0]))
            fake_arr = np.array(T.ToPILImage()(fake[0]))
            layer_out_img = Image.fromarray(np.concatenate((real_arr, fake_arr), axis=0))
            layer_out_img.save(os.path.join(global_configs['save_dir'], f"layer_{i}.png"))

            recon_loss += l1_loss(real, fake)

        loss = wgan_gp_loss + recon_loss

        print(f"Epoch {epoch+1}: Loss - {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

def save(G):
    pass

if __name__ == "__main__":
    X = load_data()
    G = load_model()

    train(X, G)

    save(G)