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

# Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/wgan_gp/wgan_gp.py#L171
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = alpha.to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = fake.to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
    G_optimizer = torch.optim.Adam(G.get_G_parameters(), lr=0.0001)
    D_optimizer = torch.optim.Adam(G.get_D_parameters(), lr=0.0001)
    l1_loss = torch.nn.L1Loss()
    for epoch in range(100000):
        # Discriminator Loss
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

        layer_outputs = G(X)
        d_loss = torch.zeros((1, 1)).cuda()
        recon_loss = torch.zeros((1, 1)).cuda()
        for i, (real, fake, real_out, fake_out) in enumerate(layer_outputs):
            d_loss += (fake_out - real_out)
            gradient_penalty = compute_gradient_penalty(G.disc_layers[-(i+1)], real, fake)
            d_loss += 10 * gradient_penalty
            recon_loss += l1_loss(real, fake)

        d_loss += recon_loss

        d_loss.backward()
        D_optimizer.step()

        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

        layer_outputs = G(X, no_noise=True)
        g_loss = torch.zeros((1, 1)).cuda()
        recon_loss = torch.zeros((1, 1)).cuda()
        for i, (real, fake, real_out, fake_out) in enumerate(layer_outputs):
            g_loss += -fake_out
            recon_loss += l1_loss(real, fake)

            real_arr = np.array(T.ToPILImage()(real[0]))
            fake_arr = np.array(T.ToPILImage()(fake[0]))
            layer_out_img = Image.fromarray(np.concatenate((real_arr, fake_arr), axis=0))
            layer_out_img.save(os.path.join(global_configs['save_dir'], f"layer_{i}.png"))

        g_loss += recon_loss
        g_loss.backward()
        G_optimizer.step()

        print(f"Epoch {epoch+1}: Recon Loss - {recon_loss.item()} | D Loss - {d_loss.item()} | G Loss - {g_loss.item()}")


        

def save(G):
    pass

if __name__ == "__main__":
    X = load_data()
    G = load_model()

    train(X, G)

    save(G)