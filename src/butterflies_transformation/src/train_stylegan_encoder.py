import sys
import pickle
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from options import load_config
from datasets import CuthillDataset
from models import VGG_Encoder
from loss.lpips.lpips import LPIPS
from tools import show_reconstruction_images, init_weights

sys.path.append("externals/stylegan3")


def load_stylegan(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    return G

def generate(G, w):
    w_input = w.unsqueeze(1).repeat((1, G.num_ws, 1)).cuda()
    synth_images = G.synthesis(w_input, noise_mode='const')
    synth_images = (synth_images + 1) * (1/2)
    synth_images = synth_images.clamp(0, 1)
    return synth_images
     

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--warmup_epochs", type=int, default=3)
parser.add_argument("--lpips_lambda", type=float, default=0.1)
parser.add_argument("--l1_lambda", type=float, default=1.0)
parser.add_argument("--z_reg_lambda", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--warmup_lr", type=float, default=0.00001)
parser.add_argument("--encoder_resume", type=str, default=None)
parser.add_argument("--decoder_resume", type=str, default=None)
parser.add_argument("--stylegan", type=str, default="externals/stylegan3/output/cuthill_curated/00005-stylegan3-r-cuthill_curated-gpus4-batch32-gamma6.6/network-snapshot-000716.pkl")
args = parser.parse_args()

options = load_config('../configs/cuthill_train.yaml')

dset = CuthillDataset(options, train=True, transform=ToTensor())
dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=4)

encoder = VGG_Encoder(z_dim=64)
decoder = load_stylegan(args.stylegan).eval()

encoder.apply(init_weights)

if args.encoder_resume is not None:
    weights = torch.load(args.encoder_resume)
    encoder.load_state_dict(weights)

encoder = encoder.cuda()
decoder = decoder.cuda()

lpips_loss_fn = LPIPS()
l1_loss_fn = torch.nn.L1Loss()

def train(epoch, optimizer):
    total_lpips_loss = 0
    total_l1_loss = 0
    total_z_reg_loss = 0
    for imgs, lbls, paths in dloader:
        imgs = imgs.cuda()
        z = encoder(imgs)
        imgs_recon = generate(decoder, z)
        
        lpips_loss = lpips_loss_fn(imgs, imgs_recon)
        l1_loss = l1_loss_fn(imgs, imgs_recon)
        z_reg_loss = torch.tensor(0) #l1_loss_fn(z, torch.zeros_like(z).cuda())

        loss = lpips_loss * args.lpips_lambda + l1_loss * args.l1_lambda# + z_reg_loss * args.z_reg_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_lpips_loss += lpips_loss.item()
        total_l1_loss += l1_loss.item()
        total_z_reg_loss += z_reg_loss.item()
    print(f"Epoch: {epoch+1} | LPIPS Loss: {total_lpips_loss} | L1 Loss: {total_l1_loss} | Z reg Loss: {total_z_reg_loss}")
    show_reconstruction_images(imgs, imgs_recon, out_path="../tmp/recon_SG.png")
    torch.save(encoder.state_dict(), f"../tmp/SG_vgg_encoder_{64}.pt")
    torch.save(decoder.state_dict(), f"../tmp/SG_vgg_decoder_{64}.pt")

optimizer = torch.optim.Adam(encoder.parameters(), lr=args.warmup_lr)
for epoch in range(args.warmup_epochs):
    train(epoch, optimizer)

optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
for epoch in range(args.epochs):
    train(epoch, optimizer)