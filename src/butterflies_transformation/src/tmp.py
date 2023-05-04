from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose

from options import load_config
from datasets import CuthillDataset, MyersJiggins
from models import VGG_VEncoder, Decoder, VGG_Decoder
from loss.lpips.lpips import LPIPS
from tools import show_reconstruction_images, init_weights


parser = ArgumentParser()
parser.add_argument("--z_dim", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--lpips_lambda", type=float, default=0.1)
parser.add_argument("--l1_lambda", type=float, default=1.0)
parser.add_argument("--z_reg_lambda", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--warmup_lr", type=float, default=0.0003)
parser.add_argument("--encoder_resume", type=str, default=None)
parser.add_argument("--decoder_resume", type=str, default=None)
parser.add_argument("--dset", type=str, default="cuthill", choices=["cuthill", "myers-jiggins"])
args = parser.parse_args()

if args.dset == "cuthill":
    start_size = 8
    options = load_config('../configs/cuthill_train.yaml')
    dset = CuthillDataset(options, train=True, transform=ToTensor())
elif args.dset == "myers-jiggins":
    start_size = 32
    options = load_config('../configs/myers_jiggins_train.yaml')
    transforms = Compose([
        Resize((512, 512)),
        ToTensor()
    ])
    dset = MyersJiggins(options, train=True, transform=transforms)

dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=4)

encoder = VGG_VEncoder(z_dim=args.z_dim)
decoder = VGG_Decoder(z_dim=args.z_dim, start_size=start_size)

encoder.apply(init_weights)
decoder.apply(init_weights)

if args.encoder_resume is not None:
    weights = torch.load(args.encoder_resume)
    encoder.load_state_dict(weights)
if args.decoder_resume is not None:
    weights = torch.load(args.decoder_resume)
    decoder.load_state_dict(weights)

encoder = encoder.cuda()
decoder = decoder.cuda()

lpips_loss_fn = LPIPS()
l1_loss_fn = torch.nn.L1Loss()
normal_loss_fn = lambda mu, sigma: (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum(1).mean()

def train(epoch, optimizer):
    total_lpips_loss = 0
    total_l1_loss = 0
    total_z_reg_loss = 0
    for imgs, lbls, paths in tqdm(dloader):
        imgs = imgs.cuda()
        z, mu, sigma = encoder(imgs, stats=True)
        imgs_recon = decoder(z)
        
        
        lpips_loss = lpips_loss_fn(imgs, imgs_recon)
        l1_loss = l1_loss_fn(imgs, imgs_recon)
        z_reg_loss = normal_loss_fn(mu, sigma)

        loss = lpips_loss * args.lpips_lambda + l1_loss * args.l1_lambda + z_reg_loss * args.z_reg_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_lpips_loss += lpips_loss.item()
        total_l1_loss += l1_loss.item()
        total_z_reg_loss += z_reg_loss.item()
    print(f"Epoch: {epoch+1} | LPIPS Loss: {total_lpips_loss} | L1 Loss: {total_l1_loss} | Z reg Loss: {total_z_reg_loss}")
    show_reconstruction_images(imgs, imgs_recon)
    torch.save(encoder.state_dict(), f"../tmp/vgg_vencoder_{args.z_dim}.pt")
    torch.save(decoder.state_dict(), f"../tmp/vgg_vdecoder_{args.z_dim}.pt")

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.warmup_lr)
for epoch in range(args.warmup_epochs):
    train(epoch, optimizer)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
for epoch in range(args.epochs):
    train(epoch, optimizer)