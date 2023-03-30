

from argparse import ArgumentParser

import copy
import os
from time import perf_counter
from typing import Optional

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

import styleGAN.dnnlib as dnnlib
import styleGAN.legacy as legacy
import sys

from models import Classifier, VGG16
from helpers import cuda_setup, set_random_seed

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def image_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.01,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    smooth_beta                = 2,
    smooth_eps                 = 1e-3,
    smooth_lambda              = 1e-3,
    mse_lambda                 = 0.2,
    norm_grad                  = False,
    projected_w: Optional[str]
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).cuda() # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().cuda()

    # Features for target image.
    target_images = target.unsqueeze(0).cuda().to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32).cuda().requires_grad_() # pylint: disable=not-callable
    if projected_w is not None:
        saved_w = torch.tensor(np.load(projected_w)['w'][0][0]) # 512 dimension
        saved_w = saved_w[None, None, :] # [1,1,512]
        w_opt = torch.tensor(saved_w).cuda().requires_grad_()

    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32).cuda()
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # MSE between target image and generated image
    MSELoss = nn.MSELoss().cuda()

    # smooth loss
    def smooth_loss_f(z):
        x_diff = z[:, :-1, :-1] - z[:, :-1, 1:]
        y_diff = z[:, :-1, :-1] - z[:, 1:, :-1]
        sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, smooth_eps, 10000000)
        return torch.norm(sq_diff, smooth_beta / 2.0) ** (smooth_beta / 2.0)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # MSE between target image and generated image
        mse_loss = MSELoss(target_images, synth_images) * mse_lambda

        # smooth loss
        smooth_loss = smooth_loss_f(synth_images) * smooth_lambda

        loss = dist + reg_loss * regularize_noise_weight + mse_loss + smooth_loss


        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Normalized gradient
        if norm_grad:
            print('normalizng gradient...')
            grad_norm = torch.norm(w_opt.grad, p=2)
            w_opt.grad = w_opt.grad / grad_norm

        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f} mse loss {float(mse_loss):<5.2f} smooth loss {float(smooth_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

def self_entropy(x):
    x = (np.exp(x) / np.exp(x).sum())
    return -(x * np.log(x)).sum()

#----------------------------------------------------------------------------

def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    num_steps: int,
    smooth_beta: float,
    smooth_eps: float,
    smooth_lambda: float,
    mse_lambda: float,
    norm_grad: bool,
    projected_w: Optional[str],
    backbone: Optional[str],
    classifier: Optional[str],
    target_lbl: int
):

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).cuda() # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    #target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_pil.save("test.png")
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1])).cuda(), # pylint: disable=not-callable
        num_steps=num_steps,
        verbose=True,
        smooth_beta=smooth_beta,
        smooth_eps=smooth_eps,
        smooth_lambda=smooth_lambda,
        mse_lambda=mse_lambda,
        norm_grad=norm_grad,
        projected_w=projected_w
    )
    print(projected_w_steps.shape)
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    weights = torch.load(backbone)
    class_weights = torch.load(classifier)

    backbone = VGG16(pretrain=False).cuda()
    backbone.load_state_dict(weights)
    classifier = Classifier(backbone.in_features, 34).cuda()
    classifier.load_state_dict(class_weights)
    classifier.eval()
    backbone.eval()

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    diffs = []
    imgs = []
    confs = []
    w_entropy = []
    heatmaps = []
    sm = nn.Softmax(dim=0)
    if save_video:
        
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        max_v = 0
        original_w = projected_w_steps[0].cpu().numpy()[0]
        for w_i, projected_w in enumerate(projected_w_steps):
            if w_i == 0:
                w_entropy.append(0)
                heatmaps.append(np.zeros_like(projected_w.cpu().numpy()[0]))
            else:
                w_entropy.append(self_entropy(projected_w.cpu().numpy()[0] - projected_w_steps[w_i-1].cpu().numpy()[0]))
                heatmaps.append(projected_w.cpu().numpy()[0] - projected_w_steps[w_i-1].cpu().numpy()[0])
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            diff_image = np.zeros_like(synth_image)
            if len(imgs) > 0:
                diff_image = ((synth_image - imgs[-1]) ** 2).sum(2).astype(np.uint8)
                max_v = max(max_v, diff_image.max())
                diff_image = np.transpose(np.stack((diff_image, diff_image, diff_image)), (1, 2, 0))
            diffs.append(diff_image)
            imgs.append(np.copy(synth_image))
            out = classifier(backbone(image_transform()(PIL.Image.fromarray(synth_image, 'RGB')).unsqueeze(0).cuda()))[0]
            confs.append(round(sm(out)[target_lbl].item(), 4) * 100)
            #print(round(sm(out)[target_lbl].item(), 4) * 100)
            out = classifier(backbone(image_transform()(PIL.Image.fromarray(target_uint8, 'RGB')).unsqueeze(0).cuda()))[0]
            #print(round(sm(out)[target_lbl].item(), 4) * 100)
            #print("="*30)
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        px = 1/plt.rcParams['figure.dpi']
        for i in range(len(imgs)):
            synth_image = imgs[i]
            diff_image = ((diffs[i] / max_v) * 255).astype(np.uint8)
            fig = plt.figure(figsize=(target_uint8.shape[1]*px, target_uint8.shape[0]*px))
            plt.plot(confs, color="blue", label="Confidence")
            plt.axvline(x=i, color="yellow")
            plt.gca().get_xaxis().set_visible(False)
            plt.legend()
            fig.canvas.draw()
            conf_graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            conf_graph = conf_graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()

            fig = plt.figure(figsize=(target_uint8.shape[1]*px, target_uint8.shape[0]*px))
            plt.plot(w_entropy, color="green", label="Entropy")
            plt.axvline(x=i, color="yellow")
            plt.gca().get_xaxis().set_visible(False)
            plt.legend()
            fig.canvas.draw()
            entropy_graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            entropy_graph = entropy_graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()

            fig = plt.figure(figsize=(target_uint8.shape[1]*px, target_uint8.shape[0]*px))
            heatmap = heatmaps[i].reshape((32, 16))
            hm = plt.imshow(heatmap)
            plt.colorbar(hm)
            fig.canvas.draw()
            heatmap_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            heatmap_fig = heatmap_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()

            video.append_data(np.concatenate([np.concatenate([target_uint8, synth_image, diff_image], axis=1), np.concatenate([conf_graph, entropy_graph, heatmap_fig], axis=1)], axis=0))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

# Get args
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--network', type=str, help='Network pickle filename')
    parser.add_argument('--target', type=str, help='Target image file to project to')
    parser.add_argument('--num-steps',              help='Number of optimization steps', type=int, default=1000)
    parser.add_argument('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True)
    parser.add_argument('--outdir',                 help='Where to save the output images', type=str)
    parser.add_argument('--smooth_beta',            help='Smooth beta', type=float, default=2)
    parser.add_argument('--smooth_eps',             help='Smooth eps', type=float, default=1e-3)
    parser.add_argument('--smooth_lambda',          help='Smooth lambda', type=float, default=0.0)
    parser.add_argument('--mse_lambda',             help='MSE lambda', type=float, default=0.0)
    parser.add_argument('--norm_grad',              help='Normalizing gradient', type=bool, default=False)
    parser.add_argument('--projected-w', help='Projection result file', type=str)
    parser.add_argument('--backbone', help='feature weights', type=str)
    parser.add_argument('--classifier', help='classifier weights', type=str)
    parser.add_argument('--target_lbl', help='target label', type=int, default=17)

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

# === MAIN === #
if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    run_projection(
        args.network,
        args.target,
        args.outdir,
        args.save_video,
        args.num_steps,
        args.smooth_beta,
        args.smooth_eps,
        args.smooth_lambda,
        args.mse_lambda,
        args.norm_grad,
        args.projected_w,
        args.backbone,
        args.classifier,
        args.target_lbl
    )

