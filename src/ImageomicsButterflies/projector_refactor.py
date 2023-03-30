from argparse import ArgumentParser
from audioop import avg

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
import matplotlib.colors as mcolors

import styleGAN.dnnlib as dnnlib
import styleGAN.legacy as legacy
import sys

import cv2

from models import Classifier, VGG16
from helpers import cuda_setup, set_random_seed

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
UNNORMALIZE = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.229, 1/0.225])

def image_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])


def to_hsv(x):
    return mcolors.rgb_to_hsv(x)

def get_lightness(x):
    cmax = x.max(2)
    cmin = x.min(2)
    return (cmax-cmin) / 2


#----------------------------------------------------------------------------
def load_img(img_path, resolution=128):
    target_pil = PIL.Image.open(img_path).convert('RGB') # (res, res, # channels)
    target_pil = target_pil.resize((resolution, resolution), PIL.Image.LANCZOS)
    return transforms.ToTensor()(target_pil)
#----------------------------------------------------------------------------
def load_imgs(img_dir, view="D", max_size=8, resolution=128):
    if img_dir is None:
        return None

    if os.path.isfile(img_dir):
        img = load_img(img_dir, resolution)
        return img.unsqueeze(0).cuda()
    
    batch = None
    i = 0
    for root, dirs, paths in os.walk(img_dir):
        for path in paths:
            fview = path.split(".")[0].split("_")[1]
            if fview != view: continue
            i += 1

            full_path = os.path.join(root, path)
            img = load_img(full_path, resolution)
            if batch is None:
                batch = img.unsqueeze(0)
            else:
                batch = torch.cat((batch, img.unsqueeze(0)), axis=0)

            if i >= max_size: break
            
    batch = batch.cuda()
    return batch
#----------------------------------------------------------------------------

# Get args
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--network', type=str, help='Network pickle filename')
    parser.add_argument('--target', type=str, help='Target directory of image files to project to')
    parser.add_argument('--buckets',              help='Interval of steps to take diff images', type=int, default=1000)
    parser.add_argument('--num-steps',              help='Number of optimization steps', type=int, default=1000)
    parser.add_argument('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True)
    parser.add_argument('--outdir',                 help='Where to save the output images', type=str)
    parser.add_argument('--smooth_beta',            help='Smooth beta', type=float, default=2)
    parser.add_argument('--smooth_eps',             help='Smooth eps', type=float, default=1e-3)
    parser.add_argument('--smooth_lambda',          help='Smooth lambda', type=float, default=0.0)
    parser.add_argument('--mse_lambda',             help='MSE lambda', type=float, default=0.0)
    parser.add_argument('--lr',             help='learning rate', type=float, default=0.001)
    parser.add_argument('--norm_grad',              help='Normalizing gradient', action="store_true", default=False)
    parser.add_argument('--avg_grad',              help='Average gradient', action="store_true", default=False)
    parser.add_argument('--min_grad',              help='min gradient', action="store_true", default=False)
    parser.add_argument('--img_to_img',              help='img to img', action="store_true", default=False)
    parser.add_argument('--latents', help='Projection result file', type=str)
    parser.add_argument('--backbone', help='feature weights', type=str)
    parser.add_argument('--classifier', help='classifier weights', type=str)
    parser.add_argument('--target_lbl', help='target label', type=int, default=17)
    parser.add_argument('--learn_param', help='learnable param', type=str, default="z")

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def load_data(latent_path=None, img_path=None, resolution=128):
    projected_ws = None
    projected_zs = None
    if latent_path is not None:
        latents = np.load(latent_path)
        if 'w' in latents.keys():
            projected_ws = torch.tensor(latents['w'][:][0]).cuda()
        if 'z' in latents.keys():
            projected_zs = torch.tensor(latents['z']).cuda()

    images = load_imgs(img_path, resolution=resolution)
    min_size = len(images)
    if projected_zs is not None:
        min_size = min(min_size, len(projected_zs))
    if projected_ws is not None:
        min_size = min(min_size, len(projected_ws))

    images = images[:min_size]
    if projected_zs is not None:
        projected_zs = projected_zs[:min_size]
    if projected_ws is not None:
        projected_ws = projected_ws[:min_size]
    
    return images, projected_ws, projected_zs


def load_models(gen_path, f_path=None, c_path=None):
    # Load networks.
    print('Loading networks from "%s"...' % gen_path)
    with dnnlib.util.open_url(gen_path) as fp:
        net = legacy.load_network_pkl(fp)
        # Generator
        G = net['G_ema'].requires_grad_(False).cuda() # type: ignore
        G.eval()

        # Discriminator
        D = net['D'].requires_grad_(False).cuda() # type: ignore
        D.eval()

    # Feature Extractor
    F = None
    if f_path is not None:
        f_weights = torch.load(f_path)
        F = VGG16(pretrain=False).cuda()
        F.load_state_dict(f_weights)
        F.eval()

    # Classifier
    C = None
    if c_path is not None:
        c_weights = torch.load(c_path)
        C = Classifier(F.in_features, 34).cuda()
        C.load_state_dict(c_weights)
        C.eval()

    return G, D, F, C

def load_latents(G, images, projected_ws, projected_zs, w_avg_samples=10000):
    if projected_zs is not None:
        projected_ws = G.mapping(projected_zs, None)
    elif projected_ws is None:
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        projected_zs = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        projected_zs = torch.from_numpy(projected_zs).cuda()
        w_samples = G.mapping(projected_zs, None)  # [N, L, C]
        w_samples = w_samples[:, :, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        projected_ws = torch.from_numpy(np.tile(np.mean(w_samples, axis=0, keepdims=True)[0], (len(images), 1, 1))).cuda()      # [1, 1, C]
        projected_zs = projected_zs.mean(0, keepdim=True).repeat([len(images), 1])

    return projected_zs, projected_ws

def project_tmp(
    images,
    G,
    D,
    F,
    C,
    target_lbl,
    learn_param                = "w",
    projected_zs               = None,
    projected_ws               = None,
    num_steps                  = 200,
    init_lr                    = 0.001,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    norm_grad                  = False,
    avg_grad                   = False,
    min_grad                   = False,
    img_to_img                 = False


):
    if images is not None:
        assert images[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).cuda() # type: ignore
    D = copy.deepcopy(D).eval().requires_grad_(False).cuda() # type: ignore

    # Load latents
    start_zs, start_ws = load_latents(G, images, projected_ws, projected_zs)

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    feat_extractor = None
    if F is None:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().cuda()
        feat_extractor = lambda x: vgg16(x, resize_images=False, return_lpips=True)
    else:
        feat_extractor = lambda x: F(NORMALIZE(x))

    # Features for target image.
    target_images = images.to(torch.float32)
    if target_images.shape[2] > 128:
        target_images = F.interpolate(target_images, size=(128, 128), mode='area')

    feat_mul = 255 if F is None else 1
    target_features = feat_extractor(target_images * feat_mul)

    if learn_param == "w":
        tmp = start_ws[:, 0, :].unsqueeze(1).clone().detach().cuda()
        org_param = tmp.clone().detach().cuda()
        learnable = torch.zeros_like(tmp[0]).cuda().requires_grad_()
        new_param = (start_ws[:, 0, :].unsqueeze(1) + learnable.unsqueeze(0).repeat([len(start_ws), 1, 1])).cuda()
        w_opt = new_param.clone().repeat([1, G.mapping.num_ws, 1])
    elif learn_param == "z":
        tmp = start_zs.clone().detach().cuda()
        if img_to_img:
            learnable = start_zs.clone().detach().cuda().requires_grad_()
            w_opt = G.mapping(learnable, None)
        else:
            learnable = torch.zeros_like(tmp[0]).cuda().requires_grad_()
            org_param = start_zs.clone().detach().cuda()
            new_param = start_zs + learnable.repeat([len(start_zs), 1])
            w_opt = G.mapping(new_param, None)

    w_out = torch.zeros([num_steps] + [start_ws.shape[0], start_ws.shape[2]], dtype=torch.float32)
    z_out = None
    if learn_param == "z":
        z_out = torch.zeros([num_steps] + list(start_zs.shape), dtype=torch.float32)
    optimizer = torch.optim.Adam([learnable], betas=(0.9, 0.999), lr=init_lr)

    # MSE between target image and generated image
    MSELoss = nn.MSELoss(reduction='none').cuda()
    CELoss = nn.CrossEntropyLoss().cuda()
    L1Loss = nn.L1Loss().cuda()
    sm = nn.Softmax(dim=2).cuda()
    

    # smooth loss
    #def smooth_loss_f(z):
    #    x_diff = z[:, :-1, :-1] - z[:, :-1, 1:]
    #    y_diff = z[:, :-1, :-1] - z[:, 1:, :-1]
    #    sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, smooth_eps, 10000000)
    #    return torch.norm(sq_diff, smooth_beta / 2.0) ** (smooth_beta / 2.0)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    all_synth_images = []

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = init_lr * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        if False: # add param for noise
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(w_opt, noise_mode='const')

        def min_grad_learn(grad):
            if min_grad and False:
                grad_imgs = torch.abs(grad).sum(1)
                #grad_imgs -= grad_imgs.view(len(grad), -1).min(1)[0].view(-1, 1, 1).repeat([1] + list(grad.shape[2:]))
                #grad_imgs /= grad_imgs.view(len(grad), -1).max(1)[0].view(-1, 1, 1).repeat([1] + list(grad.shape[2:]))
                for i in range(len(grad_imgs)):
                    grad_imgs[i] = torch.from_numpy(cv2.medianBlur(grad_imgs[i].detach().cpu().numpy(), 5)).cuda()
                thresh = 0.1
                thresh_mask = grad_imgs < thresh
                grad_imgs[thresh_mask] = 0
                tmp = grad_imgs[0].detach().cpu()
                tmp -= tmp.min()
                tmp /= tmp.max()
                transforms.ToPILImage()(tmp).save("grad_img.png")
                thresh_mask = thresh_mask.unsqueeze(1).repeat([1, 3, 1, 1])
                grad[thresh_mask] = 0
            return grad

        min_hook = synth_images.register_hook(min_grad_learn)

        real_loss = 0
        #if D is not None:
        #    logits = D(synth_images, None).cuda()
        #    real_loss = torch.nn.functional.softplus(-logits)
        synth_images = (synth_images + 1) * (1/2)
        synth_images = synth_images.clamp(0, 1)
        if step % 100 == 0:
            transforms.ToPILImage()(synth_images[0]).save("tmp.png")
        all_synth_images.append(synth_images.detach().cpu())
        #transforms.ToPILImage()(input_img[0]).save("test_img.png")
        #transforms.ToPILImage()(UNNORMALIZE(target_images[0])).save("test_img2.png")
        #input_img = input_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        #input_img = PIL.Image.fromarray(input_img, 'RGB')
        #input_img.save("test.png")
        if C is not None:
            out = C(feat_extractor(synth_images))
            lbls = torch.tensor([target_lbl] * len(images)).cuda()
            class_loss = CELoss(out, lbls)

        if not img_to_img:
            min_loss = L1Loss(learnable, torch.zeros_like(learnable))

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        #if synth_images.shape[2] > 256:
        #    synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = feat_extractor(synth_images * feat_mul)
        dist = MSELoss(target_features, synth_features).mean()

        # MSE between target image and generated image
        mse_loss = MSELoss(target_images, synth_images).mean() #* mse_lambda
        l1_loss = L1Loss(target_images, synth_images)

        # smooth loss
        #smooth_loss = smooth_loss_f(synth_images) * smooth_lambda

        #if projected_w is not None:
        #    # L1 Loss on w
        #    w_reg_loss = L1Loss(w_opt, saved_w.cuda())
#
        #    # Entropy Minimize
        #    sm_out = sm(torch.abs(w_opt - saved_w.cuda()))
        #    entropy_loss = -(sm_out * torch.log(sm_out)).sum()

        #####################################################################
        #LOSS
        #####################################################################

        #loss = mse_loss# + dist
        #loss = class_loss #+ 10 * entropy_loss #+ 10 * w_reg_loss#+ reg_loss * regularize_noise_weight + smooth_los
        #loss = class_loss #+ 10 * entropy_loss #+ 10 * w_reg_loss#+ reg_loss * regularize_noise_weight + smooth_los
        min_loss_lambda = .01
        class_loss_lambda = 1
        if img_to_img:
            loss = l1_loss + 0.01 * dist
        else:
            loss = class_loss * class_loss_lambda + min_loss * min_loss_lambda
        #####################################################################

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        min_hook.remove()

        # Normalized gradient
        if norm_grad:
            print('normalizng gradient...')
            grad_norm = torch.norm(learnable.grad, p=2)
            learnable.grad = learnable.grad / grad_norm

        synth_conf = round(nn.Softmax(dim=1)(out)[:, target_lbl].mean().item(), 4)*100

        optimizer.step()
        if img_to_img:
            print(f'step {step+1:>4d}/{num_steps}: class loss {float(class_loss):<4.2f} mse loss {float(mse_loss):<4.2f} loss {float(loss):<5.2f} dist {float(dist):<5.2f} avg confidence: {synth_conf}%')
        else:
            print(f'step {step+1:>4d}/{num_steps}: class loss {float(class_loss * class_loss_lambda):<4.2f} l1 loss {float(l1_loss):<4.2f} loss {float(loss):<5.2f} min loss {float(min_loss*min_loss_lambda):<5.2f} avg confidence: {synth_conf}%')
        #print(f"Max: {learnable.max().item()}, Min: {learnable.min().item()}")
        #logprint(f'step {step+1:>4d}/{num_steps}: real_loss {float(real_loss):<4.2f} loss {float(loss):<5.2f} class loss {float(class_loss):<5.2f} confidence: {synth_conf}%')

        # Save projected W for each optimization step.
        if learn_param == "w":
            new_param = (start_ws[:, 0, :].unsqueeze(1) + learnable.unsqueeze(0).repeat([len(start_ws), 1, 1])).cuda()
            w_opt = new_param.clone().repeat([1, G.mapping.num_ws, 1]).cuda()
            w_out[step] = w_opt.detach().cpu().clone()[:, 0, :]
        if learn_param == "z":
            if img_to_img:
                z_out[step] = learnable.detach().cpu().clone()
                w_opt = G.mapping(learnable, None)
                w_out[step] = w_opt.detach().cpu().clone()[:, 0, :]
            else:
                w_out[step] = w_opt.detach().cpu().clone()[:, 0, :]
                new_param = (start_zs + learnable.repeat([len(start_zs), 1]))
                z_out[step] = new_param.detach().cpu().clone()
                w_opt = G.mapping(new_param, None)
                w_out[step] = w_opt.detach().cpu().clone()[:, 0, :]

    return w_out, z_out, all_synth_images

def visualize(
    outdir,
    projected_ws,
    projected_zs,
    synth_imgs,
    target_imgs,
    F,
    C,
    G,
    target_lbl,
    buckets
):

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    diffs = []
    imgs = []
    confs = []
    slopes = []
    heatmaps = []
    max_v = -9999
    min_v = 9999
    conf_composite_add = None
    conf_composite_rm = None
    sm = nn.Softmax(dim=0)
        
    print("Calculating difference images")
    max_v = 0
    os.makedirs(os.path.join(outdir, "img_frames"), exist_ok=True)
    for w_i, [projected_w, synth_image] in enumerate(zip(projected_ws, synth_imgs)):
        synth_image = synth_image[0]
        transforms.ToPILImage()(synth_image).save(f'{os.path.join(outdir, "img_frames")}/{w_i}.png')
        if w_i == 0:
            PIL.Image.fromarray((synth_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)).save(f"{outdir}/base.png")
        #synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        diff_image = np.zeros_like(np.transpose(synth_image.cpu().numpy(), (1, 2, 0)))

        if len(imgs) > 0:
            hsv_diff = (get_lightness(np.transpose(synth_image.cpu().numpy(), (1, 2, 0))) - get_lightness(imgs[-1]))
            diff_image = hsv_diff#.sum(2)
            diff_image = np.transpose(np.stack((diff_image, diff_image, diff_image)), (1, 2, 0))
        diffs.append(diff_image)
        imgs.append(np.copy(np.transpose(synth_image.cpu().numpy(), (1, 2, 0))))
        input_img = NORMALIZE(synth_image)
        out = C(F(input_img.unsqueeze(0).cuda()))[0]
        conf = round(sm(out)[target_lbl].item(), 4) * 100
        confs.append(conf)
        if w_i == 0:
            conf_composite_add = np.zeros_like(diff_image[:, :, 0].astype(np.float64))
            conf_composite_rm = np.zeros_like(diff_image[:, :, 0].astype(np.float64))
            slopes.append(0.0)
            heatmaps.append(np.zeros_like(projected_w.cpu().numpy()[0]))
        else:
            heatmaps.append(projected_w.cpu().numpy()[0] - projected_ws[0].cpu().numpy()[0])
            slopes.append(conf - confs[-2])
            if w_i % buckets == 0:
                tmp_add = np.copy(diff_image)
                tmp_rm = np.copy(diff_image)
                tmp_rm[:, :, 0][tmp_rm[:, :, 0] > 0] = 0
                tmp_add[:, :, 0][tmp_add[:, :, 0] < 0] = 0
                tmp_rm[:, :, 0] *= -1
                tmp_add[:, :, 0] -= min(tmp_add[:, :, 0].min(), tmp_rm[:, :, 0].min())
                tmp_rm[:, :, 0] -= min(tmp_add[:, :, 0].min(), tmp_rm[:, :, 0].min())
                tmp_add[:, :, 0] /= max(tmp_add[:, :, 0].max(), tmp_rm[:, :, 0].max())
                tmp_rm[:, :, 0] /= max(tmp_add[:, :, 0].max(), tmp_rm[:, :, 0].max())
                conf_composite_add += slopes[-1] * cv2.medianBlur(tmp_add[:, :, 0], 3).astype(np.float64)
                conf_composite_rm += slopes[-1] * cv2.medianBlur(tmp_rm[:, :, 0], 3).astype(np.float64)
    #print(round(sm(out)[target_lbl].item(), 4) * 100)
    #out = classifier(backbone(image_transform()(PIL.Image.fromarray(img_batch[0], 'RGB')).unsqueeze(0).cuda()))[0]
    #print(round(sm(out)[target_lbl].item(), 4) * 100)
    #print("="*30)
    
    conf_composite_add -= conf_composite_add.min()
    conf_composite_rm -= conf_composite_rm.min()

    conf_composite_add /= conf_composite_add.max()
    conf_composite_rm /= conf_composite_add.max()

    conf_composite_add = (conf_composite_add * 255).astype(np.uint8)
    conf_composite_rm = (conf_composite_rm * 255).astype(np.uint8)
    PIL.Image.fromarray(conf_composite_add).save(f"{outdir}/conf_composite_add.png")
    PIL.Image.fromarray(conf_composite_rm).save(f"{outdir}/conf_composite_rm.png")
    print (f'Saving optimization progress video "{outdir}/proj.mp4"')
    video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=30, codec='libx264')
    px = 1/plt.rcParams['figure.dpi']
    key_images = []
    threshold = 1
    below_threshold = True
    os.makedirs(os.path.join(outdir, "img_frames"), exist_ok=True)
    for i in range(len(imgs)):
        synth_image = imgs[i]
        if below_threshold and slopes[i] > threshold:
            key_images.append(synth_image)
            below_threshold = False
        elif not below_threshold and slopes[i] < threshold:
            key_images.append(synth_image)
            below_threshold = True
        elif i == len(imgs) - 1 or i == 0:
            key_images.append(synth_image)

        width = synth_imgs[0][0].shape[1]
        diff_image = np.abs(diffs[i])
        diff_image = cv2.medianBlur(diff_image, 5)
        diff_image -= diff_image.min()
        diff_image /= diff_image.max()
        diff_image = (diff_image * 255).clip(0, 255).astype(np.uint8)
        fig = plt.figure(figsize=(width*2*px, width*px))
        plt.plot(confs)
        plt.axvline(x=i, color="yellow")
        plt.gca().get_xaxis().set_visible(False)
        fig.canvas.draw()
        conf_graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        conf_graph = conf_graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        fig = plt.figure(figsize=(width*px, width*px))
        plt.plot(slopes)
        plt.axvline(x=i, color="yellow")
        plt.gca().get_xaxis().set_visible(False)
        fig.canvas.draw()
        slope_graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        slope_graph = slope_graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        fig = plt.figure(figsize=(width*px, width*px))
        heatmap = heatmaps[i].reshape((32, 16))
        vmin = np.array(heatmaps).min()
        vmax = np.array(heatmaps).max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        hm = plt.imshow(heatmap, norm=norm, cmap="bwr")
        plt.colorbar(hm, location='left', shrink=0.5)
        fig.canvas.draw()
        heatmap_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_fig = heatmap_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        video.append_data(np.concatenate([np.concatenate([(target_imgs[0]* 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0), (synth_image* 255).astype(np.uint8), diff_image], axis=1), np.concatenate([conf_graph, heatmap_fig], axis=1)], axis=0))
    video.close()
    PIL.Image.fromarray(heatmap_fig).save(f"{outdir}/w_heatmap.png")

    #for i, key_img in enumerate(key_images):
    #    if i == 0: continue
    #    #diff_img = PIL.ImageChops.difference(PIL.Image.fromarray(key_img), PIL.Image.fromarray(key_images[-1]))
    #    diff_image = (to_hsv(key_img) - to_hsv(key_images[i-1])).sum(2)
    #    tmp_add = np.copy(diff_image)
    #    tmp_add[tmp_add < 0] = 0
    #    tmp_rm = np.copy(diff_image)
    #    tmp_rm[tmp_rm > 0] = 0
    #    tmp_rm *= -1
    #    tmp_add -= tmp_add.min()
    #    tmp_add /= tmp_add.max()
    #    tmp_add = (tmp_add * 255).astype(np.uint8)
    #    tmp_rm -= tmp_rm.min()
    #    tmp_rm /= tmp_rm.max()
    #    tmp_rm = (tmp_rm * 255).astype(np.uint8)
    #    tmp_add = np.transpose(np.stack((tmp_add, tmp_add, tmp_add)), (1, 2, 0))
    #    tmp_rm = np.transpose(np.stack((tmp_rm, tmp_rm, tmp_rm)), (1, 2, 0))
    #    out_img = np.concatenate([(key_images[i-1] * 255).astype(np.uint8), (key_img*255).astype(np.uint8), tmp_add], axis=1)
    #    PIL.Image.fromarray(out_img).save(f"{outdir}/Key_Diff_{i}_add.png")
    #    out_img = np.concatenate([(key_images[i-1] * 255).astype(np.uint8), (key_img*255).astype(np.uint8), tmp_rm], axis=1)
    #    PIL.Image.fromarray(out_img).save(f"{outdir}/Key_Diff_{i}_rm.png")
#

    # Save final projected frame and W vector.
    #target_pil.save(f'{outdir}/target.png')
    os.makedirs(os.path.join(outdir, "projections"), exist_ok=True)
    for i in range(len(target_imgs)):
        tgt_img = (target_imgs[i].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        PIL.Image.fromarray(tgt_img, 'RGB').save(f'{outdir}/projections/tgt_{i}.png')
        projected_w = projected_ws[-1][i].unsqueeze(0).unsqueeze(0)
        projected_w = projected_w.repeat([1, G.mapping.num_ws, 1])
        synth_image = G.synthesis(projected_w.cuda(), noise_mode='const')
        conf = round(sm(C(F(NORMALIZE(((synth_image+1)*(1/2)).clamp(0, 1))))[0])[target_lbl].item(), 4)
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/projections/proj_{i}_{conf}.png')
        projected_w = projected_ws[0][i].unsqueeze(0).unsqueeze(0)
        projected_w = projected_w.repeat([1, G.mapping.num_ws, 1])
        synth_image = G.synthesis(projected_w.cuda(), noise_mode='const')
        conf = round(sm(C(F(NORMALIZE(((synth_image+1)*(1/2)).clamp(0, 1))))[0])[target_lbl].item(), 4)
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/projections/proj_{i}_start.png')

    if projected_zs is None:
        np.savez(f'{outdir}/latents.npz', w=projected_ws[-1].cpu().numpy())
    else:
        np.savez(f'{outdir}/latents.npz', w=projected_ws[-1].cpu().numpy(), z=projected_zs[-1].cpu().numpy())
        
    
################################################
# Main
################################################
def main():
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)

    # Load Models
    G, D, F, C = load_models(args.network, f_path=args.backbone, c_path=args.classifier)

    # Load Data
    images, start_ws, start_zs = load_data(latent_path=args.latents, img_path=args.target, resolution=G.img_resolution)

    # Run Projection
    projected_ws, projected_zs, synth_imgs = project_tmp(
        images,
        G,
        D,
        F,
        C,
        target_lbl                 = args.target_lbl,
        learn_param                = args.learn_param,
        projected_zs               = start_zs,
        projected_ws               = start_ws,
        num_steps                  = args.num_steps,
        init_lr                    = args.lr,
        avg_grad                   = args.avg_grad,
        min_grad                   = args.min_grad,
        img_to_img                 = args.img_to_img
    )

    # Visualize
    visualize(
        args.outdir,
        projected_ws,
        projected_zs,
        synth_imgs,
        images,
        F,
        C,
        G,
        args.target_lbl,
        args.buckets,
    )



if __name__ == "__main__":
    main()

################################################
# Outline
################################################
"""
    images, start_ws, start_zs = load_data(...)
    G, D, F, C = load_models(...)

    p_images, p_ws, p_zs, p_confs = run_projection(...)

    save_visualization(...)

"""

################################################
# Description
################################################
"""
    Use cases:
    1) Given a batch of images, learn a z and w such that they produce those images
    2) Given a batch of images and a starting z or w, learn a new z and w that produce those images starting from z & w
    3) Given a batch of images and a class label and a starting z or w, learn a new z and w 
        that produces high confidence with minimal changes to the starting z or w (and keeps visual image change to a minimum)

    Main parts:
        A) Load images & existing z and/or w
        B) Perform a projection based on either 1, 2, or 3
        C) Visualize the projection process

"""
################################################

