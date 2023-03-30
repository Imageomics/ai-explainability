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
import matplotlib.colors as mcolors

import styleGAN.dnnlib as dnnlib
import styleGAN.legacy as legacy
import sys

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

def project(
    G,
    D,
    backbone,
    classifier,
    target_lbl,
    target, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
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
    projected_w: Optional[str],
    projected_z: Optional[str]
):
    assert target[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).cuda() # type: ignore
    if D is not None:
        D = copy.deepcopy(D).eval().requires_grad_(False).cuda() # type: ignore

    # Compute w stats.
    #logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    #z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    #w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    #w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    #w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    #w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    saved_z = np.random.RandomState(123).randn(len(target), G.z_dim)
    #if projected_z is not None:
    #    saved_z = np.load(projected_z)['z']
    z_opt = torch.tensor(saved_z).cuda().requires_grad_()
    w_avg = G.mapping(z_opt, None)[:, :1, :]  # [N, L, C]
    print(w_avg.shape)

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().cuda()

    # Features for target image.
    target_images = target.to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    #w_opt = torch.tensor(w_avg, dtype=torch.float32).cuda().requires_grad_() # pylint: disable=not-callable
    w_opt = torch.tensor(w_avg, dtype=torch.float32).cuda().requires_grad_() # pylint: disable=not-callable
    if projected_w is not None:
        saved_w = torch.tensor(np.load(projected_w)['w'][0][0]) # 512 dimension
        saved_w = saved_w[None, None, :] # [1,1,512]
        w_opt = torch.tensor(saved_w).cuda().requires_grad_()

    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32).cuda()
    z_out = torch.zeros([num_steps] + list(z_opt.shape), dtype=torch.float32).cuda()
    optimizer = torch.optim.Adam([z_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    #optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    #optimizer = torch.optim.SGD([w_opt] + list(noise_bufs.values()), lr=initial_learning_rate)

    # MSE between target image and generated image
    MSELoss = nn.MSELoss().cuda()
    CELoss = nn.CrossEntropyLoss().cuda()
    L1Loss = nn.L1Loss().cuda()
    sm = nn.Softmax(dim=2).cuda()
    

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

    all_synth_images = []

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        #w_noise = torch.randn_like(w_opt) * w_noise_scale
        #ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        ws = (w_opt).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')
        def check(grad):
            return grad
            grad_img = torch.abs(grad[0]).sum(0)
            grad_img -= grad_img.min()
            grad_img /= grad_img.max()
            thresh = 0.2
            grad_img[grad_img < thresh] = 0
            #print(grad_img)
            transforms.ToPILImage()(grad_img).save("grad_img.png")
            #exit()
            grad[0][:, grad_img < thresh] = 0
            return grad

        hook = synth_images.register_hook(check)
        real_loss = 0
        if D is not None:
            logits = D(synth_images, None).cuda()
            real_loss = torch.nn.functional.softplus(-logits)
        input_img = (synth_images + 1) * (1/2)
        input_img = input_img.clamp(0, 1)
        #transforms.ToPILImage()(input_img[0]).save("test_img.png")
        #transforms.ToPILImage()(UNNORMALIZE(target_images[0])).save("test_img2.png")
        all_synth_images.append(synth_images.detach())
        #input_img = input_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        #input_img = PIL.Image.fromarray(input_img, 'RGB')
        #input_img.save("test.png")
        out = classifier(backbone(NORMALIZE(input_img)))
        lbls = torch.tensor([target_lbl] * len(target)).cuda()
        class_loss = CELoss(out, lbls)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = MSELoss(target_features, synth_features)

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
        l1_loss = L1Loss(target_images, synth_images)

        # smooth loss
        smooth_loss = smooth_loss_f(synth_images) * smooth_lambda

        #if projected_w is not None:
        #    # L1 Loss on w
        #    w_reg_loss = L1Loss(w_opt, saved_w.cuda())
#
        #    # Entropy Minimize
        #    sm_out = sm(torch.abs(w_opt - saved_w.cuda()))
        #    entropy_loss = -(sm_out * torch.log(sm_out)).sum()

        loss = l1_loss# + dist
        #loss = class_loss #+ 10 * entropy_loss #+ 10 * w_reg_loss#+ reg_loss * regularize_noise_weight + smooth_los
        #loss = class_loss #+ 10 * entropy_loss #+ 10 * w_reg_loss#+ reg_loss * regularize_noise_weight + smooth_los
       # loss = class_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        hook.remove()

        # Normalized gradient
        if norm_grad:
            print('normalizng gradient...')
            grad_norm = torch.norm(w_opt.grad, p=2)
            w_opt.grad = w_opt.grad / grad_norm

        synth_conf = round(nn.Softmax(dim=1)(out)[0][target_lbl].item(), 4)*100

        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {float(dist):<4.2f} l1 loss {float(l1_loss):<4.2f} loss {float(loss):<5.2f} confidence: {synth_conf}%')
        #logprint(f'step {step+1:>4d}/{num_steps}: real_loss {float(real_loss):<4.2f} loss {float(loss):<5.2f} class loss {float(class_loss):<5.2f} confidence: {synth_conf}%')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().clone()[0]
        z_out[step] = z_opt.detach().clone()

        w_opt = G.mapping(z_opt, None)[:, :1, :]

        # Normalize noise.
        #with torch.no_grad():
        #    for buf in noise_bufs.values():
        #        buf -= buf.mean()
        #        buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1]), z_out, all_synth_images


def to_hsv(x):
    return mcolors.rgb_to_hsv(x)
#----------------------------------------------------------------------------
def load_imgs(img_dir, view="D", max_size=8, resolution=128):

    batch = None
    i = 0
    for root, dirs, paths in os.walk(img_dir):
        for path in paths:
            view = path.split(".")[0].split("_")[1]
            if view != view: continue
            i += 1
            full_path = os.path.join(root, path)

            target_pil = PIL.Image.open(full_path).convert('RGB')
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.resize((resolution, resolution), PIL.Image.LANCZOS)
            target_uint8 = np.array(target_pil, dtype=np.uint8).transpose([2, 0, 1])
            if batch is None:
                batch = np.expand_dims(target_uint8, axis=0)
            else:
                batch = np.concatenate((batch, np.expand_dims(target_uint8, axis=0)), axis=0)

            if i >= max_size: break
            
    batch = torch.tensor(batch).cuda()
    return batch
#----------------------------------------------------------------------------

def run_projection(
    network_pkl: str,
    target_dir: str,
    outdir: str,
    save_video: bool,
    num_steps: int,
    smooth_beta: float,
    smooth_eps: float,
    smooth_lambda: float,
    mse_lambda: float,
    norm_grad: bool,
    projected_w: Optional[str],
    projected_z: Optional[str],
    backbone: Optional[str],
    classifier: Optional[str],
    target_lbl: int,
    lr: float
):

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    D = None
    with dnnlib.util.open_url(network_pkl) as fp:
        net = legacy.load_network_pkl(fp)
        G = net['G_ema'].requires_grad_(False).cuda() # type: ignore
        G.eval()
        #D = net['D'].requires_grad_(False).cuda() # type: ignore
        #D.eval()

    # Load target images.
    img_batch = load_imgs(target_dir, resolution=G.img_resolution)

    weights = torch.load(backbone)
    class_weights = torch.load(classifier)

    backbone = VGG16(pretrain=False).cuda()
    backbone.load_state_dict(weights)
    classifier = Classifier(backbone.in_features, 34).cuda()
    classifier.load_state_dict(class_weights)
    classifier.eval()
    backbone.eval()

    #target_out = classifier(backbone(image_transform()(target_pil).unsqueeze(0).cuda()))
    #target_conf = nn.Softmax(dim=1)(target_out)[0][target_lbl]
    #print(f"Target Image confidence: {target_conf}")

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, z_steps, all_synth_images = project(
        G,
        D,
        backbone,
        classifier,
        target_lbl,
        target=img_batch, # pylint: disable=not-callable
        num_steps=num_steps,
        verbose=True,
        smooth_beta=smooth_beta,
        smooth_eps=smooth_eps,
        smooth_lambda=smooth_lambda,
        mse_lambda=mse_lambda,
        norm_grad=norm_grad,
        projected_w=projected_w,
        projected_z=projected_z,
        initial_learning_rate=lr
    )
    print(projected_w_steps.shape)
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

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
    conf_composite_remove = None
    sm = nn.Softmax(dim=0)
    if save_video:
        
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        max_v = 0
        for w_i, [projected_w, synth_image] in enumerate(zip(projected_w_steps, all_synth_images)):
            synth_image = synth_image
            synth_image = (synth_image + 1) * (1/2)
            synth_image = synth_image.clamp(0, 1)
            if w_i == 0:
                PIL.Image.fromarray((synth_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)).save(f"{outdir}/base.png")
            #synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            diff_image = np.zeros_like(np.transpose(synth_image[0].cpu().numpy(), (1, 2, 0)))

            if len(imgs) > 0:
                hsv_diff = (to_hsv(np.transpose(synth_image[0].cpu().numpy(), (1, 2, 0))) - to_hsv(imgs[-1]))
                diff_image = hsv_diff.sum(2)
                diff_image = np.transpose(np.stack((diff_image, diff_image, diff_image)), (1, 2, 0))
            diffs.append(diff_image)
            imgs.append(np.copy(np.transpose(synth_image.clamp(0, 1)[0].cpu().numpy(), (1, 2, 0))))
            input_img = NORMALIZE(synth_image.clamp(0, 1))
            out = classifier(backbone(input_img))[0]
            conf = round(sm(out)[target_lbl].item(), 4) * 100
            confs.append(conf)
            if w_i == 0:
                conf_composite_add = np.zeros_like(diff_image[:, :, 0].astype(np.float64))
                conf_composite_rm = np.zeros_like(diff_image[:, :, 0].astype(np.float64))
                slopes.append(0.0)
                heatmaps.append(np.zeros_like(projected_w.cpu().numpy()[0]))
            else:
                heatmaps.append(projected_w.cpu().numpy()[0] - projected_w_steps[0].cpu().numpy()[0])
                slopes.append(conf - confs[-2])
                tmp_add = np.copy(diff_image)
                tmp_rm = np.copy(diff_image)
                tmp_rm[:, :, 0][tmp_rm[:, :, 0] > 0] = 0
                tmp_add[:, :, 0][tmp_add[:, :, 0] < 0] = 0
                tmp_rm[:, :, 0] *= -1
                conf_composite_add += slopes[-1] * tmp_add[:, :, 0].astype(np.float64)
                conf_composite_rm += slopes[-1] * tmp_rm[:, :, 0].astype(np.float64)
            #print(round(sm(out)[target_lbl].item(), 4) * 100)
            #out = classifier(backbone(image_transform()(PIL.Image.fromarray(img_batch[0], 'RGB')).unsqueeze(0).cuda()))[0]
            #print(round(sm(out)[target_lbl].item(), 4) * 100)
            #print("="*30)
        
        conf_composite_add -= conf_composite_add.min()
        conf_composite_add /= conf_composite_add.max()
        conf_composite_add = (conf_composite_add * 255).astype(np.uint8)
        conf_composite_rm -= conf_composite_rm.min()
        conf_composite_rm /= conf_composite_rm.max()
        conf_composite_rm = (conf_composite_rm * 255).astype(np.uint8)
        PIL.Image.fromarray(conf_composite_add).save(f"{outdir}/conf_composite_add.png")
        PIL.Image.fromarray(conf_composite_rm).save(f"{outdir}/conf_composite_rm.png")
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        px = 1/plt.rcParams['figure.dpi']
        key_images = []
        threshold = 1
        below_threshold = True
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

            width = img_batch[0].shape[1]
            diff_image = (np.abs(diffs[i]) * 255).clip(0, 255).astype(np.uint8)
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
            video.append_data(np.concatenate([np.concatenate([img_batch[0].cpu().numpy().transpose(1, 2, 0), (synth_image* 255).astype(np.uint8), diff_image], axis=1), np.concatenate([conf_graph, heatmap_fig], axis=1)], axis=0))
        video.close()
        PIL.Image.fromarray(heatmap_fig).save(f"{outdir}/w_heatmap.png")

    for i, key_img in enumerate(key_images):
        if i == 0: continue
        #diff_img = PIL.ImageChops.difference(PIL.Image.fromarray(key_img), PIL.Image.fromarray(key_images[-1]))
        diff_image = (to_hsv(key_img) - to_hsv(key_images[i-1])).sum(2)
        tmp_add = np.copy(diff_image)
        tmp_add[tmp_add < 0] = 0
        tmp_rm = np.copy(diff_image)
        tmp_rm[tmp_rm > 0] = 0
        tmp_rm *= -1
        tmp_add -= tmp_add.min()
        tmp_add /= tmp_add.max()
        tmp_add = (tmp_add * 255).astype(np.uint8)
        tmp_rm -= tmp_rm.min()
        tmp_rm /= tmp_rm.max()
        tmp_rm = (tmp_rm * 255).astype(np.uint8)
        tmp_add = np.transpose(np.stack((tmp_add, tmp_add, tmp_add)), (1, 2, 0))
        tmp_rm = np.transpose(np.stack((tmp_rm, tmp_rm, tmp_rm)), (1, 2, 0))
        out_img = np.concatenate([(key_images[i-1] * 255).astype(np.uint8), (key_img*255).astype(np.uint8), tmp_add], axis=1)
        PIL.Image.fromarray(out_img).save(f"{outdir}/Key_Diff_{i}_add.png")
        out_img = np.concatenate([(key_images[i-1] * 255).astype(np.uint8), (key_img*255).astype(np.uint8), tmp_rm], axis=1)
        PIL.Image.fromarray(out_img).save(f"{outdir}/Key_Diff_{i}_rm.png")


    # Save final projected frame and W vector.
    #target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
    np.savez(f'{outdir}/projected_z.npz', z=z_steps[-1].cpu().numpy())

# Get args
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--network', type=str, help='Network pickle filename')
    parser.add_argument('--target', type=str, help='Target directory of image files to project to')
    parser.add_argument('--num-steps',              help='Number of optimization steps', type=int, default=1000)
    parser.add_argument('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True)
    parser.add_argument('--outdir',                 help='Where to save the output images', type=str)
    parser.add_argument('--smooth_beta',            help='Smooth beta', type=float, default=2)
    parser.add_argument('--smooth_eps',             help='Smooth eps', type=float, default=1e-3)
    parser.add_argument('--smooth_lambda',          help='Smooth lambda', type=float, default=0.0)
    parser.add_argument('--mse_lambda',             help='MSE lambda', type=float, default=0.0)
    parser.add_argument('--lr',             help='learning rate', type=float, default=0.001)
    parser.add_argument('--norm_grad',              help='Normalizing gradient', type=bool, default=False)
    parser.add_argument('--projected-w', help='Projection result file', type=str)
    parser.add_argument('--projected-z', help='Projection result file', type=str)
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
        args.projected_z,
        args.backbone,
        args.classifier,
        args.target_lbl,
        args.lr
    )
