import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import stylegan3_org.dnnlib as dnnlib
from superpixel import superpixel
from PIL import Image

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

def calc_w(G, start_zs, start_ws, learnable, learn_param="w", batch=False, multi_w=False):
    # Expand?
    expanded = learnable
    if not batch:
        if learn_param == "w":
            expanded = learnable.repeat([len(start_ws), 1, 1])
        elif learn_param == "z":
            expanded = learnable.repeat([len(start_zs), 1])
        else:
            assert False, "Invalide learn_param"

    if learn_param == "w":
        if multi_w:
            new_param = (start_ws + expanded).cuda()
        else:
            new_param = (start_ws.unsqueeze(1) + expanded).cuda()
            new_param = new_param.clone().repeat([1, G.mapping.num_ws, 1])
        return new_param
    
    if learn_param == "z":
        return G.mapping(start_zs + learnable, None)

    assert False, "Invalide learn_param"

def load_learnable(start_zs, start_ws, learn_param="w", batch=False, multi_w=False):
    if learn_param == "w":
        tmp = start_ws.unsqueeze(1).clone().detach().cuda()
        if multi_w:
            tmp = start_ws.clone().detach().cuda()
        if batch:
            learnable = torch.zeros_like(tmp).cuda().requires_grad_()
        else:
            learnable = torch.zeros_like(tmp[0]).cuda().requires_grad_()
    elif learn_param == "z":
        tmp = start_zs.clone().detach().cuda()
        if batch:
            learnable = torch.zeros_like(tmp).cuda().requires_grad_()
        else:
            learnable = torch.zeros_like(tmp[0]).cuda().requires_grad_()
    else:
        assert False, "Invalide learn_param"
    
    return learnable

def project(
    images,
    G,
    D,
    F,
    C,
    projection_lbl,
    learn_param                = "w",
    start_zs               = None,
    start_ws               = None,
    num_steps                  = 200,
    init_lr                    = 0.001,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    img_to_img                 = False,
    batch                      = False,
    use_entropy                = False,
    verbose                    = False,
    use_default_feat_extractor = False,
    no_regularizer             = False,
    smooth_change              = False,
    smooth_beta                = 2,
    smooth_eps                 = 1e-3,
    use_superpixel             = False,
    multi_w                    = False


):
    def logprint(x):
        if verbose:
            print(x)

    if images is not None:
        assert images[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    # =====================================================================
    # Preparing models
    G = copy.deepcopy(G).eval().requires_grad_(False).cuda() # type: ignore
    if D is not None:
        D = copy.deepcopy(D).eval().requires_grad_(False).cuda() # type: ignore

    # Load VGG16 feature detector.
    feat_extractor = None
    if F is None or use_default_feat_extractor:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().cuda()
        feat_extractor = lambda x: vgg16(x.clone() * 255, resize_images=False, return_lpips=True)
    else:
        feat_extractor = lambda x: F(NORMALIZE(x))

    # =====================================================================

    # Features for target image. pretrained expects image to be unnormalized, but our feature extractor is already normalized.
    if images is not None:
        target_images = images.to(torch.float32)
        target_features = feat_extractor(target_images)

    learnable = load_learnable(start_zs, start_ws, learn_param=learn_param, batch=(batch or img_to_img), multi_w=multi_w)
    optimizer = torch.optim.Adam([learnable], betas=(0.9, 0.999), lr=init_lr)

    w_opt = calc_w(G, start_zs, start_ws, learnable, learn_param, batch=(batch or img_to_img), multi_w=multi_w)

    if multi_w:
        w_out = torch.zeros([num_steps] + [len(w_opt), start_ws.shape[1], start_ws.shape[2]], dtype=torch.float32)
    else:
        w_out = torch.zeros([num_steps] + [len(w_opt), start_ws.shape[1]], dtype=torch.float32)
    z_out = None
    if learn_param == "z":
        z_out = torch.zeros([num_steps] + [len(w_opt), start_zs.shape[1]], dtype=torch.float32)
    all_synth_images = []
    pixel_losses = []
    perceptual_losses = []
    min_losses = []
    image_confs = []

    # Load loss functions
    MSELoss = nn.MSELoss().cuda()
    CELoss = nn.CrossEntropyLoss().cuda()
    L1Loss = nn.L1Loss().cuda()
    sm = nn.Softmax(dim=1).cuda()


    # NOTE the last update is not saved
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = init_lr * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #####################################################################
        # Generate images
        #####################################################################
        synth_images = G.synthesis(w_opt, noise_mode='const')
        synth_images = (synth_images + 1) * (1/2)
        synth_images = synth_images.clamp(0, 1)
        Image.fromarray(np.transpose((synth_images[0] * 255).detach().cpu().numpy().astype(np.uint8), [1, 2, 0])).save("test.png")
        if step == 0:
            start_images = synth_images.detach().clone()
            if use_superpixel:
                superpixel_labels = np.zeros([start_images.shape[0]] + list(start_images.shape[2:]))
                #n_sp_labels = 0
                for s_i, s_img in enumerate(start_images):
                    superpixel_labels[s_i] = superpixel(np.transpose(s_img.cpu().numpy(), [1, 2, 0]))
                    #n_sp_labels = max(n_sp_labels, max(superpixel_labels[s_i])+1)
                
        #####################################################################

        #####################################################################
        # Gradient Hook
        #####################################################################
        def img_grad_hook(gradient):
            if use_superpixel:
                def save_img(gradient, f_name):
                    img = gradient.abs().mean(0).detach().cpu().numpy()
                    img -= img.min()
                    img /= img.max()
                    img *= 255
                    img = img.astype(np.uint8)
                    Image.fromarray(img).save(f_name)
                #save_img(gradient[0], "gradient.png")
                for i, img_grad in enumerate(gradient):
                    mask_vals = []
                    for lbl in range(int(superpixel_labels[i].max())+1):
                        mask = superpixel_labels[i].astype(np.uint8) == lbl
                        v = np.abs(img_grad.detach().cpu().numpy()).sum(0)[mask].mean()
                        mask_vals.append([mask, v])
                        
                    mask_vals = sorted(mask_vals, key=lambda x: x[1], reverse=True)
                    for j, (mask, v) in enumerate(mask_vals):
                        if j < 3: continue
                        mask = mask_vals[j][0]
                        gradient[i][:, mask] = 0.0
                save_img(gradient[0], "gradient_after.png")
            return gradient
                    
        grad_hook = synth_images.register_hook(img_grad_hook)
        #####################################################################

        all_synth_images.append(synth_images.detach().cpu().numpy())
        # Save projected W for each optimization step.
        if multi_w:
            w_out[step] = w_opt.detach().cpu().clone()
        else:
            w_out[step] = w_opt.detach().cpu().clone()[:, 0, :]
        if learn_param == "z":
            if img_to_img or batch:
                z_out[step] = learnable.detach().cpu().clone()
            else:
                new_param = (start_zs + learnable.repeat([len(start_zs), 1]))
                z_out[step] = new_param.detach().cpu().clone()

        synth_features = feat_extractor(synth_images)
        
        # This should only be used with a pretrained feature extractor for the classifier on butterflies
        synth_conf = 0.0
        if C is not None and F is not None:
            out = C(F(NORMALIZE(synth_images)))
            conf = sm(out)
            image_confs.append(conf.detach().cpu().numpy().tolist())
            #print(conf[0, projection_lbl].item())
            synth_conf = round(conf[:, projection_lbl].mean().item(), 4)*100
            lbls = torch.tensor([projection_lbl] * len(w_opt)).cuda()
            class_loss = CELoss(out, lbls)

        #####################################################################
        # LOSS
        #####################################################################
        min_loss_lambda = .01
        class_loss_lambda = 1
        dist_loss_lambda = 0.01
        smooth_loss_lambda = 0.001
        if img_to_img:
            dist = MSELoss(target_features, synth_features)
            perceptual_losses.append(dist.item())
            l1_loss = L1Loss(target_images, synth_images)
            pixel_losses.append(l1_loss.item())
            loss = l1_loss + dist_loss_lambda * dist
            logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f} perceptual loss {float(dist):<5.2f} pixel loss {float(l1_loss):<5.2f} avg confidence: {synth_conf}%')
        else:
            if no_regularizer:
                min_loss = 0.0
                min_losses.append(min_loss)
            else:
                smooth_loss = 0.0
                smooth_str = ""
                if smooth_change:
                    img_diff = synth_images - start_images
                    x_diff = img_diff[:, :, :-1, :-1] - img_diff[:, :, :-1, 1:]
                    y_diff = img_diff[:, :, :-1, :-1] - img_diff[:, :, 1:, :-1]
                    sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, smooth_eps, 10000000)
                    smooth_loss = torch.norm(sq_diff, smooth_beta / 2.0) ** (smooth_beta / 2.0)
                    smooth_str = f'smooth loss: {float(smooth_loss * smooth_loss_lambda):<4.2f} '
                min_loss = L1Loss(learnable, torch.zeros_like(learnable))
                if use_entropy:
                    dim = 1 if (batch or img_to_img) else 0
                    ent_sm = nn.Softmax(dim=dim).cuda()
                    sm_out = ent_sm(torch.abs(learnable))
                    min_loss = -(sm_out * torch.log(sm_out)).sum()
                min_losses.append(min_loss.item())
            loss = class_loss * class_loss_lambda + min_loss * min_loss_lambda + smooth_loss * smooth_loss_lambda
            logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f} {smooth_str}class loss {float(class_loss * class_loss_lambda):<4.2f} min loss {float(min_loss*min_loss_lambda):<5.2f} avg confidence: {synth_conf}%')
        #####################################################################

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Update w_opt
        w_opt = calc_w(G, start_zs, start_ws, learnable, learn_param, batch=(batch or img_to_img), multi_w=multi_w)

        # Remove hook
        grad_hook.remove()

    return w_out, z_out, all_synth_images, pixel_losses, perceptual_losses, image_confs, min_losses
