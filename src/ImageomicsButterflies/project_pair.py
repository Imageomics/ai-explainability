import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import styleGAN.dnnlib as dnnlib

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

def calc_w(G, start_zs, start_ws, learnable, learn_param="w", batch=False):
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
        new_param = (start_ws.unsqueeze(1) + expanded).cuda()
        return new_param.clone().repeat([1, G.mapping.num_ws, 1])
    
    if learn_param == "z":
        return G.mapping(start_zs + learnable, None)

    assert False, "Invalide learn_param"

def project(
    G,
    D,
    F,
    C,
    projection_lbl,
    projection_lbl2,
    learn_param                = "w",
    start_zs               = None,
    start_zs2               = None,
    start_ws               = None,
    start_ws2               = None,
    num_steps                  = 200,
    init_lr                    = 0.001,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    use_entropy                = False,
    verbose                    = False,
    use_default_feat_extractor = False,
    smooth_change              = False,
    smooth_beta                = 2,
    smooth_eps                 = 1e-3


):
    def logprint(x):
        if verbose:
            print(x)

    # =====================================================================
    # Preparing models
    G = copy.deepcopy(G).eval().requires_grad_(False).cuda() # type: ignore
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

    learnable = torch.zeros_like(start_ws[0]).cuda().requires_grad_() # TODO: adapt this later
    optimizer = torch.optim.Adam([learnable], betas=(0.9, 0.999), lr=init_lr)

    w_opt = calc_w(G, start_zs, start_ws, learnable, learn_param, batch=False)
    w_opt2 = calc_w(G, start_zs2, start_ws2, -learnable, learn_param, batch=False)

    w_out = torch.zeros([num_steps] + [len(w_opt), start_ws.shape[1]], dtype=torch.float32)
    w_out2 = torch.zeros([num_steps] + [len(w_opt2), start_ws.shape[1]], dtype=torch.float32)
    z_out = None
    #if learn_param == "z":
    #    z_out = torch.zeros([num_steps] + [len(w_opt), start_zs.shape[1]], dtype=torch.float32)
    all_synth_images = []
    all_synth_images2 = []
    pixel_losses = []
    perceptual_losses = []
    min_losses = []
    image_confs = []
    image_confs2 = []

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
        synth_images2 = G.synthesis(w_opt2, noise_mode='const')
        synth_images2 = (synth_images2 + 1) * (1/2)
        synth_images2 = synth_images2.clamp(0, 1)
        if step == 0:
            start_images = synth_images.detach().clone()
            start_images2 = synth_images2.detach().clone()
        #####################################################################

        all_synth_images.append(synth_images.detach().cpu().numpy())
        all_synth_images2.append(synth_images2.detach().cpu().numpy())
        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().cpu().clone()[:, 0, :]
        w_out2[step] = w_opt2.detach().cpu().clone()[:, 0, :]
        if learn_param == "z":
            new_param = (start_zs + learnable.repeat([len(start_zs), 1]))
            z_out[step] = new_param.detach().cpu().clone()
        
        # This should only be used with a pretrained feature extractor for the classifier on butterflies
        if C is not None and F is not None:
            out = C(F(NORMALIZE(synth_images)))
            conf = sm(out)
            image_confs.append(conf.detach().cpu().numpy().tolist())
            synth_conf = round(conf[:, projection_lbl].mean().item(), 4)*100
            lbls = torch.tensor([projection_lbl] * len(w_opt)).cuda()
            class_loss = CELoss(out, lbls)
            out2 = C(F(NORMALIZE(synth_images2)))
            conf2 = sm(out2)
            image_confs2.append(conf2.detach().cpu().numpy().tolist())
            synth_conf2 = round(conf2[:, projection_lbl2].mean().item(), 4)*100
            lbls2 = torch.tensor([projection_lbl2] * len(w_opt2)).cuda()
            class_loss2 = CELoss(out2, lbls2)

        #####################################################################
        # LOSS
        #####################################################################
        min_loss_lambda = .01
        class_loss_lambda = 1
        smooth_loss_lambda = 0.1
        smooth_loss = 0.0
        smooth_str = ""
        smooth_str2 = ""
        if smooth_change:
            img_diff = synth_images - start_images
            x_diff = img_diff[:, :, :-1, :-1] - img_diff[:, :, :-1, 1:]
            y_diff = img_diff[:, :, :-1, :-1] - img_diff[:, :, 1:, :-1]
            sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, smooth_eps, 10000000)
            smooth_loss = torch.norm(sq_diff, smooth_beta / 2.0) ** (smooth_beta / 2.0)
            smooth_str = f'smooth loss: {float(smooth_loss * smooth_loss_lambda):<4.2f} '
            img_diff = synth_images2 - start_images2
            x_diff = img_diff[:, :, :-1, :-1] - img_diff[:, :, :-1, 1:]
            y_diff = img_diff[:, :, :-1, :-1] - img_diff[:, :, 1:, :-1]
            sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, smooth_eps, 10000000)
            smooth_loss2 = torch.norm(sq_diff, smooth_beta / 2.0) ** (smooth_beta / 2.0)
            smooth_str2 = f'smooth loss2: {float(smooth_loss * smooth_loss_lambda):<4.2f} '
        min_loss = L1Loss(learnable, torch.zeros_like(learnable))
        if use_entropy:
            dim = 0
            ent_sm = nn.Softmax(dim=dim).cuda()
            sm_out = ent_sm(torch.abs(learnable))
            min_loss = -(sm_out * torch.log(sm_out)).sum()
        min_losses.append(min_loss.item())
        loss = (class_loss + class_loss2) * class_loss_lambda + min_loss * min_loss_lambda + smooth_loss * smooth_loss_lambda
        logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f} {smooth_str}{smooth_str2}class loss {float((class_loss+class_loss2) * class_loss_lambda):<4.2f} min loss {float(min_loss*min_loss_lambda):<5.2f} avg confidence: {synth_conf}% avg confidence2: {synth_conf2}%')
        #####################################################################

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Update w_opt
        w_opt = calc_w(G, start_zs, start_ws, learnable, learn_param, batch=False)
        w_opt2 = calc_w(G, start_zs2, start_ws2, -learnable, learn_param, batch=False)

    return w_out, w_out2, all_synth_images, all_synth_images2, image_confs, image_confs2, min_losses
