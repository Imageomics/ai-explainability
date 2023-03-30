import copy
from matplotlib.colors import Normalize

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

def load_learnable(target_images):
    return torch.zeros_like(target_images.clone()).cuda().requires_grad_()

def project_no_generator(
    images,
    F,
    C,
    projection_lbl,
    num_steps                  = 200,
    init_lr                    = 0.001,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False


):
    def logprint(x):
        if verbose:
            print(x)

    if images is not None:
        assert images[0].shape == (3, 128, 128)

    # Features for target image. pretrained expects image to be unnormalized, but our feature extractor is already normalized.
    target_images = images.to(torch.float32)
    norm_target_images = NORMALIZE(target_images).cuda()

    learnable = load_learnable(target_images)
    optimizer = torch.optim.Adam([learnable], betas=(0.9, 0.999), lr=init_lr)

    all_synth_images = []
    image_confs = []
    min_losses = []

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

        
        synth_images_norm = norm_target_images + learnable
        synth_images = UNNORMALIZE(synth_images_norm).clamp(0, 1)
        all_synth_images.append(synth_images.detach().cpu().numpy())
        # Save projected W for each optimization step.

        synth_features = F(synth_images_norm)
        
        out = C(synth_features)
        conf = sm(out)
        image_confs.append(conf.detach().cpu().numpy().tolist())
        synth_conf = round(conf[:, projection_lbl].mean().item(), 4)*100
        lbls = torch.tensor([projection_lbl] * len(images)).cuda()
        class_loss = CELoss(out, lbls)

        #####################################################################
        # LOSS
        #####################################################################
        min_loss_lambda = .01
        class_loss_lambda = 1

        min_loss = L1Loss(learnable, torch.zeros_like(learnable))
        min_losses.append(min_loss.item())
        loss = class_loss * class_loss_lambda + min_loss * min_loss_lambda
        logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f} class loss {float(class_loss * class_loss_lambda):<4.2f} min loss {float(min_loss*min_loss_lambda):<5.2f} avg confidence: {synth_conf}%')
        #####################################################################

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

    return all_synth_images, image_confs, min_losses
