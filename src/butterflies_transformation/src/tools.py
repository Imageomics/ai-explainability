import numpy as np

import torch.nn as nn

from PIL import Image

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def show_reconstruction_images(reals, fakes, out_path="../tmp/recon.png"):
    reals = reals.cpu().detach().numpy()
    fakes = fakes.cpu().detach().numpy()

    reals = np.transpose(reals, (0, 2, 3, 1)) * 255
    fakes = np.transpose(fakes, (0, 2, 3, 1)) * 255
    final = None
    for img in reals:
        if final is None:
            final = img
        else:
            final = np.concatenate((final, img), axis=1)

    tmp = None
    for img in fakes:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final_img = np.concatenate((final, tmp), axis=0).astype(np.uint8)
    Image.fromarray(final_img).save(out_path)

def tensor_to_numpy_img(ten):
    np_img = ten.cpu().detach().numpy()
    if len(np_img.shape) == 4:
        np_img = (np.transpose(np_img, (0, 2, 3, 1)) * 255).astype(np.uint8)
    else:
        np_img = (np.transpose(np_img, (1, 2, 0)) * 255).astype(np.uint8)
    return np_img