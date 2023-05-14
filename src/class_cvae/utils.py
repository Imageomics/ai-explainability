import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pad

class MaxQueue:
    def __init__(self, size=10):
        self.size = 10
        self.arr = []

    def add(self, z, conf):
        if len(self.arr) < self.size:
            self.arr.append((z, conf))
        else:
            if len(list(filter(lambda x: x[1] < conf, self.arr))) > 0:
                self.arr.pop(0)
                self.arr.append((z, conf))

        self.arr = sorted(self.arr, key=lambda x: x[1])

    def avg_val(self):
        acc = torch.zeros_like(self.arr[0][0])
        for z, _ in self.arr:
            acc += z
        return acc / len(self.arr)

def set_seed(seed=2023):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_img_from_text(width, height, text):
    PAD = 2
    text_img = (np.ones((height, width, 3)) * 255).astype(np.uint8)
    text_img = Image.fromarray(text_img)
    text_img_dr = ImageDraw.Draw(text_img)
    font = ImageFont.load_default()
    text_img_dr.text((PAD, PAD), text, font=font, fill=(0, 0, 0))
    text_img = np.array(text_img)[:, :, :1]

    return text_img

def tensor_to_numpy_img(x):
    x = x.cpu().detach().numpy()
    x = np.transpose(x, (0, 2, 3, 1)) * 255
    return x.astype(np.uint8)

def create_diff_img(org, new):
    diffs = org.astype(np.float64) - new.astype(np.float64)
    if len(diffs.shape) > 2:
        if diffs.shape[2] > 1:
            diffs = np.sqrt((diffs**2).sum(2))
        else:
            diffs = np.abs(diffs[:, :, 0])

    diffs -= diffs.min()
    diffs /= diffs.max()

    diffs = (diffs * 255).astype(np.uint8)

    return diffs
    

def save_imgs(reals, fakes, confs, org_confs, output):
    reals = tensor_to_numpy_img(reals).astype(np.float)
    fakes = tensor_to_numpy_img(fakes).astype(np.float)
    diffs = (reals - fakes)
    diffs_pos = np.copy(diffs)
    diffs_pos[diffs_pos < 0] = 0
    diffs_neg = np.copy(diffs)
    diffs_neg[diffs_neg > 0] = 0
    diffs_neg *= -1

    final = None
    for i in range(len(reals)):
        if final is None:
            final = create_img_from_text(reals.shape[1], 14, f"{int(round(org_confs[i].item(), 2)*100)}%")
        else:
            final = np.concatenate((final, create_img_from_text(reals.shape[1], 14, f"{int(round(org_confs[i].item(), 2)*100)}%")), axis=1)
    
    tmp = None
    for i in range(len(reals)):
        if tmp is None:
            tmp = create_img_from_text(reals.shape[1], 14, f"{int(round(confs[i].item(), 2)*100)}%")
        else:
            tmp = np.concatenate((tmp, create_img_from_text(reals.shape[1], 14, f"{int(round(confs[i].item(), 2)*100)}%")), axis=1)
    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in reals:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)
    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)

    tmp = None
    for img in fakes:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in diffs_pos:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in diffs_neg:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)

    Image.fromarray(final[:, :, 0]).save(output)

def cub_pad(img):
    pad_size = max(max(img.height, img.width), 500)
    y_to_pad = pad_size - img.height
    x_to_pad = pad_size - img.width

    top_to_pad = y_to_pad // 2
    bottom_to_pad = y_to_pad - top_to_pad
    left_to_pad = x_to_pad // 2
    right_to_pad = x_to_pad - left_to_pad

    return pad(img,
        (left_to_pad, top_to_pad, right_to_pad, bottom_to_pad),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

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

def fig_to_numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 3))
    return data

def create_graph_from_tensor(ten, font_size=12):
    z = ten.detach().cpu().numpy()
    ticks = np.arange(len(z))
    colors = ['g' if v >= 0 else 'r' for v in z]
    fig, ax = plt.subplots()
    ax.bar(ticks, z, color=colors, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.set_xticks(ticks)
    ax.tick_params(axis='x', labelsize=font_size)
    fig.tight_layout(pad=0)

    return fig

def save_tensor_as_graph(ten, output="tensor_graph.png"):
    # Assumes 1-D tensor
    fig = create_graph_from_tensor(ten)
    fig.savefig(output)
    plt.close()

def calc_img_diff_loss(org_img_recon, imgs_recon, loss_fn):
    diffs = (org_img_recon - imgs_recon)
    loss = nn.L1Loss()(diffs, torch.zeros_like(diffs).cuda())
    return loss

def get_hardcode_mnist_latent_map():
    return {
        0: np.array([[1, 0, 1, 1, 1, 1, 1]]),
        1: np.array([[0, 0, 0, 0, 1, 0, 1]]),
        2: np.array([[1, 1, 1, 0, 1, 1, 0]]),
        3: np.array([[1, 1, 1, 0, 1, 0, 1]]),
        4: np.array([[0, 1, 0, 1, 1, 0, 1]]),
        5: np.array([[1, 0, 1, 1, 0, 0, 1]]),
        6: np.array([[1, 1, 1, 1, 0, 1, 1]]),
        7: np.array([[1, 0, 0, 0, 1, 0, 1]]),
        8: np.array([[1, 1, 1, 1, 1, 1, 1]]),
        9: np.array([[1, 1, 0, 1, 1, 0, 1]]),
    }

def create_z_from_label(lbls):
    z_map = get_hardcode_mnist_latent_map()

    z = z_map[lbls[0].item()]
    for i in range(1, len(lbls)):
        z = np.concatenate((z, z_map[lbls[i].item()]), axis=0)

    return torch.tensor(z).cuda()