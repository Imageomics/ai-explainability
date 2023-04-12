
import torch
import numpy as np

from PIL import Image, ImageDraw

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

def create_img_from_text(width, height, text):
    PAD = 2
    img_size = img.shape[:2]
    text_img = (np.ones((height, width, 3)) * 255).astype(np.uint8)
    text_img = Image.fromarray(text_img)
    text_img_dr = ImageDraw.Draw(text_img)
    font = ImageFont.load_default()
    text_img_dr.text((PAD, PAD), text, font=font, fill=(0, 0, 0))
    text_img = np.array(text_img)[:, :, :1]

    return text_img

def save_imgs(reals, fakes, confs, org_confs, output):
    reals = reals.cpu().detach().numpy()
    fakes = fakes.cpu().detach().numpy()

    reals = np.transpose(reals, (0, 2, 3, 1)) * 255
    fakes = np.transpose(fakes, (0, 2, 3, 1)) * 255
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
