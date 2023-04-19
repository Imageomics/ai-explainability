import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision.transforms import Resize
from PIL import Image

from iin_models.ae import IIN_AE
from models import ImageClassifier
from utils import tensor_to_numpy_img, set_seed, save_tensor_as_graph, create_z_from_label

"""
Goal: Create visual counterfactual
"""

def resize(img):
    return Resize((28, 28))(img)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--iin_ae', type=str, default=None)
    parser.add_argument('--img_classifier', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="tmp/")
    parser.add_argument('--exp_name', type=str, default="class_vcf")
    parser.add_argument('--num_features', type=int, default=20)

    return parser.parse_args()

if __name__ == "__main__":
    set_seed()
    args = get_args()

    lbls = torch.tensor([i for i in range(10)]).cuda()

    iin_ae = IIN_AE(4, args.num_features, 32, 1, 'an', False)
    iin_ae.load_state_dict(torch.load(args.iin_ae))
    iin_ae.cuda()
    iin_ae.eval()

    img_classifier = ImageClassifier(10)
    img_classifier.load_state_dict(torch.load(args.img_classifier))
    img_classifier.cuda()
    img_classifier.eval()

    sm = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    z = create_z_from_label(lbls)
    to_add = args.num_features - z.shape[1]
    z = torch.cat((z, torch.zeros((len(z), to_add)).cuda()), axis=1)
    z = z.unsqueeze(2).unsqueeze(3)
    
    imgs = iin_ae.decode(z)

    confs = sm(img_classifier(resize(imgs)))


    with torch.no_grad():
        imgs = tensor_to_numpy_img(imgs)
        for i in range(10):
            save_tensor_as_graph(z[i, :, 0, 0], os.path.join(args.output_dir, args.exp_name + f"_{i}_z.png"))
            Image.fromarray(imgs[i][:, :, 0]).save(os.path.join(args.output_dir, args.exp_name + f"_{i}.png"))
            print(confs[i][i])
