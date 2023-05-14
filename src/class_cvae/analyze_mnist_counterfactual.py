# Do counterfactual analysis

# Do on test data

"""
1. Do on all class means to all other class means
2. Perform on random sample of individual images

Every transformation should have:
Original, Reconstruction, Counterfactual image, delta z, difference highlights

Stop counterfactual training after:
1. X num of iterations
2. The moment the prediction is the target
3. The moment the prediction confidence is larger than X
"""

import random
import os

from tqdm import tqdm

import torch
import torch.nn as nn

from torchvision.datasets import MNIST
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt

from logger import Logger
from models import IIN_AE_Wrapper, ResNet50
from options import MNIST_CF_Analysis_Configs
from utils import create_z_from_label, create_graph_from_tensor, fig_to_numpy, tensor_to_numpy_img, create_diff_img, set_seed

def load_models(configs):
    num_att_vars = len(create_z_from_label(torch.tensor([0]))[0])
    if configs.only_recon:
        num_att_vars = None

    configs.num_att_vars = num_att_vars
    
    iin_ae = IIN_AE_Wrapper(configs)
    img_classifier = ResNet50(num_classes=10, img_ch=configs.in_channels)

    iin_ae.load_state_dict(torch.load(configs.ae))
    img_classifier.load_state_dict(torch.load(configs.img_classifier))

    return iin_ae, img_classifier

def resize(img):
    return T.Resize((28, 28))(img)

def load_data(configs):
    test_transform = T.Compose([
        T.Resize((configs.img_size, configs.img_size)),
        T.ToTensor()
    ])

    test_dset = MNIST(root="data", train=False, transform=test_transform)

    return test_dset

def create_counterfactual(ae, img_classifier, dset, logger, src, tgt, configs):
    ae = ae.cuda()
    img_classifier = img_classifier.cuda()
    ae.eval()
    img_classifier.eval()

    # Obtain representative Z
    src_idx = []
    for i, (img, lbl) in enumerate(dset):
        if lbl != src: continue
        src_idx.append(i)

    org_img = None
    org_z = None
    with torch.no_grad():
        if configs.start_option == "random":
            i = random.choice(src_idx)
            org_img = test_dset[i][0].unsqueeze(0)
            org_z = ae.encode(org_img.cuda())
        elif configs.start_option == "mean":
            for i in src_idx:
                img = test_dset[i][0].unsqueeze(0)
                z = ae.encode(img.cuda())
                if org_z is None:
                    org_z = z
                else:
                    org_z += z
            org_z /= len(src_idx)


    src_lbl = torch.tensor([src]).cuda()
    tgt_lbl = torch.tensor([tgt]).cuda()
    
    # Setup Optimization
    sm = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    l1_loss_fn = nn.L1Loss()
    cel_loss_fn = nn.CrossEntropyLoss()

    z_chg = torch.zeros(configs.num_attributes).cuda()
    z_chg = z_chg.requires_grad_(True)
    
    params = [z_chg]
    optimizer = torch.optim.Adam(params, lr=configs.lr)

    with torch.no_grad():
        recon_img = ae.decode(org_z)

    loop = True
    cur_iter = 0
    while loop:
        if configs.stop_option == "iters" and cur_iter >= configs.num_iters: break
        if cur_iter >= configs.max_iters: break
        cur_iter += 1

        z_edit = org_z.clone()
        z_edit[:, :configs.num_attributes] += z_chg.unsqueeze(0)
        min_chg_loss = l1_loss_fn(z_chg, torch.zeros_like(z_chg).cuda()) * configs.min_chg_lambda

        cf_example = ae.decode(z_edit)

        out = img_classifier(resize(cf_example))

        cls_loss = cel_loss_fn(out, tgt_lbl) * configs.cls_lambda
        loss = min_chg_loss + cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        confs = sm(out)[0]
        src_conf = confs[src_lbl[0]].item()
        tgt_conf = confs[tgt_lbl[0]].item()
        _, pred = torch.max(out, 1)
        pred = pred[0].item()

        
        out_str = f"({src}) => ({tgt}) || Iteration: {cur_iter+1} | Loss: {loss.item()} | Min Chg Loss: {min_chg_loss.item()} | Class Loss: {cls_loss.item()} | Source Conf: {src_conf} | Target Conf: {tgt_conf}"
        if cur_iter % 1000 == 0:
            logger.log(out_str)

        if configs.stop_option == "flip" and pred == tgt: break
        if configs.stop_option == "conf" and tgt >= configs.conf_thresh: break
    
    logger.log(out_str)
    with torch.no_grad():
        # Original, Reconstruction, Counterfactual image, delta z, difference highlights
        if org_img is not None:
            org_img = tensor_to_numpy_img(org_img)[0][:, :, 0]
        recon_img = tensor_to_numpy_img(recon_img)[0][:, :, 0]
        cf_example = tensor_to_numpy_img(cf_example)[0][:, :, 0]

        delta_z_fig = create_graph_from_tensor(z_edit[0], configs.font_size)
        delta_z = fig_to_numpy(delta_z_fig)
        plt.close()

        diff_img = create_diff_img(recon_img, cf_example)

        return CounterfactualOutput(org_img, recon_img, cf_example, delta_z, diff_img, src_conf, tgt_conf)

class CounterfactualOutput():
    def __init__(self, org_img, recon_img, cf_example, delta_z, diff_img, cf_src_conf, cf_tgt_conf):
        self.org_img = org_img
        self.recon_img = recon_img
        self.cf_example = cf_example
        self.delta_z = delta_z
        self.diff_img = diff_img
        self.cf_src_conf = cf_src_conf
        self.cf_tgt_conf = cf_tgt_conf

def save_results(save_dir, cf_output):
    if cf_output.org_img is not None:
        Image.fromarray(cf_output.org_img).save(os.path.join(save_dir, "original.png"))
    Image.fromarray(cf_output.recon_img).save(os.path.join(save_dir, "reconstruction.png"))
    Image.fromarray(cf_output.cf_example).save(os.path.join(save_dir, "counterfactual.png"))
    Image.fromarray(cf_output.delta_z).save(os.path.join(save_dir, "delta_z.png"))
    Image.fromarray(cf_output.diff_img).save(os.path.join(save_dir, "diff_img.png"))
    with open(os.path.join(save_dir, "confidence.txt"), 'w') as f:
        f.writelines([
            f"Counterfactual source confidence: {cf_output.cf_src_conf}\n",
            f"Counterfactual target confidence: {cf_output.cf_tgt_conf}",
        ])
    
    

if __name__ == "__main__":
    configs = MNIST_CF_Analysis_Configs()
    if configs.only_recon:
        configs.force_hardcode = False
        configs.num_attributes = configs.num_features
    
    set_seed(configs.seed)
        
    ae, img_classifier = load_models(configs)
    test_dset = load_data(configs)

    logger = Logger(configs.output_dir, configs.exp_name)
    results_dir = os.path.join(logger.get_path(), "results")
    os.makedirs(results_dir, exist_ok=True)

    with tqdm(total=100) as pbar:
        for src in range(10):
            for tgt in range(10):
                if src == tgt: continue
                cf_output = create_counterfactual(ae, img_classifier, test_dset, logger, src, tgt, configs)
                save_dir = os.path.join(results_dir, f"{src}_to_{tgt}")
                os.makedirs(save_dir, exist_ok=True)
                save_results(save_dir, cf_output)
                pbar.update(1)