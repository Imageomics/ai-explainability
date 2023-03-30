import os
from argparse import ArgumentParser
from time import perf_counter
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from models import Encoder
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from data_tools import NORMALIZE
from project import project
from superpixel import superpixel

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--mode', type=str, default='filtered', choices=['afhqv2', 'filtered', 'filtered_no_pre', 'filtered_cond', 'filtered_cond_v2', 'original', 'original_nohybrid'])
    parser.add_argument('--hybrid', action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_cf", type=int, default=2)
    parser.add_argument("--layers_start", type=int, default=0)
    parser.add_argument("--layers_end", type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cls_lambda', type=float, default=1)
    parser.add_argument('--div_lambda', type=float, default=1)
    parser.add_argument('--img_lambda', type=float, default=0.003)
    parser.add_argument('--img_entropy_lambda', type=float, default=0.01)
    parser.add_argument('--cf_lambda', type=float, default=0.0)
    parser.add_argument('--smooth_lambda', type=float, default=0.0000001)
    parser.add_argument('--loss_fn', type=str, default="l1")
    parser.add_argument('--src_sub', type=str, default='aglaope')
    parser.add_argument('--tgt_sub', type=str, default='emma')
    parser.add_argument('--filter_sub', action='store_true', default=False)
    parser.add_argument('--use_superpixel', action='store_true', default=False)
    parser.add_argument('--start', type=str, default='rand', choices=['begin', 'end', 'rand'])
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_superpixels', type=int, default=100)
    parser.add_argument('--selected_pixels', type=int, default=10)

    # Best lambdas
    """
    lr: 0.01
    class: 10.0
    image diff: 100
    image diff entropy: 10
    counter factual: 1.0
    smooth: 0.00001
    """

    args = parser.parse_args()

    args.res = 128
    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        #args.backbone = '../saved_models/vgg_backbone.pt'
        #args.classifier = '../saved_models/vgg_classifier.pt'
        args.backbone = '../saved_models/vgg_smooth_filtered_backbone.pt'
        args.classifier = '../saved_models/vgg_smooth_filtered_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
        args.num_classes = 27
    elif args.mode == 'filtered_no_pre':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../output/filtered_classifier_6/vgg_filtered_backbone.pt'
        args.classifier = '../output/filtered_classifier_6/vgg_filtered_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
        args.num_classes = 27
    elif args.mode == 'filtered_cond':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w_cond/checkpoints/best_model.pt'
        args.network = 'stylegan3_cond/butterfly_training_runs/00029-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.backbone = '../saved_models/vgg_smooth_filtered_backbone.pt'
        args.classifier = '../saved_models/vgg_smooth_filtered_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered_cond'
        args.num_classes = 27
    elif args.mode == 'filtered_cond_v2':
        args.encoder = None #'encoder4editing/butterfly_training_only_one_w_cond/checkpoints/best_model.pt'
        args.network = 'stylegan3_cond/butterfly_training_runs/00041-stylegan3-r-train_128_128-gpus2-batch64-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered_cond_v2'
        args.num_classes = 27
    elif args.mode == 'original':
        args.encoder = 'encoder4editing/butterfly_org_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original/00004-stylegan3-r-train_original_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone_original.pt'
        args.classifier = '../saved_models/vgg_classifier_original.pt'
        args.dataset_root_train = '../datasets/original/train/'
        args.dataset_root_test = '../datasets/original/test/'
        args.outdir = '../output/ganspace_reconstruction_original'
        args.num_classes = 37
    elif args.mode == 'original_nohybrid':
        args.encoder = 'encoder4editing/butterfly_org_no_hybrid_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original_nohybrid/00006-stylegan3-r-train_original_no_hybrid_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone_original_nohybrid.pt'
        args.classifier = '../saved_models/vgg_classifier_original_nohybrid.pt'
        args.dataset_root_train = '../datasets/original_nohybrid/train/'
        args.dataset_root_test = '../datasets/original_nohybrid/test/'
        args.outdir = '../output/ganspace_reconstruction_original_nohybrid'
        args.num_classes = 34
    elif args.mode == 'afhqv2':
        args.encoder = '/home/carlyn.1/ImagenomicsButterflies/src/encoder4editing/afhqv2_training_only_one_w/checkpoints/best_model.pt'
        args.network = 'stylegan3_org/afhqv2_model/stylegan3-r-afhqv2-512x512.pkl'
        args.backbone = '/home/carlyn.1/ImagenomicsButterflies/output/afhqv2_classifier_4/vgg_afhqv2_backbone.pt'
        args.classifier = '/home/carlyn.1/ImagenomicsButterflies/output/afhqv2_classifier_4/vgg_afhqv2_classifier.pt'
        args.dataset_root_train = '../datasets/afhqv2/train/'
        args.dataset_root_test = '../datasets/afhqv2/test/'
        args.outdir = '../output/ganspace_reconstruction_afhqv2'
        args.num_classes = 3
        args.res = 512
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def get_sub_lbl_map(dset):
    subspecies = set()
    for root, dirs, files in os.walk(dset):
        for f in files:
            subspecies.add(root.split(os.path.sep)[-1])
    
    sub_lbl_map = {}
    subspecies = sorted(list(subspecies))
    for i, sub in enumerate(subspecies):
        sub_lbl_map[sub] = i

    return sub_lbl_map


def load_images(paths, res=128):
    images = []
    for path in paths:
        img = transforms.Resize((res, res))(Image.open(path)),
        images.append(np.array(img[0]))

    return np.array(images)

def load_data(dir_path, type="train", filter_sub=None, res=128):
    print(filter_sub)
    add_sub = f"{filter_sub}_" if filter_sub is not None else ""
    print(add_sub)
    paths = load_json(os.path.join(dir_path, f"{type}_{add_sub}paths.json"))
    projections = np.transpose(np.load(os.path.join(dir_path, f"{type}_{add_sub}projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
    projections = (projections * 255).astype(np.uint8)
    ws = np.load(os.path.join(dir_path, f"{type}_{add_sub}ws.npz"))["ws"]

    originals = load_images(paths, res=res)
    subspecies = list(map(lambda x: x.split(os.path.sep)[-2], paths))

    return subspecies, originals, projections, ws

def create_images(G, w, no_repeat=False):
    if no_repeat:
        w_input = w
    else:
        w_input = w.unsqueeze(1).repeat((1, G.num_ws, 1)).cuda()
    synth_images = G.synthesis(w_input, noise_mode='const')
    synth_images = (synth_images + 1) * (1/2)
    synth_images = synth_images.clamp(0, 1)
    return synth_images

def to_numpy_img(img):
    return np.transpose(img.detach().cpu().numpy()*255, axes=(1, 2, 0)).astype(np.uint8)

def to_tensor_img(img):
    return torch.tensor(np.transpose(img.astype(np.float32) / 255, axes=(2, 0, 1))).to(torch.float)

def save_image(img, path):
    np_img = to_numpy_img(img[0])
    Image.fromarray(np_img).save(path)

def ten_grayscale(x):
    return x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114

def grayscale(x):
    rv = np.zeros_like(x[:3])
    rv = x[:, :, 0] * 0.299 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.114
    return rv

def get_diff_img(a, b):
    diff_img = (grayscale(b) - grayscale(a)).astype(np.float)
    diff_pos = np.copy(diff_img)
    diff_pos[diff_pos < 0] = 0
    diff_neg = -np.copy(diff_img)
    diff_neg[diff_neg < 0] = 0
    
    THRESH = 0.3
    diff_pos -= diff_pos.min()
    diff_pos /= diff_pos.max()
    diff_pos[diff_pos < THRESH] = 0.0
    diff_pos *= 255

    diff_neg -= diff_neg.min()
    diff_neg /= diff_neg.max()
    diff_neg[diff_neg < THRESH] = 0.0
    diff_neg *= 255

    diff_img = np.concatenate((np.expand_dims(diff_neg, 2), np.expand_dims(np.zeros_like(diff_neg), 2), np.expand_dims(diff_pos, 2)), axis=2).astype(np.uint8)

    #diff_img = np.tile(np.expand_dims(diff_img, 2), (1, 1, 3)).astype(np.uint8)
    return diff_img


def pca(data):
    data = np.array(data)
    mu = data.mean(0)
    center = data - mu
    cov = (1/len(data))*(center.T @ center)
    eig_val, eig_vec = np.linalg.eig(cov)
    W = np.copy(eig_vec)

    return (data - mu) @ W.T, W, mu, eig_vec, eig_val

def add_text(img, text=""):
    TEXT_HEIGHT = 20
    PAD = 4
    img_size = img.shape[:2]
    text_img = (np.ones((TEXT_HEIGHT, img_size[1], 3)) * 255).astype(np.uint8)
    text_img = Image.fromarray(text_img)
    text_img_dr = ImageDraw.Draw(text_img)
    text_img_dr.text((PAD, PAD), text, fill=(0, 0, 0))
    text_img = np.array(text_img)
    return np.concatenate((text_img, img), axis=0)

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    """
    ?Question: Will a couterfactual in the pca ganspace be more distinguishable?
    
    Control: Show counterfactual in w space
    Experiment: Show counterfactual in pca w space

    !Answer: Seems that there is either no difference or pca is worse.
    ! In both cases, adversarial examples seem to take over
    """

    sub_lbl_map = get_sub_lbl_map(args.dataset_root_train)
    if args.filter_sub:
        subspecies, originals, projections, ws = load_data(args.outdir, type="train", filter_sub=args.src_sub, res=args.res)
        src_ws_list = list(ws)
        subspecies, originals, projections, ws = load_data(args.outdir, type="train", filter_sub=args.tgt_sub, res=args.res)
        tgt_ws_list = list(ws)
    else:
        subspecies, originals, projections, ws = load_data(args.outdir, type="train", res=args.res)
        src_ws_list = list(map(lambda x: x[1], filter(lambda x: x[0]==args.src_sub, zip(subspecies, ws))))
        tgt_ws_list = list(map(lambda x: x[1], filter(lambda x: x[0]==args.tgt_sub, zip(subspecies, ws))))
    src_w = np.array(src_ws_list).mean(0)
    tgt_w = np.array(tgt_ws_list).mean(0)

    G, _, F, C = load_models(args.network, args.backbone, args.classifier, args.num_classes)
    change_v = torch.tensor(tgt_w - src_w).unsqueeze(0).repeat((G.num_ws, 1)).cuda() # [w_dim]
    src_w = torch.tensor(np.array(src_ws_list)[:args.batch_size]).cuda() #[batch x w_dim]
    if args.start == 'end':
        cf = torch.ones((args.num_cf, src_w.shape[-1])).unsqueeze(1).repeat((1, G.num_ws, 1)).cuda().requires_grad_()
    elif args.start == 'begin':
        cf = (torch.zeros((args.num_cf, src_w.shape[-1]))).unsqueeze(1).repeat((1, G.num_ws, 1)).cuda().requires_grad_()
    elif args.start == 'rand':
        cf = torch.rand((args.num_cf, src_w.shape[-1])).unsqueeze(1).repeat((1, G.num_ws, 1)).cuda().requires_grad_()


    """
    ? Question: Will restricting our path b/w mean images prevent adversarial cases?
    ! Answer: This seems to be the case!
    ! Problem: all traits & noise are visible in the transformation
    ? Question: Will enforcing multiple counterfactuals encourage more isolation?
    ! Insight: previous works have indicated this to be true.

    
    """

    #LEARNABLE_WS = 1

    #src_proj_ten = torch.tensor(np.transpose(src_proj / 255, axes=(2, 0, 1))).unsqueeze(0).cuda()
    #src_w = torch.tensor(src_w).unsqueeze(0).cuda()
    #src_z = torch.tensor(src_z).unsqueeze(0).cuda()
    tgt_lbl = torch.tensor([sub_lbl_map[args.tgt_sub]]).cuda()
    src_lbl = torch.tensor([sub_lbl_map[args.src_sub]]).cuda()
    #cf = torch.zeros((1, LEARNABLE_WS, src_w.shape[-1])).cuda().requires_grad_()
    #cf_z = torch.zeros_like(src_z).cuda().requires_grad_()

    G.cuda()
    F.cuda()
    C.cuda()
    F.eval()
    C.eval()

    with torch.no_grad():
        src_proj_ten = create_images(G, src_w, no_repeat=False).unsqueeze(1).repeat((1, args.num_cf, 1, 1, 1))
        src_proj_ten = src_proj_ten.view(args.batch_size*args.num_cf, src_proj_ten.shape[2], src_proj_ten.shape[3], src_proj_ten.shape[4])

    CLAMP_MIN = 1e-3
    sm = nn.Softmax(dim=1)
    CELoss = nn.CrossEntropyLoss(reduction='sum').cuda()
    sigmoid = nn.Sigmoid().cuda()
    loss_fn = None
    if args.loss_fn == "l2":
        loss_fn = nn.MSELoss().cuda()
    elif args.loss_fn == "l1":
        loss_fn = nn.L1Loss().cuda()
    elif args.loss_fn == "entropy":
        def calc_loss(x, tmp):
            #out = sm(torch.clamp(x, 0, 1))
            out = sm(x)
            return (-out * torch.log(out)).view(len(x), -1).sum().mean()

        loss_fn = calc_loss
    
    if args.optim == 'adam':
        optimizer = torch.optim.Adam([cf], betas=(0.9, 0.999), lr=args.lr)
    else:
        optimizer = torch.optim.SGD([cf], lr=args.lr)


    for epoch in range(args.epochs):
        #input_w = ((src_z + cf_z) @ torch.tensor(np.linalg.inv(W).T).cuda()) + torch.tensor(mu).cuda()
        #input_w = torch.cat((src_w.repeat((1, G.num_ws-LEARNABLE_WS, 1)), src_w.repeat((1, LEARNABLE_WS, 1)) + cf), axis=1)
        #input_w = src_w + change_v * torch.clamp(cf, CLAMP_MIN, 1)
        # cf [# counterfactuals x w_dim]
        # change_v [w_dim]
        # change_v * cf [# counterfactuals x w_dim]
        # src_w [batch_size x w_dim]

        #
        change_path = change_v * sigmoid(cf)
        #change_path = change_v * cf
        #change_path = cf
        #print(change_path.sum())
        
        if True:
            input_w = src_w.unsqueeze(1).repeat((args.num_cf, G.num_ws, 1)) + change_path
            #input_w = input_w.view(-1, input_w.shape[-1])
            #input_w = input_w.unsqueeze(1).repeat((1, G.num_ws, 1)).cuda()
            #tmp = src_w.unsqueeze(1).repeat((1, args.num_cf, 1)).view(-1, src_w.shape[-1]).unsqueeze(1).repeat((1, G.num_ws, 1))
            #tmp[:, args.layers_start:args.layers_end, :] = input_w[:, args.layers_start:args.layers_end, :]
            img = create_images(G, input_w, no_repeat=True)
        else:
            input_w = src_w.unsqueeze(1).repeat((1, args.num_cf, 1)) + change_path
            input_w = input_w.view(-1, input_w.shape[-1])
            img = create_images(G, input_w, no_repeat=False)
        img_diff = torch.abs(ten_grayscale(img) - ten_grayscale(src_proj_ten))
        #img_diff = torch.abs(img - src_proj_ten).sum(1)
        #img_diff = torch.abs(F(NORMALIZE(img)) - F(NORMALIZE(src_proj_ten)))
        # Smooth loss
        #x_diff = img_diff[:, :-1, :-1] - img_diff[:, :-1, 1:]
        #y_diff = img_diff[:, :-1, :-1] - img_diff[:, 1:, :-1]
        #sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, 1e-3, 10000000)
        #smooth_loss = torch.norm(sq_diff, 1) ** (1)

        # Img diff entropy loss
        img_diff = img_diff.view(len(img_diff), -1)
        #print(img_diff.shape)
        #print(img_diff.sum())
        #diff_prob = sm(img_diff)
        #img_diff_entropy_loss = (-diff_prob * (torch.log(diff_prob) / torch.log(torch.tensor(2.0)))).sum(1).mean()
        
        # Img diff loss
        #img_diff_loss = torch.norm(img.view(len(img), -1) - src_proj_ten.view(len(src_proj_ten), -1), dim=1, p=2).mean()
        img_diff_loss = img_diff.sum(1).mean()
        #img_diff_loss = (sigmoid(img_diff) - 0.5).sum(1).mean()

        # class loss
        out = C(F(NORMALIZE(img)))
        all_probs = sm(out) 
        prob_src = all_probs[0][src_lbl.item()].item()
        prob_tgt = all_probs[0][tgt_lbl.item()].item()
        loss = CELoss(out, tgt_lbl.repeat(args.num_cf*args.batch_size))

        # Combined
        #input_w_g = src_w[:1] + change_v[:1] * sigmoid(cf).max(dim=0)[0]
        #img_g = create_images(G, input_w_g, no_repeat=False)
        #out_g = C(F(NORMALIZE(img_g)))
        #loss += CELoss(out_g, tgt_lbl)

        # CF loss
        cf_loss = loss_fn(cf, torch.zeros_like(cf))
        #d_out = D(img, c=1)
        #d_loss = torch.nn.functional.softplus(-d_out).cuda()

        div_loss = torch.tensor(0.0).cuda()
        use_img_diff = False
        #div_matrix = sigmoid(cf * 1000)
        div_matrix = sigmoid(cf.view(len(cf), -1))
        #div_matrix = cf.view(len(cf), -1)
        if use_img_diff:
            div_matrix = sigmoid(img_diff*1000) #! Need to fix here for multiple batches
        #div_matrix = torch.clamp(cf, CLAMP_MIN, 1)
        for i in range(len(div_matrix)):
            for j in range(i+1, len(div_matrix)):
                if i == j: continue
                a = div_matrix[i]
                b = div_matrix[j]
                #div_loss += ((torch.clamp(cf[i], 0, 1) / torch.norm(torch.clamp(cf[i], 0, 1))) @ (torch.clamp(cf[j], 0, 1) / torch.norm(torch.clamp(cf[j], 0, 1))))**2
                #div_loss += ((cf[i] / torch.norm(cf[i])) @ (cf[j] / torch.norm(cf[j])))**2
                #div_loss += ((a / max(torch.norm(a), 1e-8)) @ (b / max(torch.norm(b), 1e-8)))**2
                div_loss += torch.abs(nn.CosineSimilarity(dim=0, eps=1e-8)(a, b))

                #div_loss += torch.norm(b-a)
        #div_loss = torch.sqrt(div_loss)

        #print(f"Epoch: {epoch+1} \
        #    Global Conf: {round(sm(out_g)[0][tgt_lbl.item()].item()*100, 2)}% \
        #    Conf: ({round(prob_src*100, 2)}%, {round(prob_tgt*100, 2)}%), \
        #    Class: {round(loss.item(), 4)}\
        #    CF: {round(cf_loss.item(), 4)}, \
        #    Img loss: {round(img_diff_loss.item(), 4)} \
        #    Img entropy: {round(img_diff_entropy_loss.item(), 4)} \
        #    Smooth loss: {round(smooth_loss.item(), 4)}, \
        #    Diversity Loss: {round(div_loss.item(), 4)}")
        print(f"Epoch: {epoch+1} \
            Conf: ({round(prob_src*100, 2)}%, {round(prob_tgt*100, 2)}%), \
            Class: {round(loss.item(), 4)}, \
            CF: {round(cf_loss.item(), 4)}, \
            Img loss: {round(img_diff_loss.item(), 4)}, \
            Diversity Loss: {round(div_loss.item(), 4)}")
#        loss = loss * args.cls_lambda + \
#                cf_loss * args.cf_lambda + \
#                img_diff_loss * args.img_lambda + \
#                img_diff_entropy_loss * args.img_entropy_lambda + \
#                smooth_loss * args.smooth_lambda + \
#                div_loss * args.div_lambda
        loss = loss * args.cls_lambda + \
                cf_loss * args.cf_lambda + \
                img_diff_loss * args.img_lambda + \
                div_loss * args.div_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (epoch % args.save_freq) == 0 and epoch > 0:
            use_superpixel = args.use_superpixel
            tmp_img = img.view(args.batch_size, args.num_cf, img.shape[1], img.shape[2], img.shape[3])
            tmp_probs = all_probs.view(args.batch_size, args.num_cf, -1)
            final_img = None
            for i in range(args.batch_size):
                batch_probs = tmp_probs[i]
                src_proj = to_numpy_img(src_proj_ten[i*args.num_cf])
                src_sp = superpixel(src_proj, pixels=args.num_superpixels)
                src_proj_text = add_text(src_proj, f"src: {args.src_sub}")
                row_img = np.copy(src_proj_text)
                for j, ten_img in enumerate(tmp_img[i]):
                    tgt_img_np = to_numpy_img(ten_img)
                    tgt_sp = superpixel(tgt_img_np, pixels=args.num_superpixels)
                    diff_img = get_diff_img(src_proj, tgt_img_np)
                    diff_sp = superpixel(diff_img, pixels=args.num_superpixels)
                    chosen_sp = diff_sp
                    tgt_img_np_txt = add_text(tgt_img_np, f"tgt: {args.tgt_sub} ({round(batch_probs[j][tgt_lbl.item()].item()* 100, 2)}%)")
                    diff_img = add_text(diff_img, "diff image")
                    if use_superpixel:
                        best_pixels = []
                        sp_confs = []
                        conf_diffs = []
                        total_conf = 0
                        for loop in range(2):
                            for sp_i in range(args.selected_pixels):
                                best = [-1, 0]
                                tmp_start = np.copy(src_proj)
                                for sp_lbl in best_pixels:
                                    mask = chosen_sp.astype(np.uint8) == sp_lbl
                                    tmp_start[mask] = tgt_img_np[mask]
                                for sp_lbl in range(np.max(chosen_sp)+1):
                                    if sp_lbl in best_pixels: continue
                                    mask = chosen_sp.astype(np.uint8) == sp_lbl
                                    tmp = np.copy(tmp_start)
                                    tmp[mask] = tgt_img_np[mask]
                                    if np.sum(tmp) <= 0.0: continue
                                    tmp = to_tensor_img(tmp)
                                    conf = sm(C(F(NORMALIZE(tmp.unsqueeze(0).cuda()))))[0, tgt_lbl.item()].item()
                                    if conf > best[1]:
                                        best[0] = sp_lbl
                                        best[1] = conf
                                print(f"{sp_i}: {best[1]}")
                                if best[1] < total_conf:
                                    break
                                conf_diffs.append(total_conf - best[1])
                                sp_confs.append(best[1])
                                best_pixels.append(best[0])
                                total_conf = best[1]
                            keep = [0, 0]
                            if loop == 0:
                                for k in range(len(best_pixels)):
                                    if conf_diffs[k] > keep[1]:
                                        keep[1] = conf_diffs[k]
                                        keep[0] = best_pixels[k]
                                best_pixels = [keep[0]]
                                total_conf = 0
                        result_img = np.copy(src_proj)
                        for sp_lbl in best_pixels:
                            mask = chosen_sp.astype(np.uint8) == sp_lbl
                            result_img[mask] = tgt_img_np[mask]
                        
                        result_img = to_tensor_img(result_img)
                        conf = sm(C(F(NORMALIZE(result_img.unsqueeze(0).cuda()))))[0, tgt_lbl.item()].item()
                        result_np = to_numpy_img(result_img)
                        result_diff_img = get_diff_img(src_proj, result_np).astype(np.float)
                        for sp_lbl, conf_v in zip(best_pixels, sp_confs):
                            mask = chosen_sp.astype(np.uint8) == sp_lbl
                            tmp = np.copy(result_np)
                            tmp[mask] = src_proj[mask]
                            tmp = to_tensor_img(tmp)
                            conf_v = sm(C(F(NORMALIZE(tmp.unsqueeze(0).cuda()))))[0, tgt_lbl.item()].item()
                            print(conf, conf_v)
                            conf_diff = conf - conf_v
                            result_diff_img[mask] = (np.ones_like(result_diff_img[mask])*255 * (max(0, conf_diff)))
                        result_diff_img /= result_diff_img.max()
                        result_diff_img *= 255
                        result_diff_img = result_diff_img.astype(np.uint8)
                        result_img_np_txt = add_text(result_np, f"(SP) ({round(conf* 100, 2)}%)")
                        result_diff_img = add_text(result_diff_img, "(SP) diff image")


                    if use_superpixel:
                        row_img = np.concatenate((row_img, tgt_img_np_txt, diff_img, result_img_np_txt, result_diff_img), axis=1)
                    else:
                        row_img = np.concatenate((row_img, tgt_img_np_txt, diff_img), axis=1)
                if final_img is None:
                    final_img = row_img
                else:
                    final_img = np.concatenate((final_img, row_img), axis=0)
            Image.fromarray(final_img).save("cf_test.png")
    #save_image(img, "test_tgt.png")

    

        
   

    




    