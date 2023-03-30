import os
import random
from argparse import ArgumentParser

from tqdm import tqdm 

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, LBFGS
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Res50, Classifier, VGG16, VGG16_Decoder
from loggers import Logger
from data_tools import NORMALIZE, image_transform, to_tensor, test_image_transform, rgb_img_loader, to_grayscale, cosine_similarity
from loss import TransformLoss
from datasets import ImageList

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--min_loss", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--net", type=str, default="vgg")
    parser.add_argument("--backbone", type=str, default="../saved_models/vgg_backbone.pt")
    parser.add_argument("--classifier", type=str, default="../saved_models/vgg_classifier.pt")
    parser.add_argument("--train_dataset", type=str, default="../datasets/high_res_butterfly_data_train.txt")
    parser.add_argument("--test_image", type=str, default="/local/scratch/datasets/high_res_butterfly_data_test/aglaope_M/10428111_V_aglaope_M.png")
    parser.add_argument("--view", type=str, default="V", choices=["V", "D"])
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--top_features", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta_step", type=int, default=100)
    parser.add_argument("--img_save_step", type=int, default=100)
    parser.add_argument("--reg_beta", type=float, default=2.0)
    parser.add_argument("--reg_lambda", type=float, default=0.000)
    parser.add_argument("--reg_original", type=float, default=0.000)
    parser.add_argument("--out_bounds_lambda", type=float, default=10000)
    parser.add_argument("--test_lbl", type=int, default=0)
    parser.add_argument("--target_lbl", type=int, default=20)
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--img_folder", type=str, default="imgs")
    parser.add_argument("--exp_name", type=str, default="debug")


    args = parser.parse_args()
    args.gpus = ",".join(map(lambda x: str(x), args.gpus))
    return args

def setup(args):
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus



def nearest_neighbor(feat, original_features, target_features, args):
    source_diff = np.sqrt(((original_features - feat) ** 2).sum(1))
    target_diff = np.sqrt(((target_features - feat) ** 2).sum(1))
    source_min = source_diff.min()
    target_min = target_diff.min()
    if source_min < target_min:
        return args.test_lbl, source_min, target_min
    else:
        return args.target_lbl, target_min, source_min

def get_out_of_bounds_loss(z):
    upper = NORMALIZE(torch.ones_like(z)).cuda()
    lower = NORMALIZE(torch.zeros_like(z)).cuda()
    out_loss = ((z[z > upper] - upper[z > upper]) ** 2).sum()
    out_loss += ((lower[z < lower] - z[z < lower]) ** 2).sum()
    return out_loss

if __name__ == "__main__":
    args = get_args()
    setup(args)
    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)

    # Save Args
    logger.save_json(args.__dict__, "args.json")

    train_dset = ImageList(args.train_dataset, transform=image_transform())
    train_dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    backbone = None
    if args.net == "resnet":
        backbone = Res50(pretrain=True).cuda()
        if args.backbone is not None:
            backbone.load_state_dict(torch.load(args.backbone))
    elif args.net == "vgg":
        backbone = VGG16(pretrain=True).cuda()
        if args.backbone is not None:
            backbone.load_state_dict(torch.load(args.backbone))

    classifier = Classifier(backbone.in_features, train_dset.get_num_classes()).cuda()
    classifier.load_state_dict(torch.load(args.classifier))
    backbone.eval()
    classifier.eval()
    original_activations = []
    target_activations = []
    original_features = []
    target_features = []
    test_img = to_tensor(rgb_img_loader(args.test_image))
    test_img_pil = transforms.ToPILImage()(test_img)
    test_img_input = test_image_transform()(test_img).cuda()
    with torch.no_grad():
        test_features, test_z = backbone(test_img_input.unsqueeze(0), compute_z=True)
        test_features = test_features[0].detach().cpu().numpy()
        test_z = test_z[0]
        for imgs, lbls, paths in tqdm(train_dataloader, desc="Computing Features", position=1, ncols=50, leave=False):
            features, activations = backbone(imgs.cuda(), compute_z=True)
            for i, lbl in enumerate(lbls):
                current_view = paths[i].split(os.path.sep)[-1].split("_")[1]
                if int(lbl.item()) == args.test_lbl and current_view == args.view:
                    feat = features[i].detach().cpu().numpy()
                    score = cosine_similarity(feat, test_features)
                    original_activations.append([paths[i], activations[i].detach().cpu().numpy(), score])
                    original_features.append(feat)
                elif int(lbl.item()) == args.target_lbl and current_view == args.view:
                    feat = features[i].detach().cpu().numpy()
                    score = cosine_similarity(feat, test_features)
                    target_activations.append([paths[i], activations[i].detach().cpu().numpy(), score])
                    target_features.append(feat)
        
    # NOTE, we do not remove from the features arrays (target_features, original_features)
    original_features = np.array(original_features)
    target_features = np.array(target_features)
    original_activations.sort(key=lambda x: x[2], reverse=True)
    target_activations.sort(key=lambda x: x[2], reverse=True)
    original_activations = original_activations[:args.K]
    target_activations = target_activations[:args.K]
    avg_original_activations = np.zeros_like(original_activations[0][1])
    avg_target_activations = np.zeros_like(target_activations[0][1])
    for act in original_activations:
        avg_original_activations += act[1]
    for act in target_activations:
        avg_target_activations += act[1]

    logger.log(f"Size of source set: {len(original_activations)}")
    logger.log(f"Size of target set: {len(target_activations)}")
    avg_original_activations = avg_original_activations / len(original_activations)
    avg_target_activations = avg_target_activations / len(target_activations)

    attribute_vector = torch.tensor(avg_target_activations - avg_original_activations).cuda()
    att_size = attribute_vector.shape[0]
    logger.log(f"Number of features in attribute vector: {att_size}")
    print(f"Number of features in attribute vector: {att_size}")
    print(f"Number greater than 0.0001: {(attribute_vector > 0.0001).sum()}")
    print(f"Number greater than 0.1: {(attribute_vector > 0.1).sum()}")
    #Y = []
    #sorted_vals = np.sort(attribute_vector.detach().cpu().numpy())[::-1]
    #for val in sorted_vals:
    #    if val < 0.0001: break
    #    Y.append(val)
    #print("Filtered Values")
    #fig = plt.figure(figsize=(16, 9))
    #plt.bar(range(len(Y[:10000])), Y[:10000])
    #plt.savefig(os.path.join(logger.get_save_dir(), f"attribute_vector.png"))
    #plt.close()

    if args.top_features is not None:
        val = np.sort(attribute_vector.detach().cpu().numpy()**2)[::-1][args.top_features]
        attribute_vector[(attribute_vector**2) >= val] = 0.0
        val = np.sort(attribute_vector.detach().cpu().numpy()**2)[::-1][args.top_features+2000]
        attribute_vector[(attribute_vector**2) < val] = 0.0
        att_size = ((attribute_vector**2) != 0.0).sum()
    logger.log(f"Number of filtered features in attribute vector: {att_size}")
    


    alpha = args.beta * att_size / (attribute_vector ** 2).sum()
    alpha = alpha.item()
    if args.alpha is not None:
        alpha = args.alpha
    #alpha = args.beta * attribute_vector.shape[0] / (attribute_vector ** 2).sum()
    #if args.alpha is not None:
    #    alpha = args.alpha
    #logger.log(f"Alpha: {alpha}")


    z = nn.Parameter(test_img_input.detach().clone().cuda())
    optimizer = LBFGS([z], lr=args.lr)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False

    reverse_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    reverse_size = transforms.Resize((test_img.shape[1], test_img.shape[2]))
    #cur_beta = args.beta / args.beta_step
    save_dir = os.path.join(logger.get_save_dir(), args.img_folder)
    os.makedirs(save_dir)
    logger.log(f"Target Alpha {alpha}")
    loss_fn = TransformLoss(test_img_input, test_z + attribute_vector*alpha, beta=args.reg_beta, reg_lambda=args.reg_lambda, reg_original=args.reg_original)
    loss_v = args.min_loss + 1
    i = 0
    target_lbl_cuda = torch.tensor(args.target_lbl).unsqueeze(0).cuda()
    while loss_v > args.min_loss and i < args.max_iters:
        i += 1
        #z = reset_inbounds(z)
        #z_input = test_image_transform()(z)
        feats, z_acts = backbone(z.unsqueeze(0), compute_z=True)
        out = classifier(feats)
        #target_loss = nn.CrossEntropyLoss()(out, target_lbl_cuda)
        #reverse_z = reverse_norm(z)
        #out_loss = ((reverse_z > 1).sum() + (reverse_z < 0).sum()) * 1000
        out_loss = get_out_of_bounds_loss(z) * args.out_bounds_lambda
        transform_loss, smooth_loss, change_loss = loss_fn(z, z_acts[0])
        loss = transform_loss + smooth_loss + change_loss + out_loss# + target_loss
        loss_v = loss.item()
        print(f"({i}) Loss: {loss_v}")
        print(f"({i}) Transform Loss: {transform_loss.item()}")
        print(f"({i}) Smooth Loss: {smooth_loss.item()}")
        print(f"({i}) Change Loss: {change_loss.item()}")
        print(f"({i}) Out of Bounds Loss: {out_loss.item()}")
        #print(f"({i}) Target Loss: {target_loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: loss)
        #optimizer.step()

        sm = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            feat, act = backbone(z.unsqueeze(0), compute_z=True)
        act = act[0]
        v = act - test_z
        u = attribute_vector
        angle_diff = torch.dot(v, u) / (torch.linalg.vector_norm(u)*torch.linalg.vector_norm(v))
        angle_diff = torch.acos(angle_diff) * 180/np.pi
        proj_vec = (torch.dot(v, u) / torch.linalg.vector_norm(u)**2) * u
        calc_alpha = torch.linalg.vector_norm(proj_vec) / torch.linalg.vector_norm(u)
        calc_alpha = calc_alpha.item()
        vec_dist = torch.linalg.norm(act - (test_z + attribute_vector*alpha))
        lbl, dist, other_dist = nearest_neighbor(feat.detach().cpu().numpy(), original_features, target_features, args)
        out = classifier(feat)
        target_conf = sm(out)[0][args.target_lbl].item()
        print(target_conf)
        print(sm(out)[0][args.test_lbl].item())
        print(f"Nearest Neighbor Class: {lbl}")
        print(f"Nearest Neighbor Distance: {dist}")
        print(f"Nearest Neighbor (other class) Distance: {other_dist}")
        print(f"Projected Alpha: {calc_alpha}")
        print(f"Distance to target: {vec_dist}")
        print(f"Angle to target: {angle_diff}")

        if i % args.img_save_step == 0 or loss_v <= args.min_loss or i >= args.max_iters:
            fig = plt.figure(figsize=(12, 8))

            # Source image
            source_img = reverse_size(transforms.ToPILImage()(reverse_norm(test_img_input)))
            fig.add_subplot(2, 2, 1)
            plt.imshow(source_img)
            plt.axis('off')
            plt.title(f"Source Image: {args.test_image.split(os.path.sep)[-1]}")

            # Transform image
            fig.add_subplot(2, 2, 2)
            reverse_z = reverse_norm(z)
            #reverse_z[reverse_z < 0] = 0
            #reverse_z[reverse_z > 1] = 0
            transformed_img = reverse_size(transforms.ToPILImage()(reverse_z))
            plt.imshow(transformed_img)
            plt.axis('off')
            plt.title(f"Transformed Image {round(target_conf, 4)*100}% ({round(calc_alpha, 4)}/{round(alpha, 4)})")

            # Transform image
            trans_from_source_img = (to_grayscale(reverse_z.cpu()) - to_grayscale(reverse_norm(test_img_input).cpu()))
            trans_from_source_img_inter = trans_from_source_img
            #trans_from_source_img[trans_from_source_img < 0] = 0.0
            #trans_from_source_img[trans_from_source_img > 1] = 1.0
            trans_from_source_img = reverse_size(transforms.ToPILImage()(trans_from_source_img))
            fig.add_subplot(2, 2, 3)
            plt.imshow(trans_from_source_img, cmap=plt.get_cmap("bwr"))
            plt.axis('off')
            plt.title("Transformed Image - Test Image")

            # Transform image
            source_from_trans_img =  torch.sqrt(((reverse_z.cpu() - reverse_norm(test_img_input).cpu()) ** 2).sum(0))
            source_from_trans_img = source_from_trans_img * trans_from_source_img_inter
            #source_from_trans_img = source_from_trans_img - source_from_trans_img.min()
            #source_from_trans_img = source_from_trans_img / source_from_trans_img.max()
            source_from_trans_img -= source_from_trans_img.min()
            source_from_trans_img /= source_from_trans_img.max()
            fig.add_subplot(2, 2, 4)
            #norm = colors.Normalize(vmin=-norm_val, vmax=norm_val)
            source_from_trans_img = reverse_size(transforms.ToPILImage()(source_from_trans_img))
            plt.imshow(source_from_trans_img, cmap=plt.get_cmap("PiYG"))
            plt.axis('off')
            plt.title("Transformed Image - Test Image (Euclidean & Normalized)")

            plt.savefig(os.path.join(save_dir, f"{i}_transform_results.png"))
            plt.savefig(os.path.join(logger.get_save_dir(), f"transform_results.png"))
            plt.close()

            fig = plt.figure()
            plt.imshow(source_img)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{i}_source_image.png"))
            plt.close()

            fig = plt.figure()
            plt.imshow(transformed_img)
            plt.title(f"Target Confidence: {round(target_conf, 4)*100}% | Projected Alpha: {round(calc_alpha, 4)} | Target Alpha: {round(alpha, 4)}")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{i}_transformed.png"))
            plt.close()

            fig = plt.figure()
            plt.imshow(trans_from_source_img)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{i}_grayscale_diff.png"))
            plt.close()

            fig = plt.figure()
            plt.imshow(source_from_trans_img)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{i}_euclidean_diff.png"))
            plt.close()

        logger.log(f"Projected Alpha: {calc_alpha}")
        logger.log(f"{round(sm(out)[0][args.target_lbl].item(), 4)*100}% confidence on target label")
        logger.log(f"{round(sm(out)[0][args.test_lbl].item(), 4)*100}% confidence on source label")
        logger.log(f"Nearest Neighbor Class: {lbl}")
        logger.log(f"Nearest Neighbor Distance: {dist}")
        logger.log(f"Nearest Neighbor (other class) Distance: {other_dist}")
        logger.log(f"Distance to target: {vec_dist}")
        logger.log(f"Angle to target: {angle_diff}")


        
            
        

