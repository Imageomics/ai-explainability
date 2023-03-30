import os
import torch
from argparse import ArgumentParser
from time import perf_counter
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

from models import Encoder
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from project import project

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--mode', type=str, default='filtered', choices=['filtered', 'original', 'original_nohybrid'])
    parser.add_argument('--sub', type=str, default=None)

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))

    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.dataset_root = '../datasets/train/'
        #args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
    elif args.mode == 'original':
        args.encoder = 'encoder4editing/butterfly_org_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original/00004-stylegan3-r-train_original_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.dataset_root = '../datasets/original/train/'
        #args.dataset_root_test = '../datasets/original/test/'
        args.outdir = '../output/ganspace_reconstruction_original'
    elif args.mode == 'original_nohybrid':
        args.encoder = 'encoder4editing/butterfly_org_no_hybrid_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original_nohybrid/00006-stylegan3-r-train_original_no_hybrid_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.dataset_root = '../datasets/original_nohybrid/train/'
        #args.dataset_root_test = '../datasets/original_nohybrid/test/'
        args.outdir = '../output/ganspace_reconstruction_original_nohybrid'

    args.learned_ws = os.path.join(args.outdir, "train_ws.npz")
    args.paths = os.path.join(args.outdir, "train_paths.json")
    if args.sub:
        sub1, sub2 = args.sub.split("_")

        args.learned_ws = os.path.join(args.outdir, f"train_{sub1}_ws.npz")
        args.paths = os.path.join(args.outdir, f"train_{sub1}_paths.json")
        args.learned_ws2 = os.path.join(args.outdir, f"train_{sub2}_ws.npz")
        args.paths2 = os.path.join(args.outdir, f"train_{sub2}_paths.json")
    args.outdir = os.path.join(args.outdir, "analysis")

    return args

def sample_w_global_mean(samples, G):
    print(f'Computing {args.samples} W samples...')
    z_samples = np.random.RandomState(123).randn(args.samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    w_samples = w_samples[:, 0, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=False)
    return w_avg, w_samples

def pca(data, dim=2):
    data = np.array(data)
    mu = data.mean(0)
    center = data - mu
    cov = (1/len(data))*(center.T @ center)
    eig_val, eig_vec = np.linalg.eig(cov)
    W = eig_vec[:dim]

    return (data - mu) @ W.T, W, mu, eig_vec, eig_val

def tsne(data, dim=2):
    data = np.array(data)
    z = TSNE(dim).fit_transform(data)
    return z



def pca_analysis(args, G):
    print(f'Computing {args.samples} W samples...')
    z_samples = np.random.RandomState(123).randn(args.samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    w_samples = w_samples[:, 0, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=False)      # [1, 1, C]

    w_center = w_samples - w_avg
    w_covariance = (1/args.samples)*(w_center.T @ w_center)

    eig_val, eig_vec = np.linalg.eig(w_covariance)
    eig_val_total = eig_val.sum()

    accum = 0
    for i, val in enumerate(eig_val):
        accum += val
        percent = accum / eig_val_total
        if percent >= 0.99:
            print(f"Top {i+1} eignvectors can recreate {round(percent, 4)*100}% of the data")
            break
    

def visualize_means(data, labels, w_global_mean, outdir, name='pca_means'):
    for mean, lbl in zip(data, labels):
        plt.scatter(mean[0], mean[1], label=lbl)
    plt.scatter(w_global_mean[0], w_global_mean[1], label="global", marker='x')
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()

def visualize(data, labels, outdir, name='pca'):
    for i in range(max(labels)+1):
        dps = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == i, zip(labels, data)))))
        plt.scatter(dps[:, 0], dps[:, 1], label=i)
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()

def create_image(G, w):
    w_input = torch.from_numpy(np.tile(w, (1, G.num_ws, 1))).cuda()
    synth_image = G.synthesis(w_input, noise_mode='const')[0]
    synth_image = (synth_image + 1) * (1/2)
    synth_image = synth_image.clamp(0, 1)
    out_image = (synth_image * 255).detach().cpu().numpy().astype(np.uint8)
    
    return np.transpose(out_image, axes=(1, 2, 0))

def move_along_top_vectors(G, w, w_star, eig_vec, eig_val, outdir, name="base"):
    for i in range(10):
        save_img = None
        for alpha in range(-10, 11, 1):
            w_move = w + eig_vec[i]*alpha
            img = create_image(G, w_move)
            if save_img is None:
                save_img = img
            else:
                save_img = np.concatenate((save_img, img), axis=1)
        Image.fromarray(save_img).save(os.path.join(outdir, f"{name}_w_vec_{i}.png"))    
    
    for i in range(10):
        save_img = None
        for alpha in range(-10, 11, 1):
            w_move = w_star + eig_vec[i]*alpha
            img = create_image(G, w_move)
            if save_img is None:
                save_img = img
            else:
                save_img = np.concatenate((save_img, img), axis=1)
        Image.fromarray(save_img).save(os.path.join(outdir, f"{name}_w_star_vec_{i}.png"))    

def move_along_class_vectors(G, w, w_star, class_vectors, star_class_vectors, names, outdir):
    for i, class_vec in enumerate(class_vectors):
        save_img = None
        save_img_star = None
        for alpha in range(-10, 11, 1):
            if alpha == 0:
                move = 0
            else:
                move = alpha / 10
            w_move = w + class_vectors[i]*move
            w_star_move = w_star + star_class_vectors[i]*move
            img = create_image(G, w_move)
            img_star = create_image(G, w_star_move)
            if save_img is None:
                save_img = img
            else:
                save_img = np.concatenate((save_img, img), axis=1)
            if save_img_star is None:
                save_img_star = img_star
            else:
                save_img_star = np.concatenate((save_img_star, img_star), axis=1)
        Image.fromarray(save_img).save(os.path.join(outdir, f"class_w_{names[i]}.png"))    
        Image.fromarray(save_img_star).save(os.path.join(outdir, f"class_w_star_{names[i]}.png"))    

def move_between_class_means(G, class_means, names, outdir):
    for name_A, mean_A in zip(names, class_means):
        for name_B, mean_B in zip(names, class_means):
            if name_A == name_B: continue
            vec = mean_B - mean_A

            save_img = None
            for alpha in range(0, 101, 1):
                if alpha == 0:
                    move = 0
                else:
                    move = alpha / 100
                w_move = mean_A + vec*move
                img = create_image(G, w_move)
                if save_img is None:
                    save_img = img
                else:
                    save_img = np.concatenate((save_img, img), axis=1)
            Image.fromarray(save_img).save(os.path.join(outdir, f"{name_A}_to_{name_B}.png"))


def get_class_vectors(class_means, w, w_star):
    class_vectors = class_means - w
    star_class_vectors = class_means - w_star
    return class_vectors, star_class_vectors

def center_classes(latents, labels, class_labels, class_vectors, star_class_vectors):
    class_centered_ws = []
    star_class_centered_ws = []
    vector_map = {}
    for lbl, vec, star_vec in zip(class_labels, class_vectors, star_class_vectors):
        vector_map[lbl] = [vec, star_vec]

    for w, lbl in zip(latents, labels):
        class_centered_ws.append(w - vector_map[lbl][0])
        star_class_centered_ws.append(w - vector_map[lbl][0])

    return class_centered_ws, star_class_centered_ws


if __name__ == "__main__":
    # Time
    all_start_time = perf_counter()

    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    #TODO

    # Load autoencoder
    #encoder = Encoder(size=128).cuda()
    #encoder.eval()
    #encoder.load_state_dict(torch.load(args.encoder))

    latents = []
    labels = []
    subspecies = []
    unique_subspecies = set()
    sub = None
    tmp_i = 0
    for root, dirs, files in os.walk(args.dataset_root):
        for f in files:
            sub = root.split(os.path.sep)[-1]
            unique_subspecies.add(sub)
            subspecies.append(sub)
            path = os.path.join(root, f)
            if args.learned_ws is None:
                imgs = load_imgs(path, view="D")
                #with torch.no_grad():
                #    start_ws, _ = encoder(imgs)
                #    start_ws = start_ws.view(len(imgs), 12, -1).mean(1)
                #latents.append(start_ws[0].detach().cpu().numpy())
        print(sub)
        tmp_i += 1
        #if tmp_i > 2: break
    unique_subspecies = sorted(list(unique_subspecies))
    subspecies_to_lbl = {}
    for i, sub in enumerate(unique_subspecies):
        subspecies_to_lbl[sub] = i

    for sub in subspecies:
        labels.append(subspecies_to_lbl[sub])

    if args.learned_ws is not None:
        latents = np.load(args.learned_ws, allow_pickle=True)['ws']
        labels = []
        paths = load_json(args.paths)
        for path in paths:
            sub = subspecies_to_lbl[path.split(os.path.sep)[-2]]
            labels.append(sub)
        if args.sub:
            paths2 = load_json(args.paths2)
            paths.extend(paths2)
            latents2 = np.load(args.learned_ws2, allow_pickle=True)['ws']
            latents = np.concatenate((latents, latents2), axis=0)
            for path in paths2:
                sub = subspecies_to_lbl[path.split(os.path.sep)[-2]]
                labels.append(sub)


    w_star_global_mean = np.array(latents).mean(0)

    # Calculate class-means
    class_means = []
    class_lbls = []
    for i, sub in enumerate(unique_subspecies):
        if args.sub and sub not in args.sub.split("_"): continue
        filtered_data = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == i, zip(labels, latents)))))
        class_means.append(filtered_data.mean(0))
        class_lbls.append(i)

    z, W, mu, eig_vec, eig_val = pca(latents)
    #z = tsne(latents)
    if not args.sub:
        visualize(z, labels, args.outdir, name='pca')

    encoder = None

    # Load Models
    G, _, _, _ = load_models(args.network, f_path=None, c_path=None)
    G = G.cuda()
    

    w_global_mean, sample_ws = sample_w_global_mean(args.samples, G)

    global_mean_img = create_image(G, w_global_mean)
    Image.fromarray(global_mean_img).save(os.path.join(args.outdir, "w_global_mean_img.png"))
    star_global_mean_img = create_image(G, w_star_global_mean)
    Image.fromarray(star_global_mean_img).save(os.path.join(args.outdir, "w_star_global_mean_img.png"))

    # If we want to move along directions from ganspace
    if False:
        _, _, _, eig_vec, eig_val = pca(sample_ws)
    move_along_top_vectors(G, w_global_mean, w_star_global_mean, eig_vec, eig_val, args.outdir)

    z = (np.array(class_means) - mu) @ W.T
    visualize_means(z, class_lbls, (w_global_mean - mu) @ W.T, args.outdir, name='pca_means')

    class_vectors, star_class_vectors = get_class_vectors(class_means, w_global_mean, w_star_global_mean)
    move_along_class_vectors(G, w_global_mean, w_star_global_mean, class_vectors, star_class_vectors, unique_subspecies, args.outdir)
    move_between_class_means(G, class_means, unique_subspecies, args.outdir)

    class_centered_ws, star_class_centered_ws = center_classes(latents, labels, class_lbls, class_vectors, star_class_vectors)

    z, W, mu, eig_vec, eig_val = pca(class_centered_ws)
    move_along_top_vectors(G, w_global_mean, w_star_global_mean, eig_vec, eig_val, args.outdir, "w_class_centered")
    z, W, mu, eig_vec, eig_val = pca(star_class_centered_ws)
    move_along_top_vectors(G, w_global_mean, w_star_global_mean, eig_vec, eig_val, args.outdir, "w_star_class_centered")
    

    if True:
        pca_analysis(args, G)




    