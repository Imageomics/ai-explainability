import os
import torch
from argparse import ArgumentParser
from time import perf_counter

import numpy as np

from models import Encoder
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from project import project

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--network', type=str, help='Network pickle filename', default="styleGAN/butterfly_training-runs/00000-butterfly128x128-auto4-kimg5000_nohybrid/network-snapshot-005000.pkl")
    parser.add_argument('--verbose',              help='print all messages?', action="store_true", default=False)
    parser.add_argument('--overwrite',              help='overwrite results', action="store_true", default=False)
    parser.add_argument('--backbone', help='feature weights', type=str, default="../saved_models/vgg_backbone_nohybrid_D_norm.pt")
    parser.add_argument('--classifier', help='classifier weights', type=str, default="../saved_models/vgg_classifier_nohybrid_D_norm.pt")
    parser.add_argument('--encoder', help='encoder weights', type=str, default="../saved_models/encoder.pt")
    parser.add_argument('--outdir_root', type=str, default="/research/nfs_chao_209/david/")
    parser.add_argument('--experiments_path', type=str, default="../experiments/img_to_img.json")
    parser.add_argument('--mimic_pairs_path', type=str, default="../experiments/mimic_pairs_filtered.json")
    parser.add_argument('--dataset_root', type=str, default="../datasets/high_res_butterfly_data_test_norm/")

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

if __name__ == "__main__":
    # Time
    all_start_time = perf_counter()

    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)

    # Load data
    experiments = load_json(args.experiments_path)
    mimic_pairs = load_json(args.mimic_pairs_path)
    image_paths = []
    image_labels = []
    for mimic_pair in mimic_pairs:
        erato_path = os.path.join(args.dataset_root, f"{mimic_pair['erato']['name']}_E")
        assert os.path.exists(erato_path), f"{erato_path} does not exists"
        melpomene_path = os.path.join(args.dataset_root, f"{mimic_pair['melpomene']['name']}_M")
        assert os.path.exists(melpomene_path), f"{melpomene_path} does not exists"

        if erato_path not in image_paths:
            image_paths.append(erato_path)
            image_labels.append(mimic_pair['erato']['label'])

        if melpomene_path not in image_paths:
            image_paths.append(melpomene_path)
            image_labels.append(mimic_pair['melpomene']['label'])

    # Load Models
    G, D, F, C = load_models(args.network, f_path=args.backbone, c_path=args.classifier)

    # Load autoencoder

    encoder = Encoder(size=128).cuda()
    encoder.load_state_dict(torch.load(args.encoder))


    results_dir = os.path.join(args.outdir_root, "img_to_img")
    os.makedirs(results_dir, exist_ok=args.overwrite)
    #save_json(args.__dict__, os.path.join(results_dir, "args.json"))
    for exp_i, exp in enumerate(experiments):
        exp_start_time = perf_counter()
        latent_path = exp["latent_start"]
        use_default_feat_extractor = exp["perceptual_default"]
        learn_param = exp["learn_param"]
        lr = exp["lr"]
        num_steps = exp["num_steps"]
        exp_name = f"{'random' if latent_path is None else 'autoencoder'}"
        exp_name += f"_{'default' if use_default_feat_extractor else 'butterfly'}"
        exp_name += "_" + learn_param
        exp_name += f"_{num_steps}"
        outdir = os.path.join(results_dir, exp_name)
        os.makedirs(outdir, exist_ok=args.overwrite)
        save_json(exp, os.path.join(outdir, f"exp_args.json"))
            
        if latent_path is None:
            zs, ws = load_latents(G, None)

        for species_i, (img_path, img_lbl) in enumerate(zip(image_paths, image_labels)):
            subspecies = img_path.split(os.path.sep)[-1]
            butterfly_outdir = os.path.join(outdir, subspecies)
            os.makedirs(butterfly_outdir, exist_ok=args.overwrite)
            images = load_imgs(img_path, view="D")
            if len(images) > 8:
                images = images[:8]
            if latent_path == "autoencoder":
                start_zs = None
                start_ws, _ = encoder(images)
                start_ws = start_ws.view(len(images), 12, -1).mean(1)
            else:
                if learn_param == "w":
                    start_ws = ws.repeat([len(images), 1]).clone()
                    start_zs = None
                elif learn_param == "z":
                    start_zs = zs.repeat([len(images), 1]).clone()
                    start_ws = ws.repeat([len(images), 1]).clone()
            w_out, z_out, all_synth_images, pixel_losses, perceptual_losses, image_confs, _ = project(
                images,
                G,
                D,
                F,
                C,
                img_lbl,
                learn_param                = learn_param,
                start_zs                   = start_zs,
                start_ws                   = start_ws,
                num_steps                  = num_steps,
                init_lr                    = lr,
                img_to_img                 = True,
                batch                      = True,
                verbose                    = args.verbose,
                use_default_feat_extractor = use_default_feat_extractor
            )

            # Save Data
            if z_out is None:
                np.savez(f'{butterfly_outdir}/latents.npz', w=w_out[-1].cpu().numpy())
                np.savez(f'{butterfly_outdir}/all_steps_latents.npz', w=w_out.cpu().numpy())
            else:
                np.savez(f'{butterfly_outdir}/latents.npz', w=w_out[-1].cpu().numpy(), z=z_out[-1].cpu().numpy())
                np.savez(f'{butterfly_outdir}/all_steps_latents.npz', w=w_out.cpu().numpy(), z=z_out.cpu().numpy())
            
            np.savez(f'{butterfly_outdir}/originals.npz', originals=images.detach().cpu().numpy())
            np.savez(f'{butterfly_outdir}/projections.npz', projections=np.array(all_synth_images))
            np.savez(f'{butterfly_outdir}/statistics.npz', 
                pixel_losses=np.array(pixel_losses),
                perceptual_losses=np.array(perceptual_losses),
                image_confs=np.array(image_confs) 
            )
            print(f"Exp {exp_i+1}/{len(experiments)} {species_i+1}/{len(image_paths)} {butterfly_outdir.split(os.path.sep)[-1]} | pixel loss: {round(pixel_losses[-1], 4)}")
        exp_time = f'{(perf_counter()-exp_start_time):.1f} s'
        print(f"Exp {exp_i+1}/{len(experiments)} run time: {exp_time}")
    all_time = f'{(perf_counter()-all_start_time):.1f} s'
    print(f"Total time to run: {all_time}")