import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np

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
    parser.add_argument('--backbone', help='feature weights', type=str, default="../saved_models/vgg_backbone_nohybrid_D.pt")
    parser.add_argument('--classifier', help='classifier weights', type=str, default="../saved_models/vgg_classifier_nohybrid_D.pt")
    parser.add_argument('--outdir_root', type=str, default="/research/nfs_chao_209/david/")
    parser.add_argument('--experiments_path', type=str, default="../experiments/class_fooling.json")
    parser.add_argument('--mimic_pairs_path', type=str, default="../experiments/mimic_pairs_filtered.json")
    parser.add_argument('--dataset_root', type=str, default="../datasets/high_res_butterfly_data_test/")

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
    target_species = []
    projection_labels = []
    for mimic_pair in mimic_pairs:
        erato_path = os.path.join(args.dataset_root, f"{mimic_pair['erato']['name']}_E")
        assert os.path.exists(erato_path), f"{erato_path} does not exists"
        melpomene_path = os.path.join(args.dataset_root, f"{mimic_pair['melpomene']['name']}_M")
        assert os.path.exists(melpomene_path), f"{melpomene_path} does not exists"

        #! Notice we swap the labels for fooling the classifier
        image_paths.append(erato_path)
        target_species.append(melpomene_path.split(os.path.sep)[-1])
        projection_labels.append(mimic_pair['melpomene']['label'])

        image_paths.append(melpomene_path)
        target_species.append(erato_path.split(os.path.sep)[-1])
        projection_labels.append(mimic_pair['erato']['label'])

    # Load Models
    G, D, F, C = load_models(args.network, f_path=args.backbone, c_path=args.classifier)

    results_dir = os.path.join(args.outdir_root, "class_fooling")
    os.makedirs(results_dir, exist_ok=args.overwrite)
    #save_json(args.__dict__, os.path.join(results_dir, "args.json"))
    for exp_i, exp in enumerate(experiments):
        exp_start_time = perf_counter()
        latent_path = exp["latent_start"]
        learn_param = exp["learn_param"]
        smooth = exp["smooth"]
        lr = exp["lr"]
        num_steps = exp["num_steps"]
        use_batch = exp["batch"]
        use_entropy = exp["entropy"]
        no_regularizer = exp["no_regularizer"]
        superpixel = exp["superpixel"]
        exp_name = latent_path.split(os.path.sep)[-1]
        exp_name += "_" + learn_param
        if use_batch:
            exp_name += "_batch"
        if use_entropy:
            exp_name += "_ent"
        if no_regularizer:
            exp_name += "_no_reg"
        if smooth:
            exp_name += "_smooth"
        if superpixel:
            exp_name += "_superpixel"

        outdir = os.path.join(results_dir, exp_name)
        os.makedirs(outdir, exist_ok=args.overwrite)
        save_json(exp, os.path.join(outdir, f"exp_args.json"))

        for species_i, (img_path, proj_lbl, tgt_species) in enumerate(zip(image_paths, projection_labels, target_species)):
            subspecies = img_path.split(os.path.sep)[-1]
            butterfly_outdir = os.path.join(outdir, f"{subspecies}_to_{tgt_species}")
            os.makedirs(butterfly_outdir, exist_ok=args.overwrite)
            latents = os.path.join(latent_path, subspecies, "latents.npz")
            start_zs, start_ws = load_latents(G, latents)

            w_out, z_out, all_synth_images, _, _, image_confs, min_losses = project(
                None,
                G,
                D,
                F,
                C,
                proj_lbl,
                learn_param                = learn_param,
                start_zs                   = start_zs,
                start_ws                   = start_ws,
                num_steps                  = num_steps,
                init_lr                    = lr,
                img_to_img                 = False,
                batch                      = use_batch,
                use_entropy                = use_entropy,
                verbose                    = args.verbose,
                use_default_feat_extractor = False,
                no_regularizer             = no_regularizer,
                smooth_change              = smooth,
                use_superpixel             = superpixel
            )

            # Save Data
            if z_out is None:
                np.savez(f'{butterfly_outdir}/latents.npz', w=w_out[-1].cpu().numpy())
                np.savez(f'{butterfly_outdir}/all_steps_latents.npz', w=w_out.cpu().numpy())
            else:
                np.savez(f'{butterfly_outdir}/latents.npz', w=w_out[-1].cpu().numpy(), z=z_out[-1].cpu().numpy())
                np.savez(f'{butterfly_outdir}/all_steps_latents.npz', w=w_out.cpu().numpy(), z=z_out.cpu().numpy())
            
            np.savez(f'{butterfly_outdir}/projections.npz', projections=np.array(all_synth_images))
            np.savez(f'{butterfly_outdir}/statistics.npz',
                image_confs=np.array(image_confs),
                min_losses=np.array(min_losses)
            )
            print(f"Exp {exp_i+1}/{len(experiments)} {species_i+1}/{len(image_paths)} {butterfly_outdir.split(os.path.sep)[-1]} | avg_conf: {round(np.array(image_confs)[-1][:, proj_lbl].mean(), 4)}")
        exp_time = f'{(perf_counter()-exp_start_time):.1f} s'
        print(f"Exp {exp_i+1}/{len(experiments)} run time: {exp_time}")
    all_time = f'{(perf_counter()-all_start_time):.1f} s'
    print(f"Total time to run: {all_time}")