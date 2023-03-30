import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import cv2
import matplotlib.colors as mcolors
from PIL import Image

from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from superpixel import superpixel

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--darken_factor", type=float, default=1.5)
    parser.add_argument('--verbose',              help='print all messages?', action="store_true", default=False)
    parser.add_argument('--overwrite',              help='overwrite results', action="store_true", default=False)
    parser.add_argument('--outdir_root', type=str, default="/research/nfs_chao_209/david/")
    parser.add_argument('--experiments_path', type=str, default="../experiments/visualization.json")
    parser.add_argument('--mimic_pairs_path', type=str, default="../experiments/mimic_pairs_filtered.json")
    parser.add_argument('--dataset_root', type=str, default="../datasets/high_res_butterfly_data_test/")

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def sign(x):
    rv = np.zeros_like(x)
    rv[x < 0] = -1
    rv[x > 0] = 1
    return rv

def lightness(x):
    cmax = x.max(3)
    cmin = x.min(3)
    return (cmax - cmin) / 2

def grayscale(x):
    rv = np.zeros_like(x[:3])
    rv = x[:, :, :, 0] * 0.299 + x[:, :, :, 1] * 0.587 + x[:, :, :, 2] * 0.114
    return rv

def hsv(x):
    return mcolors.rgb_to_hsv(x)

def compute_diff(prev, cur, colorspace):
    if colorspace == "lightness":
        cur_L = lightness(cur)
        prev_L = lightness(prev)
        return cur_L - prev_L
    elif colorspace == "grayscale":
        cur_gray = grayscale(cur)
        prev_gray = grayscale(prev)
        return cur_gray - prev_gray
    elif colorspace == "rgb":
        sign_val = sign(lightness(cur) - lightness(prev))
        return sign_val * ((cur - prev) ** 2).mean(3)
    elif colorspace == "hsv":
        sign_val = sign(lightness(cur) - lightness(prev))
        return sign_val * ((hsv(cur) - hsv(prev)) ** 2).mean(3)
    
    assert False, f"Invalid colorspace: {colorspace}"

def median_filter(diff):
    return cv2.medianBlur(diff, 5)

def get_diffs(projections, colorspace, use_median_filter, buckets):
    diffs = []
    prev_projs = None
    for step, projs in enumerate(projections):
        if step == 0:
            prev_projs = projs
            continue

        if step % buckets != 0:
            continue

        diff = compute_diff(prev_projs, projs, colorspace)
        if use_median_filter:
            diff = median_filter(diff)
        diffs.append(diff)
        prev_projs = projs
    
    return np.array(diffs)

def normalize(x):
    img_size = x.shape[1:]
    min_v = np.min(x, axis=(1, 2), keepdims=True)
    min_v = np.tile(min_v, list(img_size))
    x -= min_v
    max_v = np.max(x, axis=(1, 2), keepdims=True)
    max_v = np.tile(max_v, list(img_size))
    x /= max_v
    return x

def save_img(x, path):
    Image.fromarray((x * 255).astype(np.uint8)).save(path)

def superimpose(base, overlay, darken_factor, method="add"):
    blend_img = np.copy(base)
    blend_img[:, :, 0] = base[:, :, 0] / darken_factor 
    if method == "add":
        blend_img[:, :, 1] = base[:, :, 1] / darken_factor * (1-overlay) + overlay * np.ones_like(overlay)
    else:
        blend_img[:, :, 1] = base[:, :, 1] / darken_factor

    if method == "del":
        blend_img[:, :, 2] = base[:, :, 2] / darken_factor * (1-overlay) + overlay * np.ones_like(overlay)
    else:
        blend_img[:, :, 2] = base[:, :, 2] / darken_factor

    
    return blend_img

def superimpose_both(base, add_overlay, del_overlay, darken_factor):
    blend_img = np.copy(base)
    blend_img[:, :, 0] = base[:, :, 0] / darken_factor
    blend_img[:, :, 1] = base[:, :, 1] / darken_factor * (1-add_overlay) + add_overlay * np.ones_like(add_overlay)
    blend_img[:, :, 2] = base[:, :, 2] / darken_factor * (1-del_overlay) + del_overlay * np.ones_like(del_overlay)
    return blend_img

def save_results(add_diffs, del_diffs, projections, darken_factor, final_outdir):
    for d_i, (a_diff, d_diff) in enumerate(zip(add_diffs, del_diffs)):
        save_img(projections[0][d_i], os.path.join(final_outdir, f"start_{d_i}.png"))
        save_img(projections[-1][d_i], os.path.join(final_outdir, f"end_{d_i}.png"))
        save_img(a_diff, os.path.join(final_outdir, f"add_diff_{d_i}.png"))
        save_img(d_diff, os.path.join(final_outdir, f"del_diff_{d_i}.png"))
        save_img(superimpose(projections[0][d_i], a_diff, darken_factor, method="add"), os.path.join(final_outdir, f"start_add_{d_i}.png"))
        save_img(superimpose(projections[-1][d_i], a_diff, darken_factor, method="add"), os.path.join(final_outdir, f"end_add_{d_i}.png"))
        save_img(superimpose(projections[0][d_i], d_diff, darken_factor, method="del"), os.path.join(final_outdir, f"start_del_{d_i}.png"))
        save_img(superimpose(projections[-1][d_i], d_diff, darken_factor, method="del"), os.path.join(final_outdir, f"end_del_{d_i}.png"))
        save_img(superimpose_both(projections[0][d_i], a_diff, d_diff, darken_factor), os.path.join(final_outdir, f"start_all_{d_i}.png"))
        save_img(superimpose_both(projections[-1][d_i], a_diff, d_diff, darken_factor), os.path.join(final_outdir, f"end_all_{d_i}.png"))

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

    pair_data = []
    for mimic_pair in mimic_pairs:
            erato_path = os.path.join(args.dataset_root, f"{mimic_pair['erato']['name']}_E")
            assert os.path.exists(erato_path), f"{erato_path} does not exists"
            melpomene_path = os.path.join(args.dataset_root, f"{mimic_pair['melpomene']['name']}_M")
            assert os.path.exists(melpomene_path), f"{melpomene_path} does not exists"

            #! Notice we swap the labels for fooling the classifier
            melpomene = melpomene_path.split(os.path.sep)[-1]
            melpomene_lbl = mimic_pair['melpomene']['label']

            erato = erato_path.split(os.path.sep)[-1]
            erato_lbl = mimic_pair['erato']['label']

            pair_data.append((f"{melpomene}_to_{erato}", erato_lbl))
            pair_data.append((f"{erato}_to_{melpomene}", melpomene_lbl))

    results_dir = os.path.join(args.outdir_root, "visualization")
    os.makedirs(results_dir, exist_ok=args.overwrite)
    save_json(args.__dict__, os.path.join(results_dir, "args.json"))
    for exp_i, exp in enumerate(experiments):
        exp_start_time = perf_counter()
        projection_path = exp["projection_folder"]
        colorspace = exp["colorspace"]
        confidence_visualize = exp["confidence"]
        use_median_filter = exp["median_filter"]
        buckets = exp["buckets"]
        thresh = exp["thresh"]
        use_superpixel = exp["superpixel"]
        conf_thresh = exp["conf_change_thresh"]
        
        exp_name_root = "/".join(projection_path.split(os.path.sep)[-2:])
        exp_name_root += f"_{colorspace}"
        if confidence_visualize:
            exp_name_root += f"_conf_buckets_{buckets}"
        else:
            exp_name_root += f"_naive"

        if use_superpixel:
            exp_name_root += f"_superpixel"
        
        exp_name_root += f"_conf_thresh_{conf_thresh}"

        exp_name_root += f"_thresh_{thresh}"

        if use_median_filter:
            exp_name_root += f"_med_filter"

        exp_outdir = os.path.join(results_dir, exp_name_root)
        os.makedirs(exp_outdir, exist_ok=args.overwrite)
        save_json(exp, os.path.join(exp_outdir, f"exp_args.json"))

        for cls_fool_i, (cls_fool_path, tgt_lbl) in enumerate(pair_data):
            final_outdir = os.path.join(exp_outdir, cls_fool_path)
            os.makedirs(final_outdir, exist_ok=args.overwrite)
            data_path = os.path.join(projection_path, cls_fool_path)
            projections = np.load(os.path.join(data_path, "projections.npz"))["projections"] # steps x batch x 3 x 128 x 128
            projections = np.transpose(projections, axes=[0, 1, 3, 4, 2])
            if not confidence_visualize:
                diff = compute_diff(projections[0], projections[-1], colorspace)
                if use_median_filter:
                    diff = median_filter(diff)
                add_diff = np.zeros_like(diff)
                add_diff[diff > 0] = diff[diff > 0]
                add_diff = normalize(add_diff)
                add_diff[add_diff < thresh] = 0
                del_diff = np.zeros_like(diff)
                del_diff[diff < 0] = diff[diff < 0] * -1
                del_diff = normalize(del_diff)
                del_diff[del_diff < thresh] = 0

                save_results(add_diff, del_diff, projections, args.darken_factor, final_outdir)

                print(f"Exp {exp_i+1}/{len(experiments)} {cls_fool_i+1}/{len(pair_data)}: {cls_fool_path}")
                continue

            diffs = get_diffs(projections, colorspace, use_median_filter, buckets)
            step = 0
            confs = np.load(os.path.join(data_path, "statistics.npz"))["image_confs"] # steps x batch x # classes
            prev_confs = confs[0, :, tgt_lbl]
            step_i = 0
            conf_adds = np.zeros_like(diffs[0])
            conf_dels = np.zeros_like(diffs[0])
            for step in range(buckets, len(confs), buckets):
                cur_confs = confs[step, :, tgt_lbl]
                conf_diffs = cur_confs - prev_confs
                conf_diffs[np.abs(conf_diffs) < conf_thresh] = 0.0
                conf_adds += np.tile(conf_diffs.reshape(len(conf_diffs), 1, 1), [128, 128]) * diffs[step_i] * (diffs[step_i] > 0).astype(np.float32)
                conf_dels += np.tile(conf_diffs.reshape(len(conf_diffs), 1, 1), [128, 128]) * diffs[step_i] * (diffs[step_i] < 0).astype(np.float32) * -1

            if use_superpixel:
                superpixel_labels = np.zeros(list(projections.shape[1:4]))
                for s_i, s_img in enumerate(projections[0]):
                    superpixel_labels[s_i] = superpixel(s_img)
                add_diff = np.zeros_like(conf_adds)
                del_diff = np.zeros_like(conf_dels)
                for i, (img_add, img_del) in enumerate(zip(conf_adds, conf_dels)):
                    n_lbls = int(superpixel_labels[i].max())+1
                    add_v = []
                    del_v = []
                    for lbl in range(n_lbls):
                        mask = superpixel_labels[i].astype(np.uint8) == lbl
                        add_v.append(np.abs(img_add)[mask].mean())
                        del_v.append(np.abs(img_del)[mask].mean())
                    add_v = np.array(add_v)
                    del_v = np.array(del_v)
                    add_v /= add_v.max()
                    del_v /= del_v.max()
                    for lbl in range(n_lbls):
                        mask = superpixel_labels[i].astype(np.uint8) == lbl
                        add_diff[i][mask] = add_v[lbl] ** 2
                        del_diff[i][mask] = del_v[lbl] ** 2

            else:
                add_diff = normalize(conf_adds)
                add_diff[add_diff < thresh] = 0
                del_diff = normalize(conf_dels)
                del_diff[del_diff < thresh] = 0

            save_results(add_diff, del_diff, projections, args.darken_factor, final_outdir)

            print(f"Exp {exp_i+1}/{len(experiments)} {cls_fool_i+1}/{len(pair_data)}: {cls_fool_path}")
        exp_time = f'{(perf_counter()-exp_start_time):.1f} s'
        print(f"Exp {exp_i+1}/{len(experiments)} run time: {exp_time}")
    all_time = f'{(perf_counter()-all_start_time):.1f} s'
    print(f"Total time to run: {all_time}")