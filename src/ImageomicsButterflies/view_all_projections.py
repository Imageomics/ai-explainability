import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXT_HEIGHT = 20
PADDING = 2

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="/research/nfs_chao_209/david/img_to_img/autoencoder_default_w/")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    img_fnt = ImageFont.load_default()

    final_img = None
    for root, dirs, files in os.walk(args.exp_dir):
        if "projections.npz" not in files:
            continue
        subspecies = root.split(os.path.sep)[-1]
        projections = np.transpose(np.load(os.path.join(root, "projections.npz"))['projections'][-1], axes=[0, 2, 3, 1])
        originals = np.transpose(np.load(os.path.join(root, "originals.npz"))['originals'], axes=[0, 2, 3, 1])
        all_subspecies = None
        for proj, org in zip(projections, originals):
            combined = np.concatenate([org, proj], axis=1)
            if all_subspecies is None:
                all_subspecies = combined
            else:
                all_subspecies = np.concatenate([all_subspecies, combined], axis=0)

        if len(projections) < 8:
            for i in range(8 - len(projections)):
                zeros = np.zeros((org.shape[0], all_subspecies.shape[1], 3))
                all_subspecies = np.concatenate([all_subspecies, zeros], axis=0)

        all_subspecies = (all_subspecies * 255).astype(np.uint8)


        org_txt_img = Image.new(mode="RGB", size=(org.shape[1], TEXT_HEIGHT), color="white")
        draw = ImageDraw.Draw(org_txt_img)
        draw.text((PADDING, PADDING), "Original", font=img_fnt, fill=(0, 0, 0))

        proj_txt_img = Image.new(mode="RGB", size=(org.shape[1], TEXT_HEIGHT), color="white")
        draw = ImageDraw.Draw(proj_txt_img)
        draw.text((PADDING, PADDING), "Projection", font=img_fnt, fill=(0, 0, 0))

        species_txt_img = Image.new(mode="RGB", size=(org.shape[1]*2, TEXT_HEIGHT), color="white")
        draw = ImageDraw.Draw(species_txt_img)
        draw.text((PADDING, PADDING), subspecies, font=img_fnt, fill=(0, 0, 0))

        text_combine = np.concatenate([np.asarray(org_txt_img), np.asarray(proj_txt_img)], axis=1)

        complete_subspecies_img = np.concatenate([np.asarray(species_txt_img), text_combine, all_subspecies], axis=0)
        
        if final_img is None:
            final_img = complete_subspecies_img
        else:
            final_img = np.concatenate([final_img, complete_subspecies_img], axis=1)

        print(f"Rendered {subspecies}")

    Image.fromarray(final_img).save("test.png")
    Image.fromarray(final_img).save(os.path.join(args.exp_dir, "all_projections.png"))