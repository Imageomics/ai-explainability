import os

from argparse import ArgumentParser
from logger import Logger

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--iin_ae", type=str, default="output/debug/iin_ae.pt")
    parser.add_argument("--img_classifier", type=str, default="output/debug/img_classifier.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_sample", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--num_features", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default="all_cf")
    return parser.parse_args()

def get_lbl_pairs(args):
    lbl_pairs = []
    for i in range(10):
        for j in range(10):
            if i == j: continue
            lbl_pairs.append([i, j])
    return lbl_pairs

if __name__ == "__main__":
    args = get_args()
    logger = Logger(output_dir="output", exp_name=args.exp_name)
    lbl_pairs = get_lbl_pairs(args)

    for src_lbl, tgt_lbl in lbl_pairs:
        logger.log(f"Running experiment for {src_lbl} => {tgt_lbl}")
        cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python visual_counter_factual_iin_ae.py' + f' --iin_ae {args.iin_ae}' \
               + f' --img_classifier {args.img_classifier}' + f' --src_lbl {src_lbl}' \
               + f' --tgt_lbl {tgt_lbl}' + f' --num_iters {args.num_iters}' \
               + f' --output_dir {logger.get_path()}' + f' --exp_name {src_lbl}_to_{tgt_lbl}' \
               + ' --force_disentanglement' + f' --num_features {args.num_features}'
        
        if args.no_sample:
            cmd += ' --no_sample'
        
        logger.log(cmd)
        output = os.popen(cmd)
        for line in output.readlines():
            logger.log(line)

    logger.log("Program complete")
