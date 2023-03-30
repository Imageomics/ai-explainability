from argparse import ArgumentParser

from helpers import set_random_seed, cuda_setup
from loading_helpers import load_imgs, load_latents, load_models
from project import project

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--network', type=str, help='Network pickle filename', default="styleGAN/butterfly_training-runs/00000-butterfly128x128-auto4-kimg5000_nohybrid/network-snapshot-005000.pkl")
    parser.add_argument('--target', type=str, help='Target directory of image files to project to', default="../datasets/high_res_butterfly_data_test/aglaope_M/")
    parser.add_argument('--buckets',              help='Interval of steps to take diff images', type=int, default=1000)
    parser.add_argument('--num-steps',              help='Number of optimization steps', type=int, default=100)
    parser.add_argument('--batch_size',              help='batch size', type=int, default=2)
    parser.add_argument('--outdir',                 help='Where to save the output images', type=str)
    parser.add_argument('--smooth_beta',            help='Smooth beta', type=float, default=2)
    parser.add_argument('--smooth_eps',             help='Smooth eps', type=float, default=1e-3)
    parser.add_argument('--smooth_lambda',          help='Smooth lambda', type=float, default=0.0)
    parser.add_argument('--mse_lambda',             help='MSE lambda', type=float, default=0.0)
    parser.add_argument('--lr',             help='learning rate', type=float, default=0.001)
    parser.add_argument('--img_to_img',              help='img to img', action="store_true", default=False)
    parser.add_argument('--batch',              help='learn direction vector for every example', action="store_true", default=False)
    parser.add_argument('--use_entropy',              help='Use entropy for classifier fooling', action="store_true", default=False)
    parser.add_argument('--use_rand_latents',              help='Use random latents', action="store_true", default=False)
    parser.add_argument('--verbose',              help='print all messages?', action="store_true", default=False)
    parser.add_argument('--latents', help='Projection result file', type=str, default="styleGAN/results/aglaope_M_z_batch/latents.npz")
    parser.add_argument('--backbone', help='feature weights', type=str, default="../saved_models/vgg_backbone_nohybrid_D.pt")
    parser.add_argument('--classifier', help='classifier weights', type=str, default="../saved_models/vgg_classifier_nohybrid_D.pt")
    parser.add_argument('--source_lbl', help='target label', type=int, default=0)
    parser.add_argument('--target_lbl', help='target label', type=int, default=17)
    parser.add_argument('--learn_param', help='learnable param', type=str, default="w")

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

################################################
# Main
################################################

def test_user_args(args, G, D, F, C, images, start_zs, start_ws):
    projection_lbl = args.source_lbl if args.img_to_img else args.target_lbl
    project(
        images,
        G,
        D,
        F,
        C,
        projection_lbl,
        learn_param                = args.learn_param,
        start_zs                   = start_zs,
        start_ws                   = start_ws,
        num_steps                  = args.num_steps,
        init_lr                    = args.lr,
        img_to_img                 = args.img_to_img,
        batch                      = args.batch,
        use_entropy                = args.use_entropy,
        verbose                    = args.verbose
    )

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)

    # Load Models
    G, D, F, C = load_models(args.network, f_path=args.backbone, c_path=args.classifier)

    # Load Data
    images = load_imgs(args.target, view="D")

    print("Testing user specified args")
    latent_path = None if args.use_rand_latents else args.latents
    start_zs, start_ws = load_latents(G, latent_path, batch_size=args.batch_size)
    in_images = images[:args.batch_size]
    test_user_args(args, G, D, F, C, in_images, start_zs, start_ws)
    print("Test successful")

    #TODO: complete

    


