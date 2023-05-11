import yaml

from argparse import ArgumentParser

class Options:
    def __init__(self, configs):
        for key in configs:
            setattr(self, key, configs[key])

    def __str__(self):
        out = "=======Options=======\n"
        for key in self.__dict__.keys():
            out += f"{key} => {self.__dict__[key]}\n"
        out += "====================="
        return out

def load_config(path):
    with open(path) as f:
        configs = yaml.safe_load(f)
    return Options(configs)

def add_configs(obj, path):
    with open(path) as f:
        configs = yaml.safe_load(f)
    for key in configs:
        setattr(obj, key, configs[key])

class Configs:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument('--output_dir', type=str, default="output")
        parser.add_argument('--exp_name', type=str, default="debug")
        parser.add_argument('--configs', type=str, default=None)
        self.add_arguments(parser)

        args = parser.parse_args()
        if args.configs is not None:
            add_configs(args, args.configs)
        
        for key in args.__dict__:
            att = getattr(args, key)
            setattr(self, key, att)

    def add_arguments(self, parser):
        assert False, "Not implmented. Configs is an abstract class"

class MNIST_Classifier_Configs(Configs):
    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--use_pretrain_resnet', action="store_true", default=False)
        parser.add_argument('--apply_rotation', action="store_true", default=False)
        parser.add_argument('--root_dset', type=str, default="data")

class CUB_VAEGAN_Configs(Configs):
    def add_arguments(self, parser):
        parser.add_argument('--use_bbox', action='store_true', default=False)
        parser.add_argument('--continue_checkpoint', action='store_true', default=False)
        parser.add_argument('--no_scheduler', action='store_true', default=False)
        parser.add_argument('--img_classifier', type=str, default=None)
        parser.add_argument('--add_real_cls_vec', action='store_true', default=False)
        parser.add_argument('--force_hardcode', action='store_true', default=False)
        parser.add_argument('--only_recon', action='store_true', default=False)
        parser.add_argument('--use_resnet_encoder', action='store_true', default=False)
        parser.add_argument('--inject_z', action='store_true', default=False)
        parser.add_argument('--add_gan', action='store_true', default=False)
        parser.add_argument('--use_patch_gan_dis', action='store_true', default=False)
        parser.add_argument('--ae', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--extra_layers', type=int, default=0)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--recon_lambda', type=float, default=1)
        parser.add_argument('--recon_zero_lambda', type=float, default=1)
        parser.add_argument('--cls_lambda', type=float, default=0.1)
        parser.add_argument('--cls_zero_lambda', type=float, default=0.1)
        parser.add_argument('--kl_lambda', type=float, default=0.0001)
        parser.add_argument('--sparcity_lambda', type=float, default=0)
        parser.add_argument('--force_dis_lambda', type=float, default=1)
        parser.add_argument('--d_lambda', type=float, default=1)
        parser.add_argument('--g_lambda', type=float, default=1)
        parser.add_argument('--gamma', type=float, default=5)
        parser.add_argument('--pixel_loss', type=str, default="l1", choices=["l1", "mse"])
        parser.add_argument('--num_features', type=int, default=512)
        parser.add_argument('--img_size', type=int, default=256)
        parser.add_argument('--depth', type=int, default=7)
        parser.add_argument('--n_disc_layers', type=int, default=3)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--port', type=str, default="5001")
        parser.add_argument('--root_dset', type=str, default="/local/scratch/cv_datasets/CUB_200_2011/")

class MNIST_VAEGAN_Configs(Configs):
    def add_arguments(self, parser):
        parser.add_argument('--continue_checkpoint', action='store_true', default=False)
        parser.add_argument('--img_classifier', type=str, default=None)
        parser.add_argument('--add_real_cls_vec', action='store_true', default=False)
        parser.add_argument('--force_hardcode', action='store_true', default=False)
        parser.add_argument('--only_recon', action='store_true', default=False)
        parser.add_argument('--use_resnet_encoder', action='store_true', default=False)
        parser.add_argument('--inject_z', action='store_true', default=False)
        parser.add_argument('--add_gan', action='store_true', default=False)
        parser.add_argument('--use_patch_gan_dis', action='store_true', default=False)
        parser.add_argument('--apply_rotation', action='store_true', default=False)
        parser.add_argument('--ae', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--extra_layers', type=int, default=0)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--recon_lambda', type=float, default=1)
        parser.add_argument('--recon_zero_lambda', type=float, default=1)
        parser.add_argument('--cls_lambda', type=float, default=0.1)
        parser.add_argument('--cls_zero_lambda', type=float, default=0.1)
        parser.add_argument('--kl_lambda', type=float, default=0.0001)
        parser.add_argument('--sparcity_lambda', type=float, default=0)
        parser.add_argument('--force_dis_lambda', type=float, default=1)
        parser.add_argument('--d_lambda', type=float, default=1)
        parser.add_argument('--g_lambda', type=float, default=1)
        parser.add_argument('--gamma', type=float, default=10)
        parser.add_argument('--pixel_loss', type=str, default="l1", choices=["l1", "mse"])
        parser.add_argument('--num_features', type=int, default=10)
        parser.add_argument('--img_size', type=int, default=32)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--n_disc_layers', type=int, default=3)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--port', type=str, default="5001")
        parser.add_argument('--root_dset', type=str, default="data")