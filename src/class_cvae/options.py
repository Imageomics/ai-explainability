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