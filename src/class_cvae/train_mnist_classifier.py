from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import MNIST

from models import ResNet50
from logger import Logger
from options import MNIST_Classifier_Configs
from trainers.classifier_trainer import ClassifierTrainer

def load_data(configs):
    train_augs = []
    if configs.apply_rotation:
        train_augs.append(T.RandomRotation(45))
    train_augs.append(T.ToTensor())

    train_transform = T.Compose(train_augs)

    test_transform = T.Compose([
        T.ToTensor()
    ])

    train_dset = MNIST(root=configs.root_dset, train=True, transform=train_transform, download=True)
    test_dset = MNIST(root=configs.root_dset, train=False, transform=test_transform)
    train_dloader = DataLoader(train_dset, batch_size=configs.batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=configs.batch_size, shuffle=False)

    return train_dloader, test_dloader


if __name__ == "__main__":
    configs = MNIST_Classifier_Configs()
    num_classes = 10
    img_ch = 1
    train_dloader, test_dloader = load_data(configs)

    classifier = ResNet50(pretrain=configs.use_pretrain_resnet, num_classes=10, img_ch=1)
    
    logger = Logger(configs.output_dir, configs.exp_name)
    total_params = 0
    total_params += sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.log(f"Total trainable parameters: {total_params}")

    trainer = ClassifierTrainer(classifier, logger)
    trainer.train(train_dloader, test_dloader, configs)
