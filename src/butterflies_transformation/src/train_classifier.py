from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from options import load_config
from datasets import CuthillDataset
from models import VGG_Classifier
from tools import init_weights


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

options = load_config('../configs/cuthill_train.yaml')

dset = CuthillDataset(options, train=True, transform=ToTensor())
dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=4)

test_dset = CuthillDataset(options, train=False, transform=ToTensor())
test_dloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=4)

classifier = VGG_Classifier(class_num=18, pretrain=True).cuda()

classifier.apply(init_weights)

loss_fn = torch.nn.CrossEntropyLoss()

def train(epoch, optimizer):
    total_loss = 0
    correct = 0
    total = 0
    for imgs, lbls, _ in dloader:
        imgs = imgs.cuda()
        lbls = lbls.cuda()
        
        out = classifier(imgs)
        loss = loss_fn(out, lbls)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total += len(imgs)
        _, idx = torch.max(out, dim=1)
        correct += (idx == lbls).sum()

    print(f"Epoch: {epoch+1} | Train Loss: {total_loss} | Train Accuracy: {correct/total}")
    torch.save(classifier.state_dict(), "../tmp/classifier.pt")

def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls, _ in test_dloader:
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            
            out = classifier(imgs)

            total += len(imgs)
            _, idx = torch.max(out, dim=1)
            correct += (idx == lbls).sum()
        print(f"Epoch: {epoch+1} | Test Accuracy: {correct/total}")

optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    train(epoch, optimizer)
    test(epoch)