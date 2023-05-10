import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm


class ClassifierTrainer():
    def __init__(self, classifier, logger=None):
        self.classifier = classifier
        self.classifier.cuda()
        self.logger = logger

    def log(self, x):
        if self.logger is None: return
        self.logger.log(x)

    def save_model(self):
        path = "img_classifier.pt"
        if self.logger is not None:
            path = f"{self.logger.get_path()}/img_classifier.pt"

        torch.save(self.classifier.state_dict(), path)

    def train(self, train_dloader, test_dloader, configs):
        class_loss_fn = nn.CrossEntropyLoss()
        params = list(self.classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=configs.lr)

        for epoch in range(configs.epochs):
            losses = []
            correct = 0
            total = 0
            self.classifier.train()
            for (imgs, lbls) in tqdm(train_dloader):
                imgs = imgs.cuda()
                lbls = lbls.cuda()

                out = self.classifier(imgs)
                loss = class_loss_fn(out, lbls)

                _, preds = torch.max(out, dim=1)

                correct += (preds == lbls).sum().item()
                total += len(imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            self.log(f"Epoch: {epoch+1} | Class Loss: {np.mean(losses)} | Train Accuracy: {round(correct/total, 4)}")

            correct = 0
            total = 0
            self.classifier.eval()
            with torch.no_grad():
                for (imgs, lbls) in tqdm(test_dloader):
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                    out = self.classifier(imgs)

                    _, preds = torch.max(out, dim=1)

                    correct += (preds == lbls).sum().item()
                    total += len(imgs)

                self.log(f"Epoch: {epoch+1} | Test Accuracy: {round(correct/total, 4)}")

            self.save_model()
