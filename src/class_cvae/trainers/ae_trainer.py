
from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np

from PIL import Image

from lpips.lpips import LPIPS
from utils import tensor_to_numpy_img

class AE_Trainer():
    def __init__(self, ae, img_classifier, lbls_to_att_fn, img_cls_resize_fn=None):
        self.ae = ae
        self.img_classifier = img_classifier
        self.lbl_to_att_fn = lbls_to_att_fn
        self.img_cls_resize_fn = lambda x: x if img_cls_resize_fn is None else img_cls_resize_fn

    def init_stats(self):
        stats = {
            "losses" : {
                "recon" : 0,
                "zero_reg_recon" : 0,
                "latent_cls" : 0,
                "sparcity" : 0,
                "normal" : 0,
                "disentangle" : 0,
                "img_cls" : 0,
                "img_cls_zero" : 0,
                "all" : 0
            },
            "total" : 0,
            "correct" : 0
        }

    def compute_loss(self, imgs, lbls, stats, recon_lambda=1, recon_zero_lambda=1, \
                     cls_lambda=0.1, force_dis_lambda=1, sparcity_lambda=0.1, kl_lambda=0.001):
        l1_loss_fn = nn.L1Loss()
        lpips_loss_fn = LPIPS()
        class_loss_fn = nn.CrossEntropyLoss()
        recon_loss = l1_loss_fn(imgs, imgs_recon) + lpips_loss_fn(imgs, imgs_recon)

        z = self.ae.encode(imgs)
        z_force = self.lbls_to_att_fn(lbls).float()
        imgs_recon_zero_reg = self.ae.decode(torch.cat((z_force, torch.zeros_like(z[:, self.num_att_vars:])), 1))
        imgs_recon = self.ae.decode(z)

        stats["losses"]["recon"] += recon_loss.item()
        loss = recon_loss * recon_lambda
        
        recon_loss_zero = l1_loss_fn(imgs, imgs_recon_zero_reg) + lpips_loss_fn(imgs, imgs_recon_zero_reg)
        stats["losses"]["zero_reg_recon"] += recon_loss_zero.item()
        loss = recon_loss * recon_zero_lambda

        reg = nn.L1Loss()(z[:, :self.num_att_vars], z_force)
        stats["losses"]["disentangle"] += reg.item()
        loss += reg * force_dis_lambda

        normal_loss = self.ae.kl_loss()
        stats["losses"]["normal"] += normal_loss.item()
        loss += normal_loss * kl_lambda

        sparcity_loss = l1_loss_fn(z, torch.zeros_like(z))
        stats["losses"]["sparcity"] += sparcity_loss.item()
        loss += sparcity_loss * sparcity_lambda

        out = self.img_classifier(self.img_cls_resize_fn(imgs_recon))

        img_cls_loss = class_loss_fn(out, lbls)
        stats["losses"]["img_cls"] += img_cls_loss.item()
        loss += img_cls_loss * cls_lambda

        _, img_preds = torch.max(out, dim=1)

        stats["total"] += len(imgs)
        stats["correct"] += (img_preds == lbls).sum().item()

        stats["losses"]["all"] += loss.item()

        return loss
    
    def eval(self, test_dloader, logger=None):
        correct = 0
        total = 0
        self.ae.eval()
        with torch.no_grad():
            for imgs, lbls in test_dloader:
                imgs = imgs.cuda()
                lbls = lbls.cuda()

                z = self.ae.encode(imgs)
                imgs_recon = self.ae.decode()
                out = self.img_classifier(self.img_cls_resize_fn(imgs_recon))

                _, preds = torch.max(out, dim=1)

                correct += (preds == lbls).sum().item()
                total += len(imgs)

            if logger:
                self.save_imgs(imgs, imgs_recon, logger.get_path())
        return correct / total

    def save_imgs(reals, fakes, output_dir):
        reals = tensor_to_numpy_img(reals).astype(np.float)
        fakes = tensor_to_numpy_img(fakes).astype(np.float)

        final = None
        for img in reals:
            if final is None:
                final = img
            else:
                final = np.concatenate((final, img), axis=1)

        tmp = None
        for img in fakes:
            if tmp is None:
                tmp = img
            else:
                tmp = np.concatenate((tmp, img), axis=1)

        final_img = np.concatenate((final, tmp), axis=0)[:, :, 0].astype(np.uint8)

        Image.fromarray(final_img).save(f"{output_dir}/recon.png")

    def train(self, train_dloader, test_dloader, epochs=100, lr=0.0001, logger=None, \
                recon_lambda=1, recon_zero_lambda=1, cls_lambda=0.1, force_dis_lambda=1, \
                sparcity_lambda=0.1, kl_lambda=0.001):
        def log(x):
            if logger is None:
                return
            logger.log(x)

        self.ae.cuda()
        self.img_classifier.cuda()

        params = list(self.ae.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        self.img_classifier.eval()
        for epoch in range(epochs):
            stats = self.init_stats()
            self.ae.train()
            
            for imgs, lbls in tqdm(train_dloader, desc="Training"):
                imgs = imgs.cuda()
                lbls = lbls.cuda()

                loss = self.compute_loss(imgs, lbls, stats, recon_lambda, recon_zero_lambda, \
                                         cls_lambda, force_dis_lambda, sparcity_lambda, kl_lambda)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for key in stats["losses"]:
                stats["losses"][key] = round(stats["losses"][key] / len(self.train_dloader), 4)

            out_string = f"Epoch: {epoch+1} | " \
                + f"Total Loss: {stats['losses']['all']} | " \
                + f"Disentangle Loss: {stats['losses']['disentangle']} | " \
                + f"Normal Loss: {stats['losses']['normal']} | " \
                + f"Recon Loss: {stats['losses']['recon']} | " \
                + f"Zero Recon Loss: {stats['losses']['zero_reg_recon']} | " \
                + f"Sparsity Loss: {stats['losses']['sparcity']} | " \
                + f"Img Class Loss: {stats['losses']['img_cls']} | " \
                + f"Img Class Zero Loss: {stats['losses']['img_cls_zero']} | " \
                + f"Image Class Acc: {round(stats['correct']/stats['total'], 4)}"

            log(out_string)

            acc = self.eval(test_dloader, logger)

            log(f"Epoch: {epoch+1} | Test Accuracy: {round(acc, 4)}")

            if logger is not None:
                torch.save(self.ae.state_dict(), f"{logger.get_path()}/ae.pt")
                torch.save(self.img_classifier.state_dict(), f"{logger.get_path()}/img_classifier.pt")

