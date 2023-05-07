
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from PIL import Image

from lpips.lpips import LPIPS
from utils import tensor_to_numpy_img

class AE_Trainer():
    def __init__(self, ae, img_classifier, lbls_to_att_fn, img_cls_resize_fn=None, gpu_id=None):
        self.gpu_id = gpu_id
        if gpu_id is not None:
            self.ae = DDP(ae.to(gpu_id), device_ids=[gpu_id], find_unused_parameters=True)
            self.img_classifier = img_classifier.to(gpu_id)
        else:
            self.ae = ae.cuda()
            self.img_classifier = img_classifier.cuda()
            
        self.lbls_to_att_fn = lbls_to_att_fn
        self.img_cls_resize_fn = img_cls_resize_fn 
        if self.img_cls_resize_fn is None:
            self.img_cls_resize_fn = lambda x: x 
        self.num_att_vars = len(lbls_to_att_fn(torch.tensor([0]))[0])

    def init_stats(self):
        stats = {
            "losses" : {
                "recon" : 0,
                "l1" : 0,
                "lpips" : 0,
                "zero_reg_recon" : 0,
                "latent_cls" : 0,
                "sparcity" : 0,
                "normal" : 0,
                "disentangle" : 0,
                "img_cls" : 0,
                "img_cls_zero" : 0,
                "g_loss" : 0,
                "d_loss" : 0,
                "all" : 0
            },
            "total" : 0,
            "correct" : 0,
            "zero_correct" : 0
        }

        return stats

    def compute_loss(self, imgs, lbls, stats, recon_lambda=1, recon_zero_lambda=1, \
                     cls_lambda=0.1, cls_zero_lambda=0.1, force_dis_lambda=1, sparcity_lambda=0.1, \
                     kl_lambda=0.001, force_hardcode=False, pixel_loss="l1"):
        pixel_loss_fn = nn.L1Loss()
        if pixel_loss == "mse":
            pixel_loss_fn = nn.MSELoss()
        lpips_loss_fn = LPIPS(device=self.gpu_id)
        class_loss_fn = nn.CrossEntropyLoss()

        z = self.ae.module.encode(imgs)
        z_force = self.lbls_to_att_fn(lbls).float().to(z.get_device())
        if force_hardcode:
            z_with_hardcode = torch.cat((z_force, z[:, self.num_att_vars:]), 1)
            imgs_recon = self.ae.module.decode(self.ae.module.replace(z_with_hardcode))
        else:
            imgs_recon = self.ae.module.decode(self.ae.module.replace(z))

        l1_loss = pixel_loss_fn(imgs, imgs_recon)
        lpips_loss = lpips_loss_fn(imgs, imgs_recon)
        recon_loss = (l1_loss + lpips_loss) * recon_lambda
        stats["losses"]["recon"] += recon_loss.item()
        stats["losses"]["l1"] += l1_loss.item()
        stats["losses"]["lpips"] += lpips_loss.item()
        loss = recon_loss
        
        if cls_zero_lambda != 0 or recon_zero_lambda != 0:
            z_zero_with_hardcode = torch.cat((z_force, torch.zeros_like(z[:, self.num_att_vars:])), 1)
            imgs_recon_zero_reg = self.ae.module.decode(self.ae.module.replace(z_zero_with_hardcode))
            recon_loss_zero = (l1_loss_fn(imgs, imgs_recon_zero_reg) + lpips_loss_fn(imgs, imgs_recon_zero_reg)) * recon_zero_lambda
            stats["losses"]["zero_reg_recon"] += recon_loss_zero.item()
            loss += recon_loss_zero

            out_zero = self.img_classifier(self.img_cls_resize_fn(imgs_recon_zero_reg))
            img_cls_zero_loss = class_loss_fn(out_zero, lbls) * cls_zero_lambda
            stats["losses"]["img_cls_zero"] += img_cls_zero_loss.item()
            loss += img_cls_zero_loss

            _, img_preds = torch.max(out_zero, dim=1)

            stats["zero_correct"] += (img_preds == lbls).sum().item()

        if force_dis_lambda != 0:
            reg = nn.L1Loss()(z[:, :self.num_att_vars], z_force) * force_dis_lambda
            stats["losses"]["disentangle"] += reg.item()
            loss += reg

        if kl_lambda != 0:
            normal_loss = self.ae.module.kl_loss() * kl_lambda
            stats["losses"]["normal"] += normal_loss.item()
            loss += normal_loss

        if sparcity_lambda != 0:
            sparcity_loss = l1_loss_fn(z, torch.zeros_like(z)) * sparcity_lambda
            stats["losses"]["sparcity"] += sparcity_loss.item()
            loss += sparcity_loss

        if cls_lambda != 0:
            out = self.img_classifier(self.img_cls_resize_fn(imgs_recon))

            img_cls_loss = class_loss_fn(out, lbls) * cls_lambda
            stats["losses"]["img_cls"] += img_cls_loss.item()
            loss += img_cls_loss

            _, img_preds = torch.max(out, dim=1)

            stats["total"] += len(imgs)
            stats["correct"] += (img_preds == lbls).sum().item()

        stats["losses"]["all"] += loss.item()

        return loss
    
    def eval(self, test_dloader, logger=None, force_hardcode=False, add_gan=False):
        correct = 0
        zero_correct = 0
        total = 0
        self.ae.eval()
        with torch.no_grad():
            for (imgs, lbls) in tqdm(test_dloader, desc="Evaluation"):
                imgs = self.set_device(imgs)
                lbls = self.set_device(lbls)


                z = self.ae.module.encode(imgs)
                z_force = self.lbls_to_att_fn(lbls).float().to(z.get_device())
                if force_hardcode:
                    imgs_recon = self.ae.module.decode(torch.cat((z_force, z[:, self.num_att_vars:]), 1))
                else:
                    imgs_recon = self.ae.module.decode(z)
                
                out = self.img_classifier(self.img_cls_resize_fn(imgs_recon))
                _, preds = torch.max(out, dim=1)

                correct += (preds == lbls).sum().item()
                total += len(imgs)

                imgs_zero = self.ae.module.decode(torch.cat((z_force, torch.zeros_like(z[:, self.num_att_vars:])), 1))
                zero_out = self.img_classifier(self.img_cls_resize_fn(imgs_zero))
                _, zero_preds = torch.max(zero_out, dim=1)
                zero_correct += (zero_preds == lbls).sum().item()

            if logger and self.is_base_process():
                gen_imgs = None
                if add_gan:
                    gen_imgs = self.ae.module.generate(len(imgs), imgs.get_device())
                self.save_imgs(imgs, imgs_zero, imgs_recon, gen_imgs, logger.get_path())
        return correct / total, zero_correct / total

    def is_base_process(self):
        if self.gpu_id is not None and self.gpu_id != 0:
            return False
        return True

    def set_device(self, x):
        if self.gpu_id is not None:
            return x.to(self.gpu_id)
        return x.cuda()

    def save_imgs(self, reals, zero_fakes, fakes, gen_imgs, output_dir):      
        reals = tensor_to_numpy_img(reals).astype(np.uint8)
        zero_fakes = tensor_to_numpy_img(zero_fakes).astype(np.uint8)
        fakes = tensor_to_numpy_img(fakes).astype(np.uint8)

        image_sets = [reals, zero_fakes, fakes]

        if gen_imgs is not None:
            gen_imgs = tensor_to_numpy_img(gen_imgs).astype(np.uint8)
            image_sets.append(gen_imgs)

        final = None

        for img_set in image_sets:
            tmp = None
            for img in img_set:
                if tmp is None:
                    tmp = img
                else:
                    tmp = np.concatenate((tmp, img), axis=1)
            if final is None:
                final = tmp
            else:
                final = np.concatenate((final, tmp), axis=0)

        Image.fromarray(final).save(f"{output_dir}/recon.png")

    def compute_gen_loss(self, imgs, lbls, stats, force_hardcode):
        device = imgs.get_device()
        fake_imgs = self.ae.module.generate(len(imgs), device)
        fake_lbls = torch.ones(len(imgs), 1).to(device)

        fake_out = self.ae.module.discriminate(fake_imgs)
        g_loss = torch.nn.functional.softplus(-fake_out)

        g_loss = g_loss.mean()
        
        stats['losses']['g_loss'] += g_loss.item()
        stats["losses"]["all"] += g_loss.item()

        return g_loss

    def compute_dis_loss(self, imgs, lbls, stats):
        device = imgs.get_device()
        
        fake_lbls = torch.zeros(len(imgs), 1).to(device)
        real_lbls = torch.ones(len(imgs), 1).to(device)
        
        real_out = self.ae.module.discriminate(imgs)
        d_loss_real = torch.nn.functional.softplus(-real_out)

        fake_imgs = self.ae.module.generate(len(imgs), device)
        fake_out = self.ae.module.discriminate(fake_imgs)
        d_loss_fake = torch.nn.functional.softplus(fake_out)

        d_loss = (d_loss_real + d_loss_fake).mean()
        
        stats['losses']['d_loss'] += d_loss.item()
        stats["losses"]["all"] += d_loss.item()

        return d_loss

    def train(self, train_dloader, test_dloader, epochs=100, lr=0.0001, logger=None, \
                recon_lambda=1, recon_zero_lambda=1, cls_lambda=0.1, cls_zero_lambda=0.1, \
                force_dis_lambda=1, sparcity_lambda=0.1, kl_lambda=0.001, use_scheduler=False, \
                force_hardcode=False, add_gan=False, pixel_loss="l1"):
        def log(x):
            if logger is None: return
            if not self.is_base_process(): return
            
            logger.log(x)

        params = list(self.ae.module.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        if add_gan:
            gparams = list(self.ae.module.iin_ae.parameters()) + list(self.ae.module.generator.parameters())
            optimizerG = torch.optim.Adam(gparams, lr=lr, betas=(0.5, 0.999))
            optimizerD = torch.optim.Adam(self.ae.module.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        self.img_classifier.eval()
        for epoch in range(epochs):
            stats = self.init_stats()
            self.ae.train()
            if self.gpu_id is not None:
                train_dloader.sampler.set_epoch(epoch)
            num_batches = 0
            for imgs, lbls in tqdm(train_dloader, desc=f"Training Epoch {epoch+1}"):
                optimizer.zero_grad(set_to_none=True)
                
                num_batches += 1
                imgs = self.set_device(imgs)
                lbls = self.set_device(lbls)

                loss = self.compute_loss(imgs, lbls, stats, recon_lambda, recon_zero_lambda, \
                                         cls_lambda, cls_zero_lambda, force_dis_lambda, sparcity_lambda, \
                                         kl_lambda, force_hardcode, pixel_loss)
                
                loss.backward()
                optimizer.step()
                
                if add_gan:
                    d_loss = self.compute_dis_loss(imgs, lbls, stats)
                    optimizerD.zero_grad()
                    d_loss.backward()
                    optimizerD.step()
                    
                    g_loss = self.compute_gen_loss(imgs, lbls, stats, force_hardcode)
                    optimizerG.zero_grad()
                    g_loss.backward()
                    optimizerG.step()
                    


            for key in stats["losses"]:
                stats["losses"][key] = round(stats["losses"][key] / num_batches, 4)

            if stats['total'] == 0:
                stats['total'] = 1
            out_string = f"Epoch: {epoch+1} | " \
                + f"Total Loss: {stats['losses']['all']} | " \
                + f"Disentangle Loss: {stats['losses']['disentangle']} | " \
                + f"Normal Loss: {stats['losses']['normal']} | " \
                + f"Recon Loss: {stats['losses']['recon']} | " \
                + f"L1 Loss: {stats['losses']['l1']} | " \
                + f"LPIPS Loss: {stats['losses']['lpips']} | " \
                + f"Zero Recon Loss: {stats['losses']['zero_reg_recon']} | " \
                + f"Sparsity Loss: {stats['losses']['sparcity']} | " \
                + f"G Loss: {stats['losses']['g_loss']} | " \
                + f"D Loss: {stats['losses']['d_loss']} | " \
                + f"Img Class Loss: {stats['losses']['img_cls']} | " \
                + f"Img Class Zero Loss: {stats['losses']['img_cls_zero']} | " \
                + f"Image Class Acc: {round(stats['correct']/stats['total'], 4)} | " \
                + f"Image Zero Class Acc: {round(stats['zero_correct']/stats['total'], 4)}" \

            log(out_string)

            if self.is_base_process():
                acc, zero_acc = self.eval(test_dloader, logger, force_hardcode, add_gan)
                log(f"Epoch: {epoch+1} | Test Accuracy: {round(acc, 4)} | Zero Test Accuracy: {round(zero_acc, 4)}")

            if logger is not None and self.is_base_process():
                torch.save(self.ae.module.state_dict(), f"{logger.get_path()}/ae.pt")
                torch.save(self.img_classifier.state_dict(), f"{logger.get_path()}/img_classifier.pt")

            torch.distributed.barrier()

        if use_scheduler:
            scheduler.step()

