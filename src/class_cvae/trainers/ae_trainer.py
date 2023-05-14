
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from PIL import Image

from lpips.lpips import LPIPS
from utils import tensor_to_numpy_img

class AE_Trainer():
    def __init__(self, ae, img_classifier, lbls_to_att_fn, img_cls_resize_fn=None, \
                 gpu_id=None, logger=None):
        self.logger = logger
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

    def log(self, x):
        if self.logger is None: return
        if not self.is_base_process(): return
        self.logger.log(x)

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
                "swap" : 0,
                "g_loss" : 0,
                "d_loss" : 0,
                "all" : 0
            },
            "total" : 0,
            "correct" : 0,
            "zero_correct" : 0
        }

        return stats

    def compute_loss(self, imgs, lbls, stats, configs):
        pixel_loss_fn = nn.L1Loss()
        if configs.pixel_loss == "mse":
            pixel_loss_fn = nn.MSELoss()
        lpips_loss_fn = LPIPS(device=self.gpu_id)
        class_loss_fn = nn.CrossEntropyLoss()

        z = self.ae.module.encode(imgs)
        z_force = self.lbls_to_att_fn(lbls).float().to(z.get_device())
        if configs.force_hardcode:
            z_for_recon = torch.cat((z_force, z[:, self.num_att_vars:]), 1)
        else:
            z_for_recon = z

        imgs_recon = self.ae.module.decode(self.ae.module.replace(z_for_recon))

        l1_loss = pixel_loss_fn(imgs, imgs_recon)
        lpips_loss = lpips_loss_fn(imgs, imgs_recon)
        recon_loss = (l1_loss + lpips_loss) * configs.recon_lambda
        stats["losses"]["recon"] += recon_loss.item()
        stats["losses"]["l1"] += l1_loss.item()
        stats["losses"]["lpips"] += lpips_loss.item()
        loss = recon_loss

        if configs.add_gan:
            g_loss = self.compute_gen_loss(imgs_recon, configs)
            stats['losses']['g_loss'] += g_loss.item()
            loss += g_loss

        if configs.swap_lambda != 0:
            z_att = z_for_recon[:, :self.num_att_vars]
            z_var = z_for_recon[:, self.num_att_vars:]
            z_var_shuffle = z_var[torch.randperm(len(z_var))]
            z_comb = torch.cat((z_att, z_var_shuffle), 1)

            swap_recon = self.ae.module.decode(self.ae.module.replace(z_comb))
            out_swap = self.img_classifier(self.img_cls_resize_fn(swap_recon))
            swap_cls_loss = class_loss_fn(out_swap, lbls) 

            swap_loss = swap_cls_loss

            if configs.add_gan:
                swap_real_loss = self.compute_gen_loss(swap_recon, configs)
                swap_loss += swap_real_loss

            swap_loss *= configs.swap_lambda
            loss += swap_loss
            stats["losses"]["swap"] += swap_loss.item()
        
        if configs.cls_zero_lambda != 0 or configs.recon_zero_lambda != 0:
            z_zero_with_hardcode = torch.cat((z_force, torch.zeros_like(z[:, self.num_att_vars:])), 1)
            imgs_recon_zero_reg = self.ae.module.decode(self.ae.module.replace(z_zero_with_hardcode))
            recon_loss_zero = (pixel_loss_fn(imgs, imgs_recon_zero_reg) + lpips_loss_fn(imgs, imgs_recon_zero_reg)) * configs.recon_zero_lambda
            stats["losses"]["zero_reg_recon"] += recon_loss_zero.item()
            loss += recon_loss_zero

            out_zero = self.img_classifier(self.img_cls_resize_fn(imgs_recon_zero_reg))
            img_cls_zero_loss = class_loss_fn(out_zero, lbls) * configs.cls_zero_lambda
            stats["losses"]["img_cls_zero"] += img_cls_zero_loss.item()
            loss += img_cls_zero_loss

            _, img_preds = torch.max(out_zero, dim=1)

            stats["zero_correct"] += (img_preds == lbls).sum().item()

        if configs.force_dis_lambda != 0:
            reg = nn.L1Loss()(z[:, :self.num_att_vars], z_force) * configs.force_dis_lambda
            stats["losses"]["disentangle"] += reg.item()
            loss += reg

        if configs.kl_lambda != 0:
            normal_loss = self.ae.module.kl_loss() * configs.kl_lambda
            stats["losses"]["normal"] += normal_loss.item()
            loss += normal_loss

        if configs.cls_lambda != 0:
            out = self.img_classifier(self.img_cls_resize_fn(imgs_recon))

            img_cls_loss = class_loss_fn(out, lbls) * configs.cls_lambda
            stats["losses"]["img_cls"] += img_cls_loss.item()
            loss += img_cls_loss

            _, img_preds = torch.max(out, dim=1)

            stats["total"] += len(imgs)
            stats["correct"] += (img_preds == lbls).sum().item()

        stats["losses"]["all"] += loss.item()

        return loss, imgs_recon
    
    def eval(self, test_dloader, configs):
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
                if configs.force_hardcode:
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

            if self.logger is not None and self.is_base_process():
                gen_imgs = None
                if configs.add_gan:
                    gen_imgs = self.ae.module.generate(len(imgs), imgs.get_device())
                self.save_imgs(imgs, imgs_zero, imgs_recon, gen_imgs, self.logger.get_path())
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

        if final.shape[2] == 1:
            final = final[:, :, 0]

        Image.fromarray(final).save(f"{output_dir}/recon.png")

    def compute_gen_loss(self, imgs_recon, configs):
        fake_out = self.ae.module.discriminate(imgs_recon)
        g_out = torch.nn.functional.softplus(-fake_out)

        g_loss = (torch.sum(g_out) / g_out.shape[0]) * configs.g_lambda

        return g_loss

    def compute_dis_loss(self, imgs, recon, lbls, stats, configs):
        imgs_recon = recon.detach()
        
        real_out = self.ae.module.discriminate(imgs)
        d_loss_real = torch.nn.functional.softplus(-real_out).mean()

        #r1_grads = torch.autograd.grad(outputs=[real_out.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
        #r1_penalty = r1_grads.square().sum([1,2,3])
        #loss_Dr1 = r1_penalty * (configs.gamma / 2)

        fake_out = self.ae.module.discriminate(imgs_recon)
        d_loss_fake = torch.nn.functional.softplus(fake_out).mean()

        b, c, w, h = imgs.shape
        t = torch.rand((b, 1, 1, 1)).to(imgs.get_device())
        t = t.repeat(1, c, w, h)

        inter = (t * imgs + (1 - t) * imgs_recon).requires_grad_(True)
        inter_out = self.ae.module.discriminate(inter).view(b, -1)
        grads = torch.autograd.grad(outputs=inter_out, inputs=inter, grad_outputs=torch.ones_like(inter_out), \
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty_loss = torch.pow(grads.norm(2, dim=1)-1, 2).mean() * configs.gamma

        d_loss = (d_loss_real + grad_penalty_loss + d_loss_fake) * configs.d_lambda
        
        stats['losses']['d_loss'] += d_loss.item()
        stats["losses"]["all"] += d_loss.item()

        return d_loss

    def train(self, train_dloader, test_dloader, configs):

        params = list(self.ae.module.get_ae_parameters())
        optimizer = torch.optim.Adam(params, lr=configs.lr, betas=(0.5, 0.999))
        optimizerD = None
        if configs.add_gan:
            optimizerD = torch.optim.Adam(self.ae.module.discriminator.parameters(), \
                                          lr=configs.lr, betas=(0.5, 0.999))

        self.img_classifier.eval()
        for epoch in range(configs.epochs):
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

                loss, imgs_recon = self.compute_loss(imgs, lbls, stats, configs)

                loss.backward()
                optimizer.step()
                
                if configs.add_gan:
                    if configs.d_lambda != 0:
                        optimizerD.zero_grad()
                        d_loss = self.compute_dis_loss(imgs, imgs_recon, lbls, stats, configs)
                        d_loss.backward()
                        optimizerD.step()
                

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
                + f"G Loss: {stats['losses']['g_loss']} | " \
                + f"D Loss: {stats['losses']['d_loss']} | " \
                + f"Img Class Loss: {stats['losses']['img_cls']} | " \
                + f"Swap Loss: {stats['losses']['swap']} | " \
                + f"Image Class Acc: {round(stats['correct']/stats['total'], 4)} | " \
                #+ f"Zero Recon Loss: {stats['losses']['zero_reg_recon']} | " \
                #+ f"Sparsity Loss: {stats['losses']['sparcity']} | " \
                #+ f"Img Class Zero Loss: {stats['losses']['img_cls_zero']} | " \
                #+ f"Image Zero Class Acc: {round(stats['zero_correct']/stats['total'], 4)}" \

            self.log(out_string)

            if self.is_base_process():
                acc, zero_acc = self.eval(test_dloader, configs)
                self.log(f"Epoch: {epoch+1} | Test Accuracy: {round(acc, 4)} | Zero Test Accuracy: {round(zero_acc, 4)}")

            if self.logger is not None and self.is_base_process():
                torch.save(self.ae.module.state_dict(), f"{self.logger.get_path()}/ae.pt")
                torch.save(self.img_classifier.state_dict(), f"{self.logger.get_path()}/img_classifier.pt")

            torch.distributed.barrier()

