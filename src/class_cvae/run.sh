nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python train.py --exp_name classless' > out1.log &
nohup sh -c 'CUDA_VISIABLE_DEVICES=1 python train.py --exp_name latent_classifier --add_classifier' > out2.log &
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python train.py --add_img_classifier --exp_name vae_img_classifier' > out3.log &
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python train.py --add_img_classifier --pretrain_img_classifier output/img_classifier/img_classifier.pt --exp_name vae_pretrain_img_classifier' > out4.log &