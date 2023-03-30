CUDA_VISIBLE_DEVICES=2,3 python train_stylegan3.py --outdir=stylegan3/butterfly_training_runs_original \
    --data=../datasets/original/train_original_128_128.zip \
    --gpus=2 \
    --cfg=stylegan3-r \
    --batch 32 \
    --mirror=1 \
    --snap 250 \
    --gamma 6.6 \
    --kimg=5000