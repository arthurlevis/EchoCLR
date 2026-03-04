#!/usr/bin/bash

# EchoCLR
python main.py \
    --data_dir /path/to/dataset \
    --out_dir /path/to/results/ \
    --experiment_name echoclr \
    --backbone r3d_18 \
    --multi_instance \
    --frame_reordering \
    --batch_size 16 \
    --n_gpu 1 \
    --temperature 0.05 \
    --projection_dim 128 \
    --lr 0.1 \
    --num_epochs 300 \
    --patience 10 \
    --save_freq 10 \
    --clip_len 4 \
    --sampling_rate 1
