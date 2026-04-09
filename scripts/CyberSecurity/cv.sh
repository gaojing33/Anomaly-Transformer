#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
mkdir -p results

python main.py \
  --mode cv \
  --dataset CyberSecurity \
  --data_path ./dataset/CyberSecurity \
  --input_c 17 \
  --output_c 17 \
  --win_size 100 \
  --num_epochs 10 \
  --batch_size 256 \
  --lr 1e-4 \
  --anormly_ratio 1 \
  --k 3 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.0 \
  --activation gelu \
  --output_attention true \
  --prior_type mixture \
  --normalize_prior true \
  --sigma_activation softplus \
  --sigma_min 1e-4 \
  --discrepancy jsd \
  --lambda_max 3 \
  --lambda_warmup_epochs 2 \
  --sigma_smooth_weight 0.01 \
  --sigma_cross_layer_weight 0.0 \
  --cv_n_splits 3 \
  --cv_split_mode blocked \
  --cv_min_train_ratio 0.5 \
  --cv_metric f1 \
  --cv_use_adjustment false \
  --mixture_grid 1,2,3,4,5 \
  --cv_save_path results/cv_cybersecurity.json \
  --model_save_path checkpoints