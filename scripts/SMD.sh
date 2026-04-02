export CUDA_VISIBLE_DEVICES=0

# Note:
# --anormly_ratio is kept only for compatibility with the original code interface.
# Final evaluation threshold is selected by utils/evaluation.py
# using validation scores and dataset-specific anomaly ratio.

python main.py \
  --mode train \
  --dataset SMD \
  --data_path dataset/SMD \
  --input_c 38 \
  --output_c 38 \
  --win_size 100 \
  --num_epochs 10 \
  --batch_size 256 \
  --lr 1e-4 \
  --anormly_ratio 0.5 \
  --k 3 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.0 \
  --activation gelu \
  --output_attention true \
  --prior_type mixture \
  --n_mixtures 3 \
  --normalize_prior true \
  --sigma_activation softplus \
  --sigma_min 1e-4 \
  --discrepancy jsd \
  --lambda_max 3 \
  --lambda_warmup_epochs 3 \
  --sigma_smooth_weight 0.01 \
  --sigma_cross_layer_weight 0.0 \
  --model_save_path checkpoints

python main.py \
  --mode test \
  --dataset SMD \
  --data_path dataset/SMD \
  --input_c 38 \
  --output_c 38 \
  --win_size 100 \
  --num_epochs 10 \
  --batch_size 256 \
  --lr 1e-4 \
  --anormly_ratio 0.5 \
  --k 3 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.0 \
  --activation gelu \
  --output_attention true \
  --prior_type mixture \
  --n_mixtures 3 \
  --normalize_prior true \
  --sigma_activation softplus \
  --sigma_min 1e-4 \
  --discrepancy jsd \
  --lambda_max 3 \
  --lambda_warmup_epochs 3 \
  --sigma_smooth_weight 0.01 \
  --sigma_cross_layer_weight 0.0 \
  --model_save_path checkpoints