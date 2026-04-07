import os
import argparse
import torch

from torch.backends import cudnn
from utils.utils import *
from solver import Solver


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes', 'y')


def main(config):
    if torch.cuda.is_available():
        cudnn.benchmark = True

    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)

    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ========= basic training =========
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # ========= original anomaly-transformer core =========
    parser.add_argument(
        '--k',
        type=float,
        default=3.0,
        help='Original discrepancy loss weight. Used as fallback when lambda_max is None.'
    )
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument(
        '--anormly_ratio',
        type=float,
        default=4.00,
        help='Legacy parameter kept for compatibility. Final evaluation threshold is now determined by utils/evaluation.py using validation scores and dataset-specific ratio.'
    )

    # ========= dataset / paths =========
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CreditCard')
    parser.add_argument('--data_path', type=str, default='./dataset/CreditCard')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    

    # ========= model architecture =========
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'])
    parser.add_argument('--output_attention', type=str2bool, default=True)

    # ========= prior design =========
    parser.add_argument(
        '--prior_type',
        type=str,
        default='gaussian',
        choices=['gaussian', 'mixture'],
        help='Use original single Gaussian prior or Gaussian Mixture prior.'
    )
    parser.add_argument(
        '--n_mixtures',
        type=int,
        default=1,
        help='Number of Gaussian mixture components when prior_type=mixture.'
    )
    parser.add_argument(
        '--normalize_prior',
        type=str2bool,
        default=True,
        help='Whether to row-normalize prior association.'
    )
    parser.add_argument(
        '--sigma_activation',
        type=str,
        default='softplus',
        choices=['softplus', 'exp', 'original'],
        help='Positive transform used for sigma.'
    )
    parser.add_argument(
        '--sigma_min',
        type=float,
        default=1e-4,
        help='Minimum sigma value after activation.'
    )

    # ========= discrepancy / optimization =========
    parser.add_argument(
        '--discrepancy',
        type=str,
        default='jsd',
        choices=['jsd', 'kl'],
        help='Association discrepancy metric used in training.'
    )
    parser.add_argument(
        '--lambda_max',
        type=float,
        default=None,
        help='Maximum discrepancy weight. If None, fallback to k.'
    )
    parser.add_argument(
        '--lambda_warmup_epochs',
        type=int,
        default=0,
        help='Linear warm-up epochs for lambda. 0 means no warm-up.'
    )

    # ========= structured sigma regularization =========
    parser.add_argument(
        '--sigma_smooth_weight',
        type=float,
        default=0.0,
        help='Temporal smoothness regularization weight for sigma.'
    )
    parser.add_argument(
        '--sigma_cross_layer_weight',
        type=float,
        default=0.0,
        help='Cross-layer consistency regularization weight for sigma.'
    )

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')

    main(config)