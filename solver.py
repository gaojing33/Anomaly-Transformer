# GPU-version (stable)
import os
import time
import numpy as np

import torch
import torch.nn as nn

try:
    import torch_directml
    HAS_DML = True
except ImportError:
    torch_directml = None
    HAS_DML = False

from utils.utils import *
from utils.evaluation import evaluate_dataset
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def is_finite_number(x):
    return np.isfinite(float(x))


def safe_item(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().item()
    return float(x)


def my_kl_loss(p, q, eps=1e-8):
    """
    Point-wise KL divergence aggregated over the last dimension.

    Input:
        p, q: [B, H, L, L] or compatible distributions
    Return:
        [B, L]
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = p / torch.clamp(torch.sum(p, dim=-1, keepdim=True), min=eps)
    q = q / torch.clamp(torch.sum(q, dim=-1, keepdim=True), min=eps)
    res = p * (torch.log(p) - torch.log(q))
    return torch.mean(torch.sum(res, dim=-1), dim=1)



def my_jsd_loss(p, q, eps=1e-8):
    """
    Point-wise Jensen-Shannon divergence.

    Input:
        p, q: [B, H, L, L]
    Return:
        [B, L]
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = p / torch.clamp(torch.sum(p, dim=-1, keepdim=True), min=eps)
    q = q / torch.clamp(torch.sum(q, dim=-1, keepdim=True), min=eps)
    m = 0.5 * (p + q)
    m = torch.clamp(m, min=eps)
    m = m / torch.clamp(torch.sum(m, dim=-1, keepdim=True), min=eps)
    kl_pm = p * (torch.log(p) - torch.log(m))
    kl_qm = q * (torch.log(q) - torch.log(m))
    jsd = 0.5 * torch.sum(kl_pm, dim=-1) + 0.5 * torch.sum(kl_qm, dim=-1)
    return torch.mean(jsd, dim=1)



def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        if (not is_finite_number(val_loss)) or (not is_finite_number(val_loss2)):
            print(f'Invalid validation loss detected: val_loss={val_loss}, val_loss2={val_loss2}. Skip saving.')
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {
        "e_layers": 3,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 512,
        "dropout": 0.0,
        "activation": "gelu",
        "output_attention": True,
        "prior_type": "gaussian",
        "n_mixtures": 1,
        "normalize_prior": True,
        "sigma_activation": "softplus",
        "sigma_min": 1e-2,
        "discrepancy": "kl",
        "lambda_max": None,
        "lambda_warmup_epochs": 1,
        "sigma_smooth_weight": 0.0,
        "sigma_cross_layer_weight": 0.0,
        "grad_clip": 1.0,
    }

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        self.vali_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='val',
            dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_name = "cuda:0"
        elif HAS_DML:
            self.device = torch_directml.device()
            self.device_name = "directml"
        else:
            self.device = torch.device("cpu")
            self.device_name = "cpu"

        print(f"Using device: {self.device_name} -> {self.device}")

        self.build_model()
        self.criterion = nn.MSELoss()

        if self.lambda_max is None:
            self.lambda_max = self.k

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
            prior_type=self.prior_type,
            n_mixtures=self.n_mixtures,
            normalize_prior=self.normalize_prior,
            sigma_activation=self.sigma_activation,
            sigma_min=self.sigma_min,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_current_lambda(self, epoch_idx):
        if self.lambda_warmup_epochs is None or self.lambda_warmup_epochs <= 0:
            return self.lambda_max

        progress = float(epoch_idx + 1) / float(self.lambda_warmup_epochs)
        progress = min(progress, 1.0)
        return self.lambda_max * progress

    def _get_discrepancy_loss(self, p, q):
        if self.discrepancy.lower() == "jsd":
            return my_jsd_loss(p, q)
        return my_kl_loss(p, q)

    def _association_losses(self, series_list, prior_list):
        """
        Return:
            series_loss: scalar
            prior_loss:  scalar
            series_loss_map: [B, L]
            prior_loss_map:  [B, L]
        """
        series_loss = 0.0
        prior_loss = 0.0
        series_loss_map = None
        prior_loss_map = None

        num_layers = len(prior_list)

        for u in range(num_layers):
            prior_u = prior_list[u]
            series_u = series_list[u]

            cur_series = self._get_discrepancy_loss(series_u, prior_u.detach()) + \
                         self._get_discrepancy_loss(prior_u.detach(), series_u)

            cur_prior = self._get_discrepancy_loss(prior_u, series_u.detach()) + \
                        self._get_discrepancy_loss(series_u.detach(), prior_u)

            cur_series = torch.nan_to_num(cur_series, nan=0.0, posinf=1e6, neginf=-1e6)
            cur_prior = torch.nan_to_num(cur_prior, nan=0.0, posinf=1e6, neginf=-1e6)

            series_loss += torch.mean(cur_series)
            prior_loss += torch.mean(cur_prior)

            if series_loss_map is None:
                series_loss_map = cur_series
                prior_loss_map = cur_prior
            else:
                series_loss_map = series_loss_map + cur_series
                prior_loss_map = prior_loss_map + cur_prior

        series_loss = series_loss / num_layers
        prior_loss = prior_loss / num_layers
        series_loss_map = series_loss_map / num_layers
        prior_loss_map = prior_loss_map / num_layers

        return series_loss, prior_loss, series_loss_map, prior_loss_map

    def _sigma_regularization(self, sigma_list):
        reg = 0.0

        if sigma_list is None or len(sigma_list) == 0:
            return torch.tensor(0.0, device=self.device)

        if self.sigma_smooth_weight > 0:
            smooth_reg = 0.0
            for sigma in sigma_list:
                sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
                if sigma.size(2) > 1:
                    smooth_reg = smooth_reg + torch.mean((sigma[:, :, 1:, :] - sigma[:, :, :-1, :]) ** 2)
            reg = reg + self.sigma_smooth_weight * (smooth_reg / len(sigma_list))

        if self.sigma_cross_layer_weight > 0 and len(sigma_list) > 1:
            cross_reg = 0.0
            for idx in range(1, len(sigma_list)):
                s1 = torch.nan_to_num(sigma_list[idx], nan=0.0, posinf=1e6, neginf=-1e6)
                s0 = torch.nan_to_num(sigma_list[idx - 1], nan=0.0, posinf=1e6, neginf=-1e6)
                cross_reg = cross_reg + torch.mean((s1 - s0) ** 2)
            reg = reg + self.sigma_cross_layer_weight * (cross_reg / (len(sigma_list) - 1))

        if not isinstance(reg, torch.Tensor):
            reg = torch.tensor(reg, device=self.device)

        reg = torch.nan_to_num(reg, nan=0.0, posinf=1e6, neginf=-1e6)
        return reg

    def vali(self, vali_loader, epoch_idx=0):
        self.model.eval()

        loss_1 = []
        loss_2 = []

        lambda_cur = self._get_current_lambda(epoch_idx)

        with torch.no_grad():
            for _, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)

                if not torch.isfinite(input).all():
                    print('Warning: non-finite input detected in validation loader, skipping batch.')
                    continue

                output, series, prior, sigmas, mixture_weights = self.model(input)

                rec_loss = self.criterion(output, input)
                series_loss, prior_loss, _, _ = self._association_losses(series, prior)
                sigma_reg = self._sigma_regularization(sigmas)

                total_1 = rec_loss - lambda_cur * series_loss + sigma_reg
                total_2 = rec_loss + lambda_cur * prior_loss + sigma_reg

                if torch.isfinite(total_1) and torch.isfinite(total_2):
                    loss_1.append(safe_item(total_1))
                    loss_2.append(safe_item(total_2))

        if len(loss_1) == 0 or len(loss_2) == 0:
            return np.nan, np.nan

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            skipped_batches = 0

            epoch_time = time.time()
            lambda_cur = self._get_current_lambda(epoch)

            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)
                iter_count += 1

                input = input_data.float().to(self.device)
                if not torch.isfinite(input).all():
                    skipped_batches += 1
                    print(f'Warning: non-finite input detected in training batch {i}, skipping batch.')
                    continue

                output, series, prior, sigmas, mixture_weights = self.model(input)

                rec_loss = self.criterion(output, input)
                series_loss, prior_loss, _, _ = self._association_losses(series, prior)
                sigma_reg = self._sigma_regularization(sigmas)

                loss1 = rec_loss - lambda_cur * series_loss + sigma_reg
                loss2 = rec_loss + lambda_cur * prior_loss + sigma_reg

                if (not torch.isfinite(loss1)) or (not torch.isfinite(loss2)):
                    skipped_batches += 1
                    print(f'Warning: non-finite loss at epoch {epoch + 1}, batch {i + 1}; skipping optimizer step.')
                    continue

                loss1_list.append(loss1.detach().float().cpu().item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s | lambda: {:.6f} | skipped: {}'.format(
                            speed, left_time, lambda_cur, skipped_batches
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retain_graph=True)
                loss2.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list) if len(loss1_list) > 0 else np.nan

            vali_loss1, vali_loss2 = self.vali(self.vali_loader, epoch_idx=epoch)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss1: {3:.7f} | Vali Loss2: {4:.7f} | lambda: {5:.6f} | skipped: {6}".format(
                    epoch + 1, train_steps, train_loss, vali_loss1, vali_loss2, lambda_cur, skipped_batches
                )
            )

            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def _compute_energy(self, loader, criterion, temperature=50):
        """
        Compute anomaly energy on a loader.

        Return:
            energies: flattened numpy array
            labels: flattened numpy array or None
        """
        attens_energy = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for _, (input_data, labels) in enumerate(loader):
                input = input_data.float().to(self.device)
                if not torch.isfinite(input).all():
                    continue

                output, series, prior, sigmas, mixture_weights = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)
                _, _, series_loss_map, prior_loss_map = self._association_losses(series, prior)

                series_loss_map = torch.nan_to_num(series_loss_map * temperature, nan=0.0, posinf=1e6, neginf=-1e6)
                prior_loss_map = torch.nan_to_num(prior_loss_map * temperature, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

                metric = torch.softmax((-series_loss_map - prior_loss_map), dim=-1)
                cri = metric * loss
                cri = torch.nan_to_num(cri, nan=0.0, posinf=1e6, neginf=-1e6)

                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

                if labels is not None:
                    labels_list.append(labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1) if len(attens_energy) > 0 else np.array([])

        if len(labels_list) > 0:
            labels_np = np.concatenate(labels_list, axis=0).reshape(-1)
        else:
            labels_np = None

        return attens_energy, labels_np

    def test(self):
        ckpt_path = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        self.model.load_state_dict(
            torch.load(
                ckpt_path,
                map_location=self.device
            )
        )
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        temperature = 50

        val_scores, _ = self._compute_energy(self.vali_loader, criterion, temperature=temperature)
        test_scores, test_labels = self._compute_energy(self.test_loader, criterion, temperature=temperature)

        print("val_scores shape: ", np.asarray(val_scores).shape)
        print("test_scores shape:", np.asarray(test_scores).shape)
        print("test_labels shape:", np.asarray(test_labels).shape)

        raw_results, raw_pred = evaluate_dataset(
            dataset_name=self.dataset,
            val_scores=val_scores,
            test_scores=test_scores,
            test_labels=test_labels,
            use_adjustment=False
        )

        adj_results, adj_pred = evaluate_dataset(
            dataset_name=self.dataset,
            val_scores=val_scores,
            test_scores=test_scores,
            test_labels=test_labels,
            use_adjustment=True
        )

        print("========== Raw Evaluation ==========")
        print(
            "Dataset: {0} | Ratio: {1} | Threshold: {2:.6f} | Precision: {3:.4f} | Recall: {4:.4f} | F1: {5:.4f} | ROC-AUC: {6:.4f}".format(
                raw_results["dataset"],
                raw_results["ratio"],
                raw_results["threshold"],
                raw_results["precision"],
                raw_results["recall"],
                raw_results["f1"],
                raw_results["roc_auc"] if not np.isnan(raw_results["roc_auc"]) else float("nan")
            )
        )

        print("======= Adjusted Evaluation ========")
        print(
            "Dataset: {0} | Ratio: {1} | Threshold: {2:.6f} | Precision: {3:.4f} | Recall: {4:.4f} | F1: {5:.4f} | ROC-AUC: {6:.4f}".format(
                adj_results["dataset"],
                adj_results["ratio"],
                adj_results["threshold"],
                adj_results["precision"],
                adj_results["recall"],
                adj_results["f1"],
                adj_results["roc_auc"] if not np.isnan(adj_results["roc_auc"]) else float("nan")
            )
        )

        return {
            "raw": raw_results,
            "adjusted": adj_results,
            "val_scores": np.asarray(val_scores),
            "test_scores": np.asarray(test_scores),
            "test_labels": np.asarray(test_labels),
            "raw_pred": np.asarray(raw_pred),
            "adjusted_pred": np.asarray(adj_pred),
        }
