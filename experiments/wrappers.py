# experiments/wrappers.py

import os
import shutil
import tempfile
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from solver import Solver
from utils.evaluation import evaluate_dataset


class ArrayWindowDataset(Dataset):
    """
    Dataset wrapper for already-windowed data.

    X shape:
        (n_samples, window, n_features)

    y shape:
        - None
        - (n_samples, window)
        - (n_samples, window, 1)
        - (n_samples * window,)  -> will be reshaped outside before use
    """
    def __init__(self, X, y=None):
        self.X = np.asarray(X, dtype=np.float32)

        if self.X.ndim != 3:
            raise ValueError(
                f"Expected X to have shape (n_samples, window, n_features), got {self.X.shape}"
            )

        n_samples, win_size, _ = self.X.shape

        if y is None:
            self.y = np.zeros((n_samples, win_size), dtype=np.float32)
        else:
            y = np.asarray(y)

            if y.ndim == 1:
                if y.shape[0] != n_samples * win_size:
                    raise ValueError(
                        f"1D y has incompatible length {y.shape[0]}, expected {n_samples * win_size}"
                    )
                y = y.reshape(n_samples, win_size)

            elif y.ndim == 2:
                if y.shape != (n_samples, win_size):
                    raise ValueError(
                        f"2D y has shape {y.shape}, expected {(n_samples, win_size)}"
                    )

            elif y.ndim == 3:
                if y.shape[0] != n_samples or y.shape[1] != win_size:
                    raise ValueError(
                        f"3D y has shape {y.shape}, expected first two dims {(n_samples, win_size)}"
                    )
                if y.shape[2] == 1:
                    y = y.squeeze(-1)
                else:
                    raise ValueError(
                        f"3D y has unsupported last dimension {y.shape[2]}, expected 1"
                    )
            else:
                raise ValueError(f"Unsupported y shape: {y.shape}")

            self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _make_loader(X, y=None, batch_size=256, shuffle=False):
    dataset = ArrayWindowDataset(X, y)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def _init_solver_from_config(config):
    """
    Create a Solver instance without going through the original __init__,
    because the original __init__ assumes file-based datasets.
    """
    solver = Solver.__new__(Solver)
    solver.__dict__.update(Solver.DEFAULTS, **config)

    solver.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solver.build_model()
    solver.criterion = nn.MSELoss()

    if solver.lambda_max is None:
        solver.lambda_max = solver.k

    return solver


def _default_anormly_ratio_from_dataset(dataset_name):
    """
    Legacy compatibility parameter only.
    It does NOT control final threshold selection in evaluation.py.
    """
    mapping = {
        "SMD": 0.5,
        "MSL": 1.0,
        "SMAP": 1.0,
        "PSM": 1.0,
        "SWaT": 0.1,
    }
    return mapping.get(dataset_name, 1.0)


def _build_solver_config(
    dataset_name,
    input_c,
    output_c,
    model_save_path,
    batch_size=256,
    win_size=100,
    lr=1e-4,
    num_epochs=10,
    k=3.0,
    e_layers=3,
    d_model=512,
    n_heads=8,
    d_ff=512,
    dropout=0.0,
    activation="gelu",
    output_attention=True,
    prior_type="gaussian",
    n_mixtures=1,
    normalize_prior=True,
    sigma_activation="softplus",
    sigma_min=1e-4,
    discrepancy="jsd",
    lambda_max=None,
    lambda_warmup_epochs=0,
    sigma_smooth_weight=0.0,
    sigma_cross_layer_weight=0.0,
    anormly_ratio=None,
):
    if anormly_ratio is None:
        anormly_ratio = _default_anormly_ratio_from_dataset(dataset_name)

    return {
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "mode": "train",
        "k": k,
        "win_size": win_size,
        "input_c": input_c,
        "output_c": output_c,
        "anormly_ratio": anormly_ratio,  # legacy compatibility only
        "pretrained_model": None,
        "dataset": dataset_name,
        "data_path": "",  # unused in array-based wrapper
        "model_save_path": model_save_path,
        "e_layers": e_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "activation": activation,
        "output_attention": output_attention,
        "prior_type": prior_type,
        "n_mixtures": n_mixtures,
        "normalize_prior": normalize_prior,
        "sigma_activation": sigma_activation,
        "sigma_min": sigma_min,
        "discrepancy": discrepancy,
        "lambda_max": lambda_max,
        "lambda_warmup_epochs": lambda_warmup_epochs,
        "sigma_smooth_weight": sigma_smooth_weight,
        "sigma_cross_layer_weight": sigma_cross_layer_weight,
    }


def run_our_model(
    X_train,
    X_val,
    X_test,
    dataset_name,
    y_test,
    use_adjustment=False,
    ratio=None,
    cleanup=True,
    **kwargs,
):
    """
    Unified wrapper for our model that REALLY trains on the provided arrays.

    This function:
    1. builds array-based dataloaders from X_train / X_val / X_test
    2. trains a fresh Solver model on X_train
    3. computes val_scores on X_val
    4. computes test_scores on X_test
    5. evaluates using the shared protocol in utils/evaluation.py

    Required assumptions:
    - X_train / X_val / X_test are already windowed arrays with shape
      (n_samples, win_size, n_features)
    - y_test is window-level labels, shape compatible with X_test
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "run_our_model expects X_train, X_val, X_test to be windowed arrays "
            "with shape (n_samples, win_size, n_features)"
        )

    win_size = kwargs.get("win_size", X_train.shape[1])
    input_c = kwargs.get("input_c", X_train.shape[2])
    output_c = kwargs.get("output_c", X_train.shape[2])
    batch_size = kwargs.get("batch_size", 256)

    if X_train.shape[1] != win_size or X_val.shape[1] != win_size or X_test.shape[1] != win_size:
        raise ValueError("All input arrays must share the same window size.")

    if X_train.shape[2] != input_c or X_val.shape[2] != input_c or X_test.shape[2] != input_c:
        raise ValueError("All input arrays must share the same feature dimension.")

    # Dummy labels for train / val since training loss does not use them directly
    y_train_dummy = np.zeros((X_train.shape[0], win_size), dtype=np.float32)
    y_val_dummy = np.zeros((X_val.shape[0], win_size), dtype=np.float32)

    train_loader = _make_loader(X_train, y_train_dummy, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val_dummy, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    temp_root = kwargs.get("temp_root", None)
    if temp_root is None:
        temp_dir = tempfile.mkdtemp(prefix=f"anom_wrap_{dataset_name.lower()}_")
    else:
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, f"{dataset_name}_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

    model_save_path = kwargs.get("model_save_path", temp_dir)

    config = _build_solver_config(
        dataset_name=dataset_name,
        input_c=input_c,
        output_c=output_c,
        model_save_path=model_save_path,
        batch_size=batch_size,
        win_size=win_size,
        lr=kwargs.get("lr", 1e-4),
        num_epochs=kwargs.get("num_epochs", 10),
        k=kwargs.get("k", 3.0),
        e_layers=kwargs.get("e_layers", 3),
        d_model=kwargs.get("d_model", 512),
        n_heads=kwargs.get("n_heads", 8),
        d_ff=kwargs.get("d_ff", 512),
        dropout=kwargs.get("dropout", 0.0),
        activation=kwargs.get("activation", "gelu"),
        output_attention=kwargs.get("output_attention", True),
        prior_type=kwargs.get("prior_type", "gaussian"),
        n_mixtures=kwargs.get("n_mixtures", 1),
        normalize_prior=kwargs.get("normalize_prior", True),
        sigma_activation=kwargs.get("sigma_activation", "softplus"),
        sigma_min=kwargs.get("sigma_min", 1e-4),
        discrepancy=kwargs.get("discrepancy", "jsd"),
        lambda_max=kwargs.get("lambda_max", None),
        lambda_warmup_epochs=kwargs.get("lambda_warmup_epochs", 0),
        sigma_smooth_weight=kwargs.get("sigma_smooth_weight", 0.0),
        sigma_cross_layer_weight=kwargs.get("sigma_cross_layer_weight", 0.0),
        anormly_ratio=kwargs.get("anormly_ratio", None),
    )

    solver = _init_solver_from_config(config)

    # Replace file-based loaders with array-based loaders
    solver.train_loader = train_loader
    solver.vali_loader = val_loader
    solver.test_loader = test_loader
    solver.thre_loader = test_loader  # not used in new evaluation path, kept for compatibility

    try:
        # Train from scratch on the provided arrays
        solver.train()

        # Load best checkpoint
        ckpt_path = os.path.join(model_save_path, f"{dataset_name}_checkpoint.pth")
        solver.model.load_state_dict(torch.load(ckpt_path, map_location=solver.device))
        solver.model.eval()

        criterion = nn.MSELoss(reduction="none")
        temperature = 50

        # Produce scores directly from the provided arrays
        val_scores, _ = solver._compute_energy(val_loader, criterion, temperature=temperature)
        test_scores, test_labels = solver._compute_energy(test_loader, criterion, temperature=temperature)

        val_scores = np.asarray(val_scores).reshape(-1)
        test_scores = np.asarray(test_scores).reshape(-1)
        test_labels = np.asarray(test_labels).astype(int).reshape(-1)

        results, test_pred = evaluate_dataset(
            dataset_name=dataset_name,
            val_scores=val_scores,
            test_scores=test_scores,
            test_labels=test_labels,
            use_adjustment=use_adjustment,
            ratio=ratio,
        )

        results["model"] = "OurModel"

        return results, test_pred, val_scores, test_scores

    finally:
        if cleanup and temp_root is None and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)