from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.wrappers import run_our_model


@dataclass
class FoldResult:
    fold_id: int
    n_mixtures: int
    metrics: Dict[str, float]


def _as_3d_float32(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n_samples, win_size, n_features), got {X.shape}"
        )
    return X


def _as_labels(y: np.ndarray, n_samples: int, win_size: int, name: str) -> np.ndarray:
    y = np.asarray(y)

    if y.ndim == 1:
        if y.shape[0] == n_samples * win_size:
            return y.reshape(n_samples, win_size)
        raise ValueError(
            f"{name} is 1D with length {y.shape[0]}, expected {n_samples * win_size}"
        )

    if y.ndim == 2:
        if y.shape != (n_samples, win_size):
            raise ValueError(
                f"{name} has shape {y.shape}, expected {(n_samples, win_size)}"
            )
        return y

    if y.ndim == 3:
        if y.shape[:2] != (n_samples, win_size) or y.shape[2] != 1:
            raise ValueError(
                f"{name} has unsupported shape {y.shape}, expected "
                f"({n_samples}, {win_size}, 1)"
            )
        return y.squeeze(-1)

    raise ValueError(f"{name} has unsupported shape: {y.shape}")


def _contiguous_folds(
    n_samples: int,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Blocked CV split for time series windows.

    For each fold:
      - validation = one contiguous block
      - training   = all remaining windows
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > n_samples:
        raise ValueError(
            f"n_splits={n_splits} cannot exceed n_samples={n_samples}"
        )

    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]], axis=0)
        folds.append((train_idx, val_idx))
        current = stop

    return folds


def _forward_expanding_folds(
    n_samples: int,
    n_splits: int,
    min_train_ratio: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Forward-chaining split:
      Fold 1: first chunk train, next chunk val
      Fold 2: larger prefix train, next chunk val
      ...
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not (0.0 < min_train_ratio < 1.0):
        raise ValueError("min_train_ratio must be in (0, 1)")

    min_train = max(1, int(round(n_samples * min_train_ratio)))
    remaining = n_samples - min_train
    if remaining < n_splits:
        raise ValueError(
            f"Not enough samples for forward CV: n_samples={n_samples}, "
            f"min_train={min_train}, n_splits={n_splits}"
        )

    val_block = remaining // n_splits
    extra = remaining % n_splits

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    cursor = min_train

    for i in range(n_splits):
        this_block = val_block + (1 if i < extra else 0)
        val_start = cursor
        val_end = cursor + this_block

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)

        if len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))
        train_end = val_end
        cursor = val_end

    if len(folds) < 2:
        raise ValueError("Forward CV produced fewer than 2 valid folds.")

    return folds


def _summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _pick_best(summary_rows: List[Dict], metric: str, higher_is_better: bool = True) -> Dict:
    if not summary_rows:
        raise ValueError("summary_rows is empty")

    key = f"cv_{metric}_mean"
    if key not in summary_rows[0]:
        raise KeyError(f"Metric summary key not found: {key}")

    return max(summary_rows, key=lambda r: r[key]) if higher_is_better else min(
        summary_rows, key=lambda r: r[key]
    )


def cross_validate_n_mixtures(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_mixture_candidates: Sequence[int],
    n_splits: int = 3,
    split_mode: str = "blocked",
    min_train_ratio: float = 0.5,
    metric: str = "f1",
    higher_is_better: bool = True,
    use_adjustment: bool = False,
    ratio: Optional[float] = None,
    verbose: bool = True,
    run_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Cross-validate the Gaussian mixture prior component count.
    """
    run_kwargs = copy.deepcopy(run_kwargs or {})

    X = _as_3d_float32(X, "X")
    n_samples, win_size, n_features = X.shape
    y = _as_labels(y, n_samples=n_samples, win_size=win_size, name="y")

    if not n_mixture_candidates:
        raise ValueError("n_mixture_candidates is empty")

    candidates = [int(v) for v in n_mixture_candidates]
    if any(v < 1 for v in candidates):
        raise ValueError(f"All n_mixture_candidates must be >= 1, got {candidates}")

    run_kwargs.setdefault("win_size", win_size)
    run_kwargs.setdefault("input_c", n_features)
    run_kwargs.setdefault("output_c", n_features)
    run_kwargs.setdefault("prior_type", "mixture")

    if run_kwargs["prior_type"] != "mixture":
        raise ValueError(
            "cross_validate_n_mixtures is only for mixture prior. "
            "Please set prior_type='mixture'."
        )

    if split_mode == "blocked":
        folds = _contiguous_folds(n_samples=n_samples, n_splits=n_splits)
    elif split_mode == "forward":
        folds = _forward_expanding_folds(
            n_samples=n_samples,
            n_splits=n_splits,
            min_train_ratio=min_train_ratio,
        )
    else:
        raise ValueError(f"Unsupported split_mode={split_mode!r}. Use 'blocked' or 'forward'.")

    fold_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for n_mix in candidates:
        metric_values: List[float] = []

        if verbose:
            print("=" * 80)
            print(f"[CV] Evaluating n_mixtures = {n_mix}")
            print("=" * 80)

        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            fold_kwargs = copy.deepcopy(run_kwargs)
            fold_kwargs["prior_type"] = "mixture"
            fold_kwargs["n_mixtures"] = n_mix

            results, _, _, _ = run_our_model(
                X_train=X_train,
                X_val=X_val,
                X_test=X_val,
                dataset_name=dataset_name,
                y_test=y_val,
                use_adjustment=use_adjustment,
                ratio=ratio,
                **fold_kwargs,
            )

            if metric not in results:
                raise KeyError(
                    f"Requested metric={metric!r} not found in results keys: "
                    f"{list(results.keys())}"
                )

            metric_value = float(results[metric])
            metric_values.append(metric_value)

            row = {
                "fold_id": fold_id,
                "n_mixtures": n_mix,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                **{
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in results.items()
                },
            }
            fold_rows.append(row)

            if verbose:
                print(
                    f"[CV] fold={fold_id}/{len(folds)} | "
                    f"n_mixtures={n_mix} | {metric}={metric_value:.6f}"
                )

        stats = _summarize_metric(metric_values)
        summary_row = {
            "n_mixtures": n_mix,
            f"cv_{metric}_mean": stats["mean"],
            f"cv_{metric}_std": stats["std"],
            f"cv_{metric}_min": stats["min"],
            f"cv_{metric}_max": stats["max"],
            "num_folds": len(metric_values),
        }
        summary_rows.append(summary_row)

        if verbose:
            print(
                f"[CV] n_mixtures={n_mix} | "
                f"mean_{metric}={stats['mean']:.6f} | std={stats['std']:.6f}"
            )

    best_row = _pick_best(
        summary_rows=summary_rows,
        metric=metric,
        higher_is_better=higher_is_better,
    )

    output = {
        "best_n_mixtures": int(best_row["n_mixtures"]),
        "best_row": best_row,
        "summary_rows": summary_rows,
        "fold_rows": fold_rows,
        "config": {
            "dataset_name": dataset_name,
            "n_mixture_candidates": candidates,
            "n_splits": n_splits,
            "split_mode": split_mode,
            "min_train_ratio": min_train_ratio,
            "metric": metric,
            "higher_is_better": higher_is_better,
            "use_adjustment": use_adjustment,
            "ratio": ratio,
            "run_kwargs": run_kwargs,
        },
    }

    if verbose:
        print("=" * 80)
        print("[CV] Best n_mixtures found:")
        print(json.dumps(output["best_row"], indent=2, ensure_ascii=False))
        print("=" * 80)

    return output


def print_cv_report(cv_output: Dict) -> None:
    print("\n[CV] Summary by n_mixtures")
    print("-" * 80)
    for row in cv_output["summary_rows"]:
        print(
            f"n_mixtures={row['n_mixtures']:>2d} | "
            f"mean={row[next(k for k in row if k.endswith('_mean'))]:.6f} | "
            f"std={row[next(k for k in row if k.endswith('_std'))]:.6f}"
        )
    print("-" * 80)
    print(f"Best n_mixtures: {cv_output['best_n_mixtures']}")


def save_cv_output(cv_output: Dict, save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cv_output, f, indent=2, ensure_ascii=False)


def run_cv_from_config(config):
    """
    Entry point used by main.py when mode='cv'.
    """
    from experiments.wrappers import load_dataset_for_cv

    mixture_candidates = [
        int(x.strip()) for x in config.mixture_grid.split(",") if x.strip()
    ]

    X, y = load_dataset_for_cv(
        dataset=config.dataset,
        data_path=config.data_path,
        win_size=config.win_size,
        input_c=config.input_c,
        batch_size=config.batch_size,
        source="test",
        step=getattr(config, "cv_step", None),
    )

    run_kwargs = {
        "lr": config.lr,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "win_size": config.win_size,
        "input_c": config.input_c,
        "output_c": config.output_c,
        "k": config.k,
        "e_layers": config.e_layers,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "dropout": config.dropout,
        "activation": config.activation,
        "output_attention": config.output_attention,
        "prior_type": "mixture",
        "normalize_prior": config.normalize_prior,
        "sigma_activation": config.sigma_activation,
        "sigma_min": config.sigma_min,
        "discrepancy": config.discrepancy,
        "lambda_max": config.lambda_max,
        "lambda_warmup_epochs": config.lambda_warmup_epochs,
        "sigma_smooth_weight": config.sigma_smooth_weight,
        "sigma_cross_layer_weight": config.sigma_cross_layer_weight,
        "model_save_path": config.model_save_path,
        "dataset": config.dataset,
        "data_path": config.data_path,
        "pretrained_model": config.pretrained_model,
        "anormly_ratio": config.anormly_ratio,
    }

    cv_output = cross_validate_n_mixtures(
        X=X,
        y=y,
        dataset_name=config.dataset,
        n_mixture_candidates=mixture_candidates,
        n_splits=config.cv_n_splits,
        split_mode=config.cv_split_mode,
        min_train_ratio=config.cv_min_train_ratio,
        metric=config.cv_metric,
        higher_is_better=True,
        use_adjustment=config.cv_use_adjustment,
        ratio=config.cv_ratio,
        verbose=True,
        run_kwargs=run_kwargs,
    )

    print_cv_report(cv_output)

    if config.cv_save_path:
        save_cv_output(cv_output, config.cv_save_path)
        print(f"[CV] Saved results to: {config.cv_save_path}")

    return cv_output