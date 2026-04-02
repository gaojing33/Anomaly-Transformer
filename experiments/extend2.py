# experiments/extend2.py

import numpy as np
import pandas as pd

from utils.evaluation import get_anomaly_ratio


def sample_train_subset(X_train, fraction, random_state=6):
    """
    Randomly sample a subset of the training data.

    Args:
        X_train: training set
        fraction: float in (0, 1]
        random_state: seed

    Returns:
        X_train_sub
    """
    X_train = np.asarray(X_train)

    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    n_train = len(X_train)
    n_sub = max(1, int(n_train * fraction))

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_train, size=n_sub, replace=False)

    return X_train[indices]


def run_extend2_for_one_dataset(
    dataset_name,
    X_train,
    X_val,
    X_test,
    y_test,
    model_fn,
    model_name="OurModel",
    train_fractions=(0.2, 0.5, 1.0),
    use_adjustment=True,
    random_state=6,
    **model_kwargs,
):
    """
    Run reduced-train-data robustness experiment for one dataset.

    model_fn should follow unified interface:
        results, test_pred, val_scores, test_scores = model_fn(
            X_train_sub, X_val, X_test,
            dataset_name=dataset_name,
            y_test=y_test,
            use_adjustment=use_adjustment,
            ratio=paper_ratio,
            **model_kwargs
        )
    """
    rows = []

    paper_ratio = get_anomaly_ratio(dataset_name)

    X_train = np.asarray(X_train)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X_train))
    n_total = len(X_train)

    for frac in train_fractions:
        n_sub = max(1, int(n_total * frac))
        indices = perm[:n_sub]
        X_train_sub = X_train[indices]

        results, _, _, _ = model_fn(
            X_train_sub,
            X_val,
            X_test,
            dataset_name=dataset_name,
            y_test=y_test,
            use_adjustment=use_adjustment,
            ratio=paper_ratio,
            **model_kwargs,
        )

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "train_fraction": frac,
            "n_train_used": len(X_train_sub),
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "roc_auc": results["roc_auc"],
            "ratio": results["ratio"],
            "threshold": results["threshold"],
            "use_adjustment": results["use_adjustment"],
        }
        rows.append(row)

    return rows


def build_extend2_table_for_our_model(
    dataset_list,
    data_dict,
    model_fn,
    model_name="OurModel",
    train_fractions=(0.2, 0.5, 1.0),
    use_adjustment=True,
    random_state=6,
    **model_kwargs,
):
    """
    data_dict example:
    {
        dataset_name: {
            "X_train": ...,
            "X_val": ...,
            "X_test": ...,
            "y_test": ...
        }
    }
    """
    all_rows = []

    for dataset_name in dataset_list:
        data = data_dict[dataset_name]

        rows = run_extend2_for_one_dataset(
            dataset_name=dataset_name,
            X_train=data["X_train"],
            X_val=data["X_val"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            model_fn=model_fn,
            model_name=model_name,
            train_fractions=train_fractions,
            use_adjustment=use_adjustment,
            random_state=random_state,
            **model_kwargs,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    keep_cols = [
        "dataset",
        "model",
        "train_fraction",
        "n_train_used",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "ratio",
        "threshold",
        "use_adjustment",
    ]
    df = df[keep_cols]

    return df


def get_extend2_subtable(df_extend2, dataset_name):
    """
    Return a compact subtable for one dataset.
    """
    sub_df = df_extend2[
        df_extend2["dataset"] == dataset_name
    ][[
        "train_fraction",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]].reset_index(drop=True)

    return sub_df