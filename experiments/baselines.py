# experiments/baselines.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from utils.evaluation import evaluate_dataset


def flatten_windows(X):
    """
    X: (n_samples, window, n_features)
    -> (n_samples, window * n_features)
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (n_samples, window, n_features), got {X.shape}")
    return X.reshape(X.shape[0], -1)


def run_isolation_forest_baseline(
    X_train,
    X_val,
    X_test,
    dataset_name,
    y_test,
    use_adjustment=False,
    ratio=None,
    random_state=6,
    n_estimators=200,
):
    """
    Run Isolation Forest baseline for anomaly detection.

    Returns:
        results, test_pred, val_scores, test_scores
    """
    X_train_2d = flatten_windows(X_train)
    X_val_2d = flatten_windows(X_val)
    X_test_2d = flatten_windows(X_test)

    y_test = np.asarray(y_test).astype(int).reshape(-1)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_2d)

    # sklearn: larger score_samples => more normal
    # negate so that larger => more anomalous
    val_scores = -model.score_samples(X_val_2d)
    test_scores = -model.score_samples(X_test_2d)

    results, test_pred = evaluate_dataset(
        dataset_name=dataset_name,
        val_scores=val_scores,
        test_scores=test_scores,
        test_labels=y_test,
        use_adjustment=use_adjustment,
        ratio=ratio,
    )

    results["model"] = "IsolationForest"

    return results, test_pred, val_scores, test_scores


def run_lof_baseline(
    X_train,
    X_val,
    X_test,
    dataset_name,
    y_test,
    use_adjustment=False,
    ratio=None,
    n_neighbors=20,
):
    """
    Run LOF baseline for anomaly detection.

    Returns:
        results, test_pred, val_scores, test_scores
    """
    X_train_2d = flatten_windows(X_train)
    X_val_2d = flatten_windows(X_val)
    X_test_2d = flatten_windows(X_test)

    y_test = np.asarray(y_test).astype(int).reshape(-1)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
    )
    model.fit(X_train_2d)

    # sklearn: larger decision_function => more normal
    # negate so that larger => more anomalous
    val_scores = -model.decision_function(X_val_2d)
    test_scores = -model.decision_function(X_test_2d)

    results, test_pred = evaluate_dataset(
        dataset_name=dataset_name,
        val_scores=val_scores,
        test_scores=test_scores,
        test_labels=y_test,
        use_adjustment=use_adjustment,
        ratio=ratio,
    )

    results["model"] = "LOF"

    return results, test_pred, val_scores, test_scores


def build_results_table(dataset_list, data_dict, model_fns, use_adjustment=False):
    """
    model_fns example:
    {
        "IsolationForest": run_isolation_forest_baseline,
        "LOF": run_lof_baseline,
        "OurModel": run_our_model
    }

    data_dict example:
    {
        dataset_name: {
            "X_train": ...,
            "X_val": ...,
            "X_test": ...,
            "y_test": ...
        }
    }

    Each model function should return:
        results, test_pred, val_scores, test_scores
    """
    all_results = []

    for dataset_name in dataset_list:
        data = data_dict[dataset_name]

        for model_name, model_fn in model_fns.items():
            results, _, _, _ = model_fn(
                data["X_train"],
                data["X_val"],
                data["X_test"],
                dataset_name=dataset_name,
                y_test=data["y_test"],
                use_adjustment=use_adjustment,
            )

            results["model"] = model_name
            all_results.append(results)

    df = pd.DataFrame(all_results)

    keep_cols = [
        "dataset",
        "model",
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