# utils/evaluation.py

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def get_anomaly_ratio(dataset_name: str) -> float:
    ratio_map = {
        "SWaT": 0.001,
        "SMD": 0.005,
        "MSL": 0.01,
        "SMAP": 0.01,
        "PSM": 0.01,
    }
    if dataset_name not in ratio_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ratio_map[dataset_name]


def get_threshold_from_validation(val_scores: np.ndarray, ratio: float) -> float:
    val_scores = np.asarray(val_scores).reshape(-1)
    threshold = np.quantile(val_scores, 1.0 - ratio)
    return float(threshold)


def predict_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores).reshape(-1)
    return (scores > threshold).astype(int)


def adjust_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred).astype(int).reshape(-1).copy()
    y_true = np.asarray(y_true).astype(int).reshape(-1)

    in_segment = False
    start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            in_segment = True
            start = i

        if y_true[i] == 0 and in_segment:
            end = i
            if np.any(y_pred[start:end] == 1):
                y_pred[start:end] = 1
            in_segment = False

    if in_segment:
        if np.any(y_pred[start:] == 1):
            y_pred[start:] = 1

    return y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    scores = np.asarray(scores).reshape(-1)

    results = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC requires both classes present
    if len(np.unique(y_true)) < 2:
        results["roc_auc"] = np.nan
    else:
        results["roc_auc"] = roc_auc_score(y_true, scores)

    return results


def evaluate_dataset(
    dataset_name: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    use_adjustment: bool = False,
    ratio: float = None
):
    if ratio is None:
        ratio = get_anomaly_ratio(dataset_name)

    threshold = get_threshold_from_validation(val_scores, ratio)

    test_scores = np.asarray(test_scores).reshape(-1)
    test_labels = np.asarray(test_labels).astype(int).reshape(-1)

    test_pred = predict_from_threshold(test_scores, threshold)

    if use_adjustment:
        test_pred = adjust_predictions(test_pred, test_labels)

    results = {
        "dataset": dataset_name,
        "ratio": ratio,
        "threshold": threshold,
        "use_adjustment": use_adjustment,
    }

    results.update(compute_metrics(test_labels, test_pred, test_scores))

    return results, test_pred