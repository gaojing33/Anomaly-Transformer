# experiments/extend1.py

import numpy as np
import pandas as pd

from utils.evaluation import evaluate_dataset


def get_extend1_ratios(dataset_name):
    """
    Ratio settings for threshold + adjustment ablation.
    """
    ratio_map = {
        "SWaT": {
            "smaller": 0.0005,
            "paper": 0.001,
            "larger": 0.002,
        },
        "SMD": {
            "smaller": 0.002,
            "paper": 0.005,
            "larger": 0.01,
        },
        "MSL": {
            "smaller": 0.005,
            "paper": 0.01,
            "larger": 0.02,
        },
        "SMAP": {
            "smaller": 0.005,
            "paper": 0.01,
            "larger": 0.02,
        },
        "PSM": {
            "smaller": 0.005,
            "paper": 0.01,
            "larger": 0.02,
        },
    }

    if dataset_name not in ratio_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return ratio_map[dataset_name]


def run_extend1_for_our_model(
    dataset_name,
    val_score,
    test_score,
    y_test
):
    """
    Run threshold + adjustment ablation for one dataset
    using fixed anomaly scores from our model.
    """
    ratio_dict = get_extend1_ratios(dataset_name)

    setting_configs = [
        ("raw + smaller ratio", False, ratio_dict["smaller"]),
        ("raw + paper ratio", False, ratio_dict["paper"]),
        ("raw + larger ratio", False, ratio_dict["larger"]),
        ("adjusted + smaller ratio", True, ratio_dict["smaller"]),
        ("adjusted + paper ratio", True, ratio_dict["paper"]),
        ("adjusted + larger ratio", True, ratio_dict["larger"]),
    ]

    rows = []

    for setting_name, use_adjustment, ratio in setting_configs:
        results, _ = evaluate_dataset(
            dataset_name=dataset_name,
            val_scores=val_score,
            test_scores=test_score,
            test_labels=y_test,
            ratio=ratio,
            use_adjustment=use_adjustment,
        )

        rows.append({
            "dataset": dataset_name,
            "model": "OurModel",
            "setting": setting_name,
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "roc_auc": results["roc_auc"],
            "ratio": results["ratio"],
            "threshold": results["threshold"],
            "use_adjustment": results["use_adjustment"],
        })

    return rows


def build_extend1_table_for_our_model(dataset_list, score_dict):
    """
    score_dict example:
    {
        "SMD": {
            "val_score": ...,
            "test_score": ...,
            "y_test": ...
        },
        ...
    }
    """
    all_rows = []

    for dataset_name in dataset_list:
        dataset_info = score_dict[dataset_name]

        val_score = dataset_info["val_score"]
        test_score = dataset_info["test_score"]
        y_test = dataset_info["y_test"]

        rows = run_extend1_for_our_model(
            dataset_name=dataset_name,
            val_score=val_score,
            test_score=test_score,
            y_test=y_test,
        )
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def get_extend1_subtable(df_extend1, dataset_name):
    """
    Return a compact subtable for one dataset.
    """
    sub_df = df_extend1[
        df_extend1["dataset"] == dataset_name
    ][[
        "setting",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]].reset_index(drop=True)

    return sub_df