import numpy as np
import pandas as pd

from experiments import (
    run_isolation_forest_baseline,
    run_lof_baseline,
    run_our_model,
    build_results_table,
    build_extend1_table_for_our_model,
    get_extend1_subtable,
    build_extend2_table_for_our_model,
    get_extend2_subtable,
)

# =========================================================
# Example data format
# Each X should already be windowed:
#   X.shape = (n_samples, win_size, n_features)
# Each y_test should be aligned with X_test windows:
#   y_test.shape = (n_samples, win_size) or flattenable equivalent
# =========================================================

# Replace these dummy arrays with your real processed data
win_size = 100

data_dict = {
    "SMD": {
        "X_train": np.random.randn(200, win_size, 38).astype(np.float32),
        "X_val": np.random.randn(50, win_size, 38).astype(np.float32),
        "X_test": np.random.randn(60, win_size, 38).astype(np.float32),
        "y_test": np.random.randint(0, 2, size=(60, win_size)).astype(np.int32),
    },
    "MSL": {
        "X_train": np.random.randn(200, win_size, 55).astype(np.float32),
        "X_val": np.random.randn(50, win_size, 55).astype(np.float32),
        "X_test": np.random.randn(60, win_size, 55).astype(np.float32),
        "y_test": np.random.randint(0, 2, size=(60, win_size)).astype(np.int32),
    },
}

dataset_list = ["SMD", "MSL"]

# =========================================================
# 1. Baseline comparison table
# =========================================================

model_fns = {
    "IsolationForest": run_isolation_forest_baseline,
    "LOF": run_lof_baseline,
    # OurModel can also be added here, but note it trains from scratch
    # and needs model kwargs below.
}

df_baseline_raw = build_results_table(
    dataset_list=dataset_list,
    data_dict=data_dict,
    model_fns=model_fns,
    use_adjustment=False,
)

df_baseline_adj = build_results_table(
    dataset_list=dataset_list,
    data_dict=data_dict,
    model_fns=model_fns,
    use_adjustment=True,
)

print("=== Baseline Raw ===")
print(df_baseline_raw)

print("=== Baseline Adjusted ===")
print(df_baseline_adj)

# =========================================================
# 2. Run our model on one dataset
# =========================================================

res, pred, val_scores, test_scores = run_our_model(
    X_train=data_dict["SMD"]["X_train"],
    X_val=data_dict["SMD"]["X_val"],
    X_test=data_dict["SMD"]["X_test"],
    dataset_name="SMD",
    y_test=data_dict["SMD"]["y_test"],
    use_adjustment=True,
    input_c=38,
    output_c=38,
    win_size=100,
    batch_size=64,
    num_epochs=2,   # example only
    prior_type="mixture",
    n_mixtures=3,
    discrepancy="jsd",
    lambda_max=3,
    lambda_warmup_epochs=1,
    sigma_smooth_weight=0.01,
)

print("=== Our Model Result ===")
print(res)

# =========================================================
# 3. Extend 1: threshold + adjustment ablation
# =========================================================

score_dict = {
    "SMD": {
        "val_score": val_scores,
        "test_score": test_scores,
        "y_test": data_dict["SMD"]["y_test"],
    }
}

df_extend1 = build_extend1_table_for_our_model(
    dataset_list=["SMD"],
    score_dict=score_dict,
)

print("=== Extend1 Full Table ===")
print(df_extend1.round(4))

print("=== Extend1 SMD Subtable ===")
print(get_extend1_subtable(df_extend1, "SMD").round(4))

# =========================================================
# 4. Extend 2: reduced-train-data robustness
# =========================================================

df_extend2 = build_extend2_table_for_our_model(
    dataset_list=["SMD"],
    data_dict={
        "SMD": data_dict["SMD"]
    },
    model_fn=run_our_model,
    model_name="OurModel",
    train_fractions=(0.2, 0.5, 1.0),
    use_adjustment=True,
    random_state=6,
    input_c=38,
    output_c=38,
    win_size=100,
    batch_size=64,
    num_epochs=2,   # example only
    prior_type="mixture",
    n_mixtures=3,
    discrepancy="jsd",
    lambda_max=3,
    lambda_warmup_epochs=1,
    sigma_smooth_weight=0.01,
)

print("=== Extend2 Full Table ===")
print(df_extend2.round(4))

print("=== Extend2 SMD Subtable ===")
print(get_extend2_subtable(df_extend2, "SMD").round(4))