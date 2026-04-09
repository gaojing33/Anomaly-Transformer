# Anomaly-Transformer (Enhanced Version)

This repository is an enhanced implementation of **Anomaly Transformer** for unsupervised multivariate time-series anomaly detection.

Compared with the original version, this project supports:

- Gaussian prior and **Mixture Prior**
- **JSD / KL** discrepancy options
- **Lambda warm-up**
- **Structured sigma regularization**
- A unified evaluation protocol based on:
  - validation-score threshold selection
  - optional segment adjustment
  - Precision / Recall / F1 / ROC-AUC
- Extended experiment modules for:
  - traditional baselines
  - threshold/adjustment ablation
  - reduced-train-data robustness

---

## 1. Project Structure

```text
ANOMALY-TRANSFORMER/
│
├── data_factory/
│   ├── __init__.py
│   └── data_loader.py
│
├── experiments/
│   ├── __init__.py
│   ├── baselines.py
│   ├── example_usage.py
│   ├── extend1.py
│   ├── extend2.py
│   └── wrappers.py
│
├── model/
│   ├── __init__.py
│   ├── AnomalyTransformer.py
│   ├── attn.py
│   └── embed.py
│
├── pics/
│
├── scripts/
│   ├── MSL.sh
│   ├── PSM.sh
│   ├── SMAP.sh
│   ├── SMD.sh
│   └── Start.sh
│
├── utils/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── logger.py
│   └── utils.py
│
├── .gitignore
├── change.md
├── LICENSE
├── main.py
├── README.md
├── results.txt
└── solver.py
```

Repository summary:

- **6 main directories**
- **21 files**

---

## 2. What Each Directory and File Does

### `data_factory/`

Responsible for **file-based dataset loading and DataLoader construction** for the main model pipeline.

#### `data_factory/__init__.py`

Package initialization file.

#### `data_factory/data_loader.py`

Loads benchmark datasets and builds dataloaders for:

- `train`
- `val`
- `test`
- `thre`

It currently supports:

- `SMD`
- `MSL`
- `SMAP`
- `PSM`

Main responsibilities:

- read dataset files from disk
- standardize data using `StandardScaler`
- create sliding-window datasets
- return PyTorch `DataLoader` objects used by `solver.py`

---

### `experiments/`

Responsible for **baseline experiments, extended analyses, and unified experiment interfaces**.

#### `experiments/__init__.py`

Exports commonly used experiment functions so they can be imported more easily.

#### `experiments/baselines.py`

Implements traditional baselines:

- Isolation Forest
- Local Outlier Factor (LOF)

Also provides:

- `flatten_windows(...)`
- `build_results_table(...)`

Used for comparing:

- traditional baselines
- your model wrapper
- any future models under a common interface

#### `experiments/example_usage.py`

A compact example showing how to use:

- baseline comparison
- `run_our_model(...)`
- Extend 1
- Extend 2

This file is a usage demo. It is not the main training entry for the core model.

#### `experiments/extend1.py`

Implements **threshold + adjustment ablation**.

It varies:

- threshold ratio
- raw vs adjusted predictions

Typical use:

- compare “paper ratio” vs smaller/larger ratio
- compare raw predictions vs segment-adjusted predictions

#### `experiments/extend2.py`

Implements **reduced-train-data robustness experiments**.

It evaluates model performance when using only part of the training set, for example:

- 20%
- 50%
- 100%

#### `experiments/wrappers.py`

Provides a unified wrapper for **your model** so it can be used like a baseline function.

Most importantly, the wrapper:

- accepts already-windowed arrays
- trains a fresh model on the provided arrays
- generates validation and test scores
- evaluates using the shared protocol in `utils/evaluation.py`

This file is what makes:

- baseline comparison
- extend1
- extend2

work under a consistent experimental interface.

---

### `model/`

Responsible for the **Anomaly Transformer model architecture**.

#### `model/__init__.py`

Package initialization file.

#### `model/AnomalyTransformer.py`

Defines the overall model.

Main responsibilities:

- build the embedding layer
- stack encoder layers
- connect anomaly attention blocks
- return:
  - reconstruction output
  - series associations
  - prior associations
  - sigma parameters
  - mixture weights

#### `model/attn.py`

Implements the core attention mechanism.

Main responsibilities:

- anomaly attention
- series-association computation
- prior-association computation
- support for:
  - original Gaussian prior
  - Gaussian Mixture Prior
- return sigma and mixture information for regularization

This is the main file modified for:

- Mixture Prior
- flexible prior modeling
- structured sigma support

#### `model/embed.py`

Implements:

- token embedding
- positional embedding
- combined data embedding

Input:

- windowed time-series tensor of shape `[B, L, C]`

Output:

- embedded representation of shape `[B, L, d_model]`

---

### `pics/`

Stores figures or images used in documentation or project notes.

This folder does not affect training or evaluation.

---

### `scripts/`

Provides shell scripts for quick execution on supported datasets.

#### `scripts/SMD.sh`

Train and test the model on the `SMD` dataset.

#### `scripts/MSL.sh`

Train and test the model on the `MSL` dataset.

#### `scripts/SMAP.sh`

Train and test the model on the `SMAP` dataset.

#### `scripts/PSM.sh`

Train and test the model on the `PSM` dataset.

#### `scripts/Start.sh`

A combined example script that shows how to launch one dataset quickly and keeps other datasets as commented templates.

---

### `utils/`

Common utility functions and the **shared evaluation protocol**.

#### `utils/__init__.py`

Package initialization file.

#### `utils/evaluation.py`

Implements the unified evaluation protocol.

Main responsibilities:

- get dataset-specific anomaly ratio
- choose threshold from validation scores
- convert scores to binary predictions
- optionally apply segment-level adjustment
- compute:
  - Precision
  - Recall
  - F1
  - ROC-AUC

This file is the **single source of truth** for evaluation.

#### `utils/logger.py`

Logging-related utilities.

#### `utils/utils.py`

General utility helpers used across the project.

---

### Root-level files

#### `.gitignore`

Git ignore rules.

#### `change.md`

Modification notes / change log.

#### `LICENSE`

Open-source license.

#### `main.py`

Main command-line entry.

Used to:

- parse arguments
- create a `Solver`
- run `train` or `test`

#### `README.md`

Project documentation.

#### `results.txt`

Text file for storing or recording results.

#### `solver.py`

Core training / validation / testing pipeline for the main model.

Main responsibilities:

- build the model
- construct optimizer
- train
- validate
- save best checkpoint
- generate anomaly scores
- call `utils/evaluation.py` for final evaluation

---

## 3. Environment and Dependencies

Recommended environment:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn
```

If you use GPU, make sure your PyTorch installation matches your CUDA version.

---

## 4. Two Ways to Use This Project

This repository contains **two different execution styles**:

1. **Main model pipeline**

   - file-based
   - uses dataset files from disk
   - entered through `main.py`, `solver.py`, and `scripts/*.sh`
2. **Experiment pipeline**

   - array-based
   - uses already-windowed NumPy arrays
   - entered through `experiments/baselines.py`, `wrappers.py`, `extend1.py`, `extend2.py`

These two styles serve different purposes.

---

## 5. Input Data Format for the Main Model Pipeline

The main model pipeline uses the loaders in `data_factory/data_loader.py`.

### 5.1 Supported file-based datasets

#### SMD

Expected directory:

```text
dataset/SMD/
├── SMD_train.npy
├── SMD_test.npy
└── SMD_test_label.npy
```

#### MSL

Expected directory:

```text
dataset/MSL/
├── MSL_train.npy
├── MSL_test.npy
└── MSL_test_label.npy
```

#### SMAP

Expected directory:

```text
dataset/SMAP/
├── SMAP_train.npy
├── SMAP_test.npy
└── SMAP_test_label.npy
```

#### PSM

Expected directory:

```text
dataset/PSM/
├── train.csv
├── test.csv
└── test_label.csv
```

---

### 5.2 Required shapes for file-based datasets

#### For `SMD / MSL / SMAP`

- `*_train.npy`: shape `(T_train, num_features)`
- `*_test.npy`: shape `(T_test, num_features)`
- `*_test_label.npy`: shape `(T_test,)` or `(T_test, 1)` or equivalent binary labels

Examples:

- SMD uses `num_features = 38`
- MSL uses `num_features = 55`
- SMAP uses `num_features = 25`

#### For `PSM`

- `train.csv`: first column is ignored, remaining columns are features
- `test.csv`: first column is ignored, remaining columns are features
- `test_label.csv`: first column is ignored, remaining columns are labels

---

### 5.3 What the loader does automatically

The loader in `data_loader.py` will:

- read files from disk
- standardize features using `StandardScaler`
- internally generate sliding windows
- produce dataloaders for:
  - training
  - validation
  - testing
  - threshold/evaluation support

So for the **main model pipeline**, you should provide **raw sequential data**, not pre-windowed arrays.

---

## 6. Input Data Format for the Experiment Pipeline

The files in `experiments/` use **already-windowed arrays**.

### Required shapes

```python
X_train.shape == (n_samples, win_size, n_features)
X_val.shape   == (n_val, win_size, n_features)
X_test.shape  == (n_test, win_size, n_features)
```

`y_test` should be compatible with the test windows, for example:

```python
y_test.shape == (n_test, win_size)
```

or any equivalent shape that can be reshaped consistently.

### Important distinction

- `main.py` / `solver.py` / `data_factory/` expect **raw file-based sequential data**
- `experiments/` expects **already-windowed arrays**

---

## 7. How to Run the Main Model

### 7.1 Quick start with shell scripts

Run one of the prepared scripts:

```bash
bash scripts/SMD.sh
```

or

```bash
bash scripts/MSL.sh
bash scripts/SMAP.sh
bash scripts/PSM.sh
```

You can also use:

```bash
bash scripts/Start.sh
```

---

### 7.2 Run from command line directly

Example for `SMD`:

```bash
python main.py   --mode train   --dataset SMD   --data_path dataset/SMD   --input_c 38   --output_c 38   --win_size 100   --num_epochs 10   --batch_size 256   --lr 1e-4   --prior_type mixture   --n_mixtures 3   --discrepancy jsd   --lambda_max 3   --lambda_warmup_epochs 3   --sigma_smooth_weight 0.01
```

Then test:

```bash
python main.py   --mode test   --dataset SMD   --data_path dataset/SMD   --input_c 38   --output_c 38   --win_size 100   --batch_size 256   --lr 1e-4   --prior_type mixture   --n_mixtures 3   --discrepancy jsd   --lambda_max 3   --lambda_warmup_epochs 3   --sigma_smooth_weight 0.01
```

---

## 8. What Happens During Training and Testing

### Training

When `--mode train` is used:

- model is built in `solver.py`
- dataloaders are created from `data_factory/data_loader.py`
- training runs for the configured number of epochs
- early stopping monitors validation losses
- the best checkpoint is saved

### Testing

When `--mode test` is used:

- the saved checkpoint is loaded
- validation anomaly scores are generated
- test anomaly scores are generated
- final evaluation is performed through `utils/evaluation.py`

This evaluation includes:

- validation-score threshold selection
- raw evaluation
- adjusted evaluation
- Precision / Recall / F1 / ROC-AUC

---

## 9. Where Results and Checkpoints Are Saved

### 9.1 Main model checkpoints

The best model checkpoint is saved to:

```text
{model_save_path}/{dataset}_checkpoint.pth
```

By default:

```text
checkpoints/SMD_checkpoint.pth
checkpoints/MSL_checkpoint.pth
checkpoints/SMAP_checkpoint.pth
checkpoints/PSM_checkpoint.pth
```

### 9.2 Main model printed outputs

During training and testing, results are printed to the terminal.

These include:

- training loss
- validation losses
- threshold
- raw evaluation metrics
- adjusted evaluation metrics

### 9.3 `results.txt`

This file can be used as a manual result log, but the current code does not automatically append every result there unless you modify it to do so.

### 9.4 Experiment outputs

Files in `experiments/` typically return:

- pandas DataFrames
- dictionaries of metrics
- score arrays

These are **not automatically saved** unless you explicitly save them yourself, for example:

```python
df.to_csv("my_results.csv", index=False)
```

or

```python
np.save("val_scores.npy", val_scores)
```

---

## 10. Unified Evaluation Protocol

The final evaluation protocol is defined in:

```text
utils/evaluation.py
```

It does the following:

1. get dataset-specific anomaly ratio
2. choose threshold from validation scores
3. convert test scores into binary predictions
4. optionally apply segment-level adjustment
5. compute:
   - Precision
   - Recall
   - F1
   - ROC-AUC

This file should be treated as the **authoritative evaluation definition** shared across:

- optimization experiments
- reproduction experiments
- baselines
- ablation studies

---

## 11. How to Run Baselines

Baseline code is in:

```text
experiments/baselines.py
```

Implemented baselines:

- Isolation Forest
- LOF

You usually use them together with windowed arrays.

Typical pattern:

```python
from experiments.baselines import run_isolation_forest_baseline

results, pred, val_scores, test_scores = run_isolation_forest_baseline(
    X_train, X_val, X_test,
    dataset_name="SMD",
    y_test=y_test,
    use_adjustment=True
)
```

---

## 12. How to Run Our Model Through the Unified Experiment Interface

Use:

```text
experiments/wrappers.py
```

Main function:

```python
run_our_model(...)
```

This wrapper:

- accepts already-windowed arrays
- trains a fresh model on the provided `X_train`
- evaluates on `X_val` and `X_test`
- returns results under the same interface as the baselines

Example:

```python
from experiments.wrappers import run_our_model

results, pred, val_scores, test_scores = run_our_model(
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    dataset_name="SMD",
    y_test=y_test,
    use_adjustment=True,
    input_c=38,
    output_c=38,
    win_size=100,
    batch_size=64,
    num_epochs=2,
    prior_type="mixture",
    n_mixtures=3,
    discrepancy="jsd",
    lambda_max=3,
    lambda_warmup_epochs=1,
    sigma_smooth_weight=0.01,
)
```

---

## 13. How to Run Extend 1

File:

```text
experiments/extend1.py
```

Purpose:

- ablation on threshold ratio
- comparison between raw and adjusted evaluation

Expected input:

- `val_score`
- `test_score`
- `y_test`

Example:

```python
from experiments.extend1 import build_extend1_table_for_our_model

score_dict = {
    "SMD": {
        "val_score": val_scores,
        "test_score": test_scores,
        "y_test": y_test,
    }
}

df_extend1 = build_extend1_table_for_our_model(
    dataset_list=["SMD"],
    score_dict=score_dict,
)
```

---

## 14. How to Run Extend 2

File:

```text
experiments/extend2.py
```

Purpose:

- evaluate robustness under reduced training data

Expected input:

- `X_train`
- `X_val`
- `X_test`
- `y_test`
- `model_fn`

Example:

```python
from experiments.extend2 import build_extend2_table_for_our_model
from experiments.wrappers import run_our_model

df_extend2 = build_extend2_table_for_our_model(
    dataset_list=["SMD"],
    data_dict={
        "SMD": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_test": y_test,
        }
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
    num_epochs=2,
    prior_type="mixture",
    n_mixtures=3,
    discrepancy="jsd",
    lambda_max=3,
    lambda_warmup_epochs=1,
    sigma_smooth_weight=0.01,
)
```

---

## 15. Example Usage Script

See:

```text
experiments/example_usage.py
```

This file demonstrates how to:

- run baseline comparisons
- run your model
- run extend1
- run extend2

It is a template and should be replaced with real data arrays.

---

## 16. Notes and Practical Tips

- `--anormly_ratio` is retained mainly for backward compatibility in the command-line interface.
- Final threshold selection is governed by `utils/evaluation.py`, not by the old hard-coded test logic.
- For the experiment pipeline, make sure your arrays are already windowed before calling baseline or wrapper functions.
- For first-time debugging, use a small `num_epochs` first to verify the full pipeline.
- If you want experiment outputs saved automatically, add explicit saving lines such as:
  - `df.to_csv(...)`
  - `np.save(...)`

---
