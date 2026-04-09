# Anomaly-Transformer (Improvement3)

This repository is an enhanced implementation of **Anomaly Transformer** for  **unsupervised multivariate time-series anomaly detection** , built on top of the original THUML Anomaly-Transformer codebase. The current `improvement3` branch keeps the original training/testing pipeline while adding a more flexible prior design, a more robust optimization setup, unified evaluation utilities, experiment wrappers, and a new  **cross-validation module for selecting the number of mixture components in the Gaussian Mixture prior** .

## 1. What is new in this branch

Compared with the original implementation, this branch supports:

* **Gaussian prior** and **Gaussian Mixture prior**
* **JSD / KL** discrepancy options
* **Lambda warm-up**
* **Structured sigma regularization**
* A unified evaluation protocol based on  **validation-score threshold selection** , optional  **segment adjustment** , and **Precision / Recall / F1 / ROC-AUC**
* Extended experiment modules under `experiments/`
* A new **mixture-prior cross-validation workflow** through `experiments/cv_mixture.py` and `main.py --mode cv`

## 2. Repository structure

```text
ANOMALY-TRANSFORMER/
│
├── checkpoints/
├── data_factory/
│   ├── __init__.py
│   └── data_loader.py
│
├── dataset/
│   └── FallingPeople/
│
├── experiments/
│   ├── __init__.py
│   ├── baselines.py
│   ├── cv_mixture.py
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
├── scripts/
├── utils/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── logger.py
│   └── utils.py
│
├── check_dml.py
├── main.py
├── solver.py
└── README.md
```

This layout is consistent with the current `improvement3` branch, including the `dataset/FallingPeople` folder and the new `experiments/cv_mixture.py` module.

## 3. File overview

### `main.py`

Main CLI entry point. It supports:

* `--mode train`
* `--mode test`
* `--mode cv`

It also exposes the enhanced prior and CV-related arguments, including `--prior_type`, `--n_mixtures`, `--discrepancy`, `--lambda_max`, `--lambda_warmup_epochs`, `--sigma_smooth_weight`, `--sigma_cross_layer_weight`, `--cv_n_splits`, `--cv_split_mode`, `--cv_metric`, `--mixture_grid`, `--cv_save_path`, and `--cv_step`.

### `solver.py`

Core training/testing pipeline used by the original file-based workflow. It is still the main implementation for model training and inference in `train` and `test` mode.

### `data_factory/data_loader.py`

Builds dataset loaders for the main pipeline. This is the source of the file-based dataloading logic reused by both the original training/testing flow and the new CV wrapper flow.

### `model/`

Contains the Anomaly Transformer architecture:

* `AnomalyTransformer.py`: overall model definition
* `attn.py`: attention-related implementation
* `embed.py`: embedding layers

### `utils/evaluation.py`

Unified evaluation module for validation-based threshold selection and downstream metrics such as Precision, Recall, F1, and ROC-AUC. The current branch README also highlights this unified evaluation protocol.

### `experiments/wrappers.py`

Provides a unified wrapper that allows the model to be trained and evaluated directly from already-windowed arrays. This is the bridge that makes experiment modules and CV reusable without rewriting the whole solver logic.

### `experiments/cv_mixture.py`

Implements cross-validation for selecting the best **number of Gaussian mixture components** (`n_mixtures`) for the mixture prior. It supports:

* blocked or forward split
* configurable CV metric
* candidate grid through `--mixture_grid`
* JSON output through `--cv_save_path`

### `experiments/baselines.py`, `extend1.py`, `extend2.py`, `example_usage.py`

These files provide additional experiment workflows such as baseline comparison, ablation-style extensions, and example usage.

## 4. Supported modes

### Train

Trains the model using the standard file-based pipeline.

### Test

Loads the trained checkpoint and evaluates the model.

### CV

Runs cross-validation over the candidate values in `--mixture_grid` to select the best `n_mixtures` for the mixture prior. This mode is available directly from `main.py`.

## 5. Key arguments

### Core training arguments

* `--lr`
* `--num_epochs`
* `--batch_size`
* `--win_size`
* `--input_c`
* `--output_c`
* `--dataset`
* `--data_path`
* `--model_save_path`

### Prior / optimization arguments

* `--prior_type {gaussian, mixture}`
* `--n_mixtures`
* `--normalize_prior`
* `--sigma_activation`
* `--sigma_min`
* `--discrepancy {jsd, kl}`
* `--lambda_max`
* `--lambda_warmup_epochs`
* `--sigma_smooth_weight`
* `--sigma_cross_layer_weight`

### Cross-validation arguments

* `--cv_n_splits`
* `--cv_split_mode {blocked, forward}`
* `--cv_min_train_ratio`
* `--cv_metric {f1, roc_auc, precision, recall}`
* `--cv_use_adjustment`
* `--cv_ratio`
* `--mixture_grid`
* `--cv_save_path`
* `--cv_step`

## 6. Environment setup

Create and activate a Python environment first, then install the dependencies required by the project. At minimum, the current code imports and uses **PyTorch** and  **NumPy** , and the experiment modules also rely on the standard scientific Python stack. After installation, verify that your editor and terminal are using the same interpreter. This is especially important in VS Code so that Pylance resolves `torch` and `numpy` correctly. The current codebase clearly imports `numpy`, `torch`, `torch.nn`, and `torch.utils.data` in the main training and wrapper modules.

A typical workflow is:

```bash
python -m venv .venv
# activate the environment
pip install torch numpy
```

You may also need to install any other packages used elsewhere in your local workflow.

## 7. Data format

This repository supports the original file-based dataset loading pipeline through `data_factory/data_loader.py`, and the current branch already includes a `dataset/FallingPeople/` folder. The exact file naming and split structure should follow what `data_loader.py` expects for the chosen dataset. For CV and array-based experiments, the wrapper path assumes **already-windowed arrays** with shape:

* `X`: `(n_samples, win_size, n_features)`
* `y`: `(n_samples, win_size)` or another compatible label shape that can be reshaped into that form.

## 8. Example commands

### 8.1 Train

```bash
python main.py \
  --mode train \
  --dataset FallingPeople \
  --data_path ./dataset/FallingPeople \
  --input_c 7 \
  --output_c 7 \
  --win_size 100 \
  --num_epochs 3 \
  --batch_size 256 \
  --lr 1e-4 \
  --k 3 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.0 \
  --activation gelu \
  --output_attention true \
  --prior_type mixture \
  --n_mixtures 3 \
  --normalize_prior true \
  --sigma_activation softplus \
  --sigma_min 1e-4 \
  --discrepancy jsd \
  --lambda_max 3 \
  --lambda_warmup_epochs 2 \
  --sigma_smooth_weight 0.01 \
  --sigma_cross_layer_weight 0.0 \
  --model_save_path checkpoints
```

### 8.2 Test

```bash
python main.py \
  --mode test \
  --dataset FallingPeople \
  --data_path ./dataset/FallingPeople \
  --input_c 7 \
  --output_c 7 \
  --win_size 100 \
  --batch_size 256 \
  --lr 1e-4 \
  --k 3 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.0 \
  --activation gelu \
  --output_attention true \
  --prior_type mixture \
  --n_mixtures 3 \
  --normalize_prior true \
  --sigma_activation softplus \
  --sigma_min 1e-4 \
  --discrepancy jsd \
  --lambda_max 3 \
  --lambda_warmup_epochs 2 \
  --sigma_smooth_weight 0.01 \
  --sigma_cross_layer_weight 0.0 \
  --model_save_path checkpoints
```

### 8.3 Cross-validation for `n_mixtures`

```bash
python main.py \
  --mode cv \
  --dataset FallingPeople \
  --data_path ./dataset/FallingPeople \
  --input_c 7 \
  --output_c 7 \
  --win_size 100 \
  --num_epochs 3 \
  --batch_size 256 \
  --lr 1e-4 \
  --k 3 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.0 \
  --activation gelu \
  --output_attention true \
  --prior_type mixture \
  --normalize_prior true \
  --sigma_activation softplus \
  --sigma_min 1e-4 \
  --discrepancy jsd \
  --lambda_max 3 \
  --lambda_warmup_epochs 2 \
  --sigma_smooth_weight 0.01 \
  --sigma_cross_layer_weight 0.0 \
  --model_save_path checkpoints \
  --mixture_grid 1,2,3,4,5 \
  --cv_n_splits 3 \
  --cv_split_mode blocked \
  --cv_metric f1 \
  --cv_save_path cv_fallingpeople.json
```

These commands are aligned with the argument names currently exposed in `main.py`.

## 9. Smoke test recommendation

Before running full experiments, run a small smoke test:

1. `train` with `num_epochs=1`
2. `test` on the saved checkpoint
3. `cv` with a tiny grid such as `--mixture_grid 1,2` and `--cv_n_splits 2`

This confirms that:

* the original train/test path still works
* the new CV path works
* dataset loading, checkpoint saving, and evaluation are all wired correctly

## 10. Notes on evaluation and CV

The current branch uses a unified evaluation protocol centered on validation-score thresholding and optional adjustment. The new CV workflow is specifically designed to search over `n_mixtures` for the  **mixture prior** . In practice, you should keep the model architecture and optimization settings fixed while searching only over `n_mixtures`, so the comparison isolates the effect of the prior complexity.

## 11. Suggested workflow

A practical workflow for this branch is:

1. Run smoke tests on `train`, `test`, and `cv`
2. Use `--mode cv` to choose the best `n_mixtures`
3. Re-train the model with that selected `n_mixtures`
4. Run final `test`
5. Use the files under `experiments/` for ablations, baselines, and robustness studies

## 12. Acknowledgement

This repository is forked from `thuml/Anomaly-Transformer` and extends it with additional prior designs, evaluation utilities, experiment wrappers, and mixture-prior cross-validation support.
