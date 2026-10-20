<div align="center">

# 🩺 Good Old ML vs. Generalist Tabular Foundation Models
### Bridging the Benchmarking Gap in Medical Data

**Code and experimental pipelines accompanying the research paper**  
*Good Old ML vs. Generalist Tabular Foundation Models: Bridging the Benchmarking Gap in Medical Data*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Tasks](https://img.shields.io/badge/Tasks-Classification%20%7C%20Regression-00897B)
![Models](https://img.shields.io/badge/Models-Traditional%20ML%20%2B%20TFMs-7B1FA2)
![Evaluation](https://img.shields.io/badge/Evaluation-10--fold%20CV-455A64)

</div>

---

## 🔎 Overview

Do generalist **Tabular Foundation Models (TFMs)** outperform well-established **machine learning (ML)** methods on medical tabular data? And how does the answer change with the **amount of labeled training data**, the **adaptation strategy**, and the **reliability of predicted probabilities**?

This repository contains the experimental code for investigating these questions across medical **classification** and **regression** tasks. It brings together:

- **Predictive benchmarking** of TFMs and conventional ML baselines.
- **Training-data scaling** experiments with five training fractions.
- **Model adaptation** experiments, including inference and fine-tuning strategies where supported.
- **Uncertainty and calibration** analysis for classification probabilities.
- **Statistical comparisons and publication figures** generated from benchmark results.

> **Scope:** These are the research scripts used for the experiments, not a plug-and-play software package. Dataset paths, output locations, model dependencies, and cluster settings must be configured for your environment.

## 🗂️ Repository structure

```text
.
├── Experiments/
│   ├── classification_benchmark.py
│   ├── regression_benchmark.py
│   ├── classification_uncertainty_analysis.py
│   └── statistical_analysis.py
│
├── yaml/
│   ├── dataset_info_classification.yaml
│   └── dataset_info_regression.yaml
│
├── sh/
│   ├── submit_tabular_class.sh
│   ├── submit_tabular_reg.sh
│   └── submit_tabular_unc.sh
│
└── README.md
```

| Folder | What it contains |
|:--|:--|
| **`Experiments/`** | Benchmarking, uncertainty evaluation, and statistical analysis scripts. |
| **`yaml/`** | Dataset names, file paths, target columns, and task metadata. |
| **`sh/`** | Example SLURM submission scripts for running experiments on an HPC cluster. |

## 🧪 Experimental scripts

### 1. `classification_benchmark.py` — Classification performance

Evaluates TFMs and traditional ML classifiers on medical classification datasets.

- **Models:** TabPFN, TabDPT, TabICL, TabICLv2, Mitra, ConTextTab, OrionMSP, OrionBiX, and ML baselines such as logistic regression, random forests, extra trees, boosting methods, SVM, k-NN, and MLP.
- **Evaluation:** 10-fold `StratifiedGroupKFold` cross-validation.
- **Training fractions:** 10%, 20%, 50%, 75%, and 100% of the training partition.
- **TFM strategies:** Inference, base fine-tuning, PEFT, and supported model-specific fine-tuning modes.
- **ML model selection:** Inner `GridSearchCV`, optimized for balanced accuracy.
- **Metrics:** Balanced accuracy, macro precision, macro recall, macro AUROC, and macro AUPRC.

**Current configuration:** `BINARY_ONLY = True`. This script runs the **binary-classification rerun**, including the corrected macro-AUPRC calculation; it does not currently execute the multiclass experiments.

### 2. `regression_benchmark.py` — Regression performance

Evaluates regression methods on medical tabular datasets using:

- **Evaluation:** 10-fold `GroupKFold` cross-validation.
- **Training fractions:** 10%, 20%, 50%, 75%, and 100%.
- **Models:** Tabular Models + Linear regression, Ridge, Elastic Net, random forests, extra trees, gradient boosting, SVR, k-NN, MLP, and optional external boosting libraries.
- **Hyperparameter selection:** Inner `GridSearchCV`, optimized for RMSE.
- **Metrics:** MAE, RMSE, and R².

**Current configuration:** `MODELS = ML_BASELINES`. Although TFM definitions and adaptation options are present in the source, **this version runs only conventional ML regression baselines**. The statistical analysis can also incorporate separately generated regression TFM results.

### 3. `classification_uncertainty_analysis.py` — Confidence & reliability

Goes beyond predictive accuracy to examine **whether models know when they may be wrong**.

This script uses the **100% training fraction** and collects out-of-fold predictions and native probability estimates to evaluate:

| Aspect | Measures |
|:--|:--|
| **Predictive confidence** | Maximum class probability, confidence on correct and incorrect predictions, top-two-class probability margin. |
| **Uncertainty** | Predictive entropy and normalized entropy. |
| **Error detection** | AUROC and AUPRC for identifying incorrect predictions. |
| **Calibration** | Expected Calibration Error (ECE), Brier score, and log loss. |
| **Selective prediction** | Risk–coverage curves and Area Under the Risk–Coverage Curve (AURC). |

It also provides **TFM-versus-ML family comparisons** with dataset-level paired statistical tests and Holm adjustment.

> The analysis uses each fitted model's **native `predict_proba` outputs**. It does **not** apply post-hoc probability calibration.

### 4. `statistical_analysis.py` — Final paper analyses

Processes existing benchmark-result CSV files to generate the final comparative analyses, tables, and figures.

Its workflow includes:

1. Summarizing TFM and ML performance across training fractions.
2. Performing **paired Wilcoxon tests**, **Holm multiple-testing correction**, and model-level comparisons.
3. Exploring relationships between **dataset characteristics** and predictive performance.
4. Exporting publication-oriented **CSV tables**, **PNG figures**, and **text/Markdown reports**.

## 📊 Datasets & evaluation design

The YAML configurations list covers **14 classification tasks** and **5 regression tasks**.

The classification tasks include **9 binary** and **5 multiclass** datasets. The three growth-hormone-response datasets (`Gh_T0`, `Gh_T1`, `Gh_T2`) are combined as **one GH cohort** for relevant inferential comparisons in `statistical_analysis.py`.

**Training-data fractions:** `10% · 20% · 50% · 75% · 100%`  
**Outer cross-validation:** 10 folds  
**Main classification metric:** Balanced accuracy  
**Regression metrics:** MAE, RMSE, R²

Dataset identifiers, targets, and original file locations are documented in:

- [`yaml/dataset_info_classification.yaml`](yaml/dataset_info_classification.yaml)
- [`yaml/dataset_info_regression.yaml`](yaml/dataset_info_regression.yaml)

**Dataset files are not included.** Users must obtain the relevant datasets from their original sources, follow the applicable terms of use, and update the YAML paths. Some research datasets may have restricted access.

## 🚀 Getting started

### 1. Prepare a Python environment

The provided HPC scripts refer to a Conda environment named `tabular311` (Python 3.11). Install the core scientific Python dependencies, the **TabTune** implementation used by the scripts, and any model-specific packages required by the models you choose to run.

For example, the core dependencies include:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy scikit-learn pyyaml matplotlib \
  huggingface_hub torch xgboost lightgbm catboost
```

> **TabTune is a required external dependency** (`tabtune.TabularPipeline.pipeline`). Install a compatible version separately according to its own setup instructions. Some TFMs require additional packages, model weights, GPU resources, or Hugging Face access. This repository does not include a pinned environment/lock file.

### 2. Configure dataset and output paths

Before running any benchmark:

1. Update each dataset's **`path`** in the appropriate file under `yaml/`.
2. Update **`YAML_PATH`** in the relevant Python scripts to point to those YAML files.
3. Update **`OUTPUT_DIR`** in the Python scripts to a writable destination.
4. Review the model lists and active switches (`BINARY_ONLY`, `MODELS`, embedding options) for the experiments you intend to reproduce.

The committed scripts contain **absolute paths from the original HPC environment**. They will not run unchanged on another machine.

If gated models are used, provide a valid Hugging Face token through the environment rather than committing it to Git:

```bash
export HF_TOKEN="your_token_here"
```

### 3. Run the experiments

From the repository root, after updating the paths:

```bash
# Classification: binary benchmark
python Experiments/classification_benchmark.py

# Regression: currently ML baselines only
python Experiments/regression_benchmark.py

# Classification: uncertainty and calibration at 100% training
python Experiments/classification_uncertainty_analysis.py
```

The benchmark scripts export per-dataset and combined **fold-level** and **summary** CSV files to their configured output directories, along with execution logs. The uncertainty script additionally exports probability-level analyses and related tables/figures.

### 4. Reproduce the final statistical analysis

Once **all required input CSVs** are available, run:

```bash
python Experiments/statistical_analysis.py \
  --old-class /path/to/original_classification_results.csv \
  --fixed-binary /path/to/corrected_binary_results.csv \
  --reg /path/to/combined_regression_results.csv \
  --class-char /path/to/dataset_characteristics_class.csv \
  --reg-char /path/to/dataset_characteristics_reg.csv \
  --out-dir ./results/final_analysis
```

The final analysis writes files such as:

```text
results/final_analysis/
├── classification_final_merged.csv
├── old_vs_fixed_binary_audit_rows.csv
├── old_vs_fixed_binary_audit_summary.csv
├── old_vs_final_best_BA_comparison.csv
├── tables/
├── figures/
├── analysis_report.txt
├── manuscript_update_notes.md
└── analysis_metadata.json
```

> **Note:** The `--old-class` input must include the original multiclass tasks. The `--fixed-binary` input contains the nine corrected binary tasks. The final regression input must contain the TFM and ML results needed for the intended comparison. The command above shows the expected **interfaces**, not bundled files.

## 🖥️ Running on an HPC cluster (SLURM)

Example job scripts are provided in [`sh/`](sh/):

```bash
sbatch sh/submit_tabular_class.sh
sbatch sh/submit_tabular_reg.sh
sbatch sh/submit_tabular_unc.sh
```

Before submitting, edit the `.sh` files to match your cluster's **account/QoS**, **Conda environment**, **working directory**, **log-output path**, and **GPU/resource limits**. The included scripts contain site-specific placeholders and absolute paths (including `cd /Experiments`) and must be adapted before use. Do not store authentication tokens in submission scripts.

## ♻️ Reproducibility notes

- The experiments use a fixed random seed (`42`) in the benchmark scripts.
- Outer cross-validation separates groups when group identifiers are available; otherwise, each row receives its own group identifier.
- The provided source contains additional preprocessing and embedding options that are **disabled by default**.
- **Binary and multiclass classification results have different provenance** in the final statistical pipeline: a corrected binary rerun and retained earlier multiclass results.
- Running all benchmarks may require considerable compute, particularly for foundation-model fine-tuning and repeated hyperparameter searches.
- For exact paper-level reproduction, retain the **historical result CSVs**, **dataset-descriptor CSVs**, and software/model versions corresponding to the original experiments.

## 📄 Reference

If this code supports your research, please cite:

**Good Old ML vs. Generalist Tabular Foundation Models: Bridging the Benchmarking Gap in Medical Data**

*Bibliographic details and DOI can be added here when available.*

---

<div align="center">

**Understanding when foundation models help — and when good old ML is still competitive.** 🧬

</div>
