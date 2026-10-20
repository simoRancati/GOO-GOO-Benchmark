"""
TABULAR FOUNDATION MODELS - CLASSIFICATION UNCERTAINTY ANALYSIS (EXTENDED ML)
===============================================================

Purpose
-------
Dedicated 100%-training-fraction analysis of out-of-fold predictive confidence,
uncertainty, calibration, and failure detection for TFMs vs conventional ML.

The script preserves the original benchmark design:
- same YAML datasets
- same 10-fold StratifiedGroupKFold outer CV
- same preprocessing
- same TFM models and tuning strategies
- same extended ML baselines and inner group-aware GridSearchCV

Differences from the main benchmark:
- ONLY train_fraction = 1.00 is evaluated
- every outer-test prediction is saved (OOF predictions)
- only native predict_proba outputs are accepted for uncertainty analysis
- sample-level uncertainty quantities are computed:
    * predicted confidence = max_k p(y=k|x)
    * chance-normalized confidence
    * predictive entropy and normalized entropy
    * probability assigned to the true class
    * top-1/top-2 probability margin
- dataset/configuration-level metrics are computed:
    * confidence/entropy on errors vs correct predictions
    * error-detection AUROC and AUPRC
    * top-label Expected Calibration Error (ECE)
    * multiclass Brier score
    * log loss
    * risk-coverage curve and AURC
- family summaries use a two-stage aggregation so every model has equal weight:
    1) median across strategies/configurations within dataset x model
    2) median across models within dataset x family (TFM vs ML)
- paired Wilcoxon tests use datasets as independent units; Holm correction is
  performed separately for the primary uncertainty outcomes and supplementary
  calibration outcomes.

Interpretation
--------------
For WRONG predictions:
- lower predicted confidence = better error awareness
- higher normalized entropy = better error awareness
For error detection:
- AUROC/AUPRC higher = uncertainty better separates wrong from correct cases
For selective prediction:
- AURC lower = better risk-confidence ordering (while also reflecting error rate)
For calibration:
- ECE, Brier, and log loss lower = better probabilistic quality

IMPORTANT
---------
This script does NOT post-hoc calibrate the models. It evaluates the probabilities
that the fitted models natively return. This is intentional: the goal is to compare
the reliability of the original model outputs.
"""

import os
import json
import yaml
import logging
import glob
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# =========================
# TabTune
# =========================
from tabtune.TabularPipeline.pipeline import TabularPipeline

# =========================
# Huggingface
# =========================
from huggingface_hub import login
import os

# Check token. Export it before launching, e.g. `export HF_TOKEN=...`.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ Logged into Hugging Face Hub")
else:
    print("⚠️ HF_TOKEN not set - may fail for gated models")

# =========================
# Optional: AE
# =========================
AE_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    AE_AVAILABLE = False

# =========================
# Optional: TabPFN embeddings
# =========================
TABPFN_EMB_AVAILABLE = True
try:
    from tabpfn_extensions import TabPFNClassifier as TabPFNExtClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding
except Exception:
    TABPFN_EMB_AVAILABLE = False


# =========================
# Optional: strong external ML baselines
# =========================
CATBOOST_AVAILABLE = True
try:
    from catboost import CatBoostClassifier
except Exception:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None

XGBOOST_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

LIGHTGBM_AVAILABLE = True
try:
    from lightgbm import LGBMClassifier
except Exception:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None


# ============================================================
# CONFIG
# ============================================================
YAML_PATH = "/blue/salemi/simonerancati/TabProject/dataset_info_classification.yaml"

TASK_NAME = "classification_uncertainty_100pct_extended_ml"
OUTPUT_DIR = f"/blue/salemi/simonerancati/TabProject/Results_Paper/outputs_{TASK_NAME}"
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

RANDOM_STATE = 42
N_SPLITS = 10
TRAIN_FRACTIONS = [1.00]
EXCLUDED_DATASETS = {"student_depression"}
# Inner CV for ML baseline hyperparameter tuning
ML_GRIDSEARCH_CV_SPLITS = 3
ML_GRIDSEARCH_SCORING = "balanced_accuracy"
ML_GRIDSEARCH_N_JOBS = -1

# models from your design
FOUNDATION_MODELS = [
    "TabPFN",
    #"TabPFNv26",
    #"TabPFNv25",
    "TabDPT",
    "TabICL",
    "TabICLv2",
    "Mitra",
    "ContextTab",
    "OrionMSP",
    #"OrionMSPv15",
    "OrionBix",
]

# Extended classical/deep baselines. All of them are tuned with GridSearchCV.
# The switches allow you to reduce runtime without changing the rest of the code.
ENABLE_NEURAL_BASELINES = True
ENABLE_DISTANCE_BASELINES = True
ENABLE_OPTIONAL_BOOSTERS = True

ML_BASELINES = [
    "LogReg",
    "RandomForest",
    "ExtraTrees",
    "HistGB",
    "GradientBoosting",
    "SVM_RBF",
]

if ENABLE_DISTANCE_BASELINES:
    ML_BASELINES.append("KNN")

if ENABLE_NEURAL_BASELINES:
    ML_BASELINES.append("MLP")

if ENABLE_OPTIONAL_BOOSTERS and CATBOOST_AVAILABLE:
    ML_BASELINES.append("CatBoost")

if ENABLE_OPTIONAL_BOOSTERS and XGBOOST_AVAILABLE:
    ML_BASELINES.append("XGBoost")

if ENABLE_OPTIONAL_BOOSTERS and LIGHTGBM_AVAILABLE:
    ML_BASELINES.append("LightGBM")

MODELS = FOUNDATION_MODELS + ML_BASELINES
# Compact but meaningful grids. Expanding every grid substantially increases
# runtime because tuning is repeated within every outer fold and data fraction.
ML_PARAM_GRIDS = {
    "LogReg": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },

    "RandomForest": {
        "n_estimators": [300, 600],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    },

    "ExtraTrees": {
        "n_estimators": [300, 600],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    },

    "HistGB": {
        "max_iter": [200, 400],
        "learning_rate": [0.03, 0.1],
        "max_leaf_nodes": [15, 31],
        "l2_regularization": [0.0, 0.1],
    },

    "GradientBoosting": {
        "n_estimators": [100, 300],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [1, 5],
    },

    "SVM_RBF": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
    },

    "KNN": {
        "n_neighbors": [3, 5, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },

    # Feed-forward neural-network baseline.
    "MLP": {
        "hidden_layer_sizes": [(64,), (128,), (128, 64)],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 1e-2],
    },

    "CatBoost": {
        "iterations": [300, 600],
        "depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg": [3, 7],
    },

    "XGBoost": {
        "n_estimators": [300, 600],
        "max_depth": [3, 6],
        "learning_rate": [0.03, 0.1],
    },

    "LightGBM": {
        "n_estimators": [300, 600],
        "num_leaves": [15, 31],
        "learning_rate": [0.03, 0.1],
        "min_child_samples": [10, 20],
    },
}


# Strategie standard per tutti i foundation models
STANDARD_TUNING_STRATEGIES = [
    "inference",
    "base-ft",
    "peft",
]

# Modi di fine-tuning TabPFN-like per classification
# Nota: turn_by_turn / tbt escluso perché è regression-only
TABPFN_FINETUNE_MODES_BY_MODEL = {
    "TabPFN": [
        "meta-learning",
        "sft",
    ],
    "TabPFNv25": [
        "meta-learning",
        "sft",
    ],
    "TabPFNv26": [
        "meta-learning",
        "sft",
        "native",
    ],
}

TABPFN_FINETUNE_MODELS = list(TABPFN_FINETUNE_MODES_BY_MODEL.keys())

# AE latent spaces to probe
AE_LATENT_DIMS = [8, 16, 32, 64, 128]

# Fairer comparison for TabPFN embeddings
USE_AE_EMBEDDINGS = False
USE_TABPFN_EMBEDDINGS = False
USE_TABPFN_EMB_PCA = False
TABPFN_EMB_PCA_DIMS = [32, 64, 128]
# Set to 0 for the complete analysis. Change it only if you deliberately want to
# start from a later dataset. Completed datasets can also be skipped automatically.
START_FROM_DATASET_IDX = 0
SKIP_COMPLETED_DATASETS = True

# Uncertainty-analysis settings
ECE_N_BINS = 10
RISK_COVERAGE_LEVELS = np.round(np.arange(0.05, 1.001, 0.05), 2).tolist()
OVERCONFIDENCE_THRESHOLD = 0.90

PRIMARY_UNCERTAINTY_METRICS = [
    "median_confidence_errors",
    "median_normalized_entropy_errors",
    "error_detection_auroc_confidence",
    "aurc",
]

SUPPLEMENTARY_CALIBRATION_METRICS = [
    "ece_toplabel",
    "multiclass_brier",
    "log_loss",
]

# preprocessing config
LOW_CARD_MAX_UNIQUE = 20

# AE config
AE_EPOCHS = 20
AE_BATCH_SIZE = 256
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5

# TabPFN embedding config
TABPFN_EMB_N_FOLD = 5
TABPFN_EMB_N_ESTIMATORS = 1

# general
DEVICE = "cuda" if (AE_AVAILABLE and torch.cuda.is_available()) else "cpu"
ID_COLS = ["PATIENT_ID", "ASS_NUMBER"]

# Use near-default TabTune settings: keep minimal explicit params
INFERENCE_PARAMS = {
    "device": DEVICE,
}

BASE_FT_PARAMS = {
    "device": DEVICE,
}

PEFT_FT_PARAMS = {
    "device": DEVICE,
}


# ============================================================
# LOGGING
# ============================================================
def setup_logger():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger(TASK_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger()


def log(msg: str):
    logger.info(msg)


# ============================================================
# YAML / DATASET LOADING
# ============================================================
def load_dataset_configs(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["datasets"]


def read_table(path):
    path = str(path)
    lower_path = path.lower()

    if lower_path.endswith(".csv"):
        # Lettura robusta del CSV: prova a riconoscere automaticamente il separatore
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            for sep in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(path, sep=sep)
                    if df.shape[1] > 1:
                        break
                except Exception:
                    continue
            else:
                raise

        # Se per errore ha letto tutto come una colonna, ritenta con separatori comuni
        if df.shape[1] == 1:
            for sep in [";", ",", "\t", "|"]:
                try:
                    df_try = pd.read_csv(path, sep=sep)
                    if df_try.shape[1] > 1:
                        df = df_try
                        break
                except Exception:
                    pass

    elif lower_path.endswith(".tsv") or lower_path.endswith(".txt"):
        # TSV: separatore tab
        df = pd.read_csv(path, sep="\t")

        # Safety check: se legge tutto come una colonna, prova rilevamento automatico
        if df.shape[1] == 1:
            try:
                df_try = pd.read_csv(path, sep=None, engine="python")
                if df_try.shape[1] > 1:
                    df = df_try
            except Exception:
                pass

    elif lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        df = pd.read_excel(path)

    else:
        raise ValueError(f"Unsupported file extension: {path}")

    # Rimuove colonne indice salvate nel CSV/TSV/Excel
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Converte StringDtype pandas in object per compatibilità con numpy/sklearn
    for c in df.columns:
        dtype_str = str(df[c].dtype)
        if (
            dtype_str.startswith("string")
            or "StringDtype" in dtype_str
            or "string[" in dtype_str
        ):
            df[c] = df[c].astype("object")

    return df


def load_single_dataset(ds_cfg):
    name = ds_cfg["name"]
    path = ds_cfg["path"]
    target_col = ds_cfg["target"]

    df = read_table(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset '{name}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    y = df[target_col].copy()
    X = df.drop(columns=[target_col], errors="ignore").copy()

    # Converte eventuali StringDtype in object anche in X
    for c in X.columns:
        dtype_str = str(X[c].dtype)
        if (
            dtype_str.startswith("string")
            or "StringDtype" in dtype_str
            or "string[" in dtype_str
        ):
            X[c] = X[c].astype("object")

    # Rimuove colonne indice salvate nel CSV
    unnamed_cols = [c for c in X.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        X = X.drop(columns=unnamed_cols, errors="ignore")

    # dataset-specific special case from your YAML
    if name == "myocardial_infarction" and "group_class" in ds_cfg:
        positive_classes = [x.strip() for x in str(ds_cfg["group_class"]).split(",")]
        y = y.astype(str).isin(positive_classes).astype(int)

    else:
        # Caso robusto:
        # - se target è string/object/category -> codici categorici
        # - se target è numerico intero -> lo lascia numerico
        # - se target è numerico non intero -> codici categorici
        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
        ):
            y = y.astype("category").cat.codes

        elif pd.api.types.is_integer_dtype(y):
            y = y.to_numpy()

        else:
            y = pd.Series(y).astype("category").cat.codes

    valid = pd.Series(y).notna().to_numpy()
    X = X.loc[valid].reset_index(drop=True)
    y = np.asarray(y)[valid]

    # groups
    if "PATIENT_ID" in X.columns:
        groups = X["PATIENT_ID"].astype(str).to_numpy()
        X = X.drop(columns=["PATIENT_ID"], errors="ignore")
    else:
        groups = np.arange(len(X)).astype(str)

    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")

    # remove all-NaN columns
    X = X.dropna(axis=1, how="all")

    return X.reset_index(drop=True), y, groups

# ============================================================
# HELPERS
# ============================================================
def is_valid_combination(model_name, preprocessing, tuning_strategy, finetune_mode=None):
    # ML baselines: solo preprocessing numerici, mai auto, solo inference
    if model_name in ML_BASELINES:
        if preprocessing == "auto":
            return False
        if tuning_strategy != "inference":
            return False
        if finetune_mode is not None:
            return False

    # I finetune_mode sono ammessi solo quando tuning_strategy == "finetune"
    if finetune_mode is not None and tuning_strategy != "finetune":
        return False

    # Il tuning_strategy == "finetune" è ammesso solo per TabPFN-like
    if tuning_strategy == "finetune":
        if model_name not in TABPFN_FINETUNE_MODELS:
            return False

        allowed_modes = TABPFN_FINETUNE_MODES_BY_MODEL.get(model_name, [])
        if finetune_mode not in allowed_modes:
            return False

    return True

def build_preprocessing_list():
    #preprocessings = ["auto", "hybrid_ohe_freq"] #Drop auto, in this way
    preprocessings = ["hybrid_ohe_freq"]

    if USE_AE_EMBEDDINGS and AE_AVAILABLE:
        preprocessings.extend([f"ae_embedding_{d}" for d in AE_LATENT_DIMS])

    if USE_TABPFN_EMBEDDINGS and TABPFN_EMB_AVAILABLE:
        preprocessings.append("tabpfn_embedding")
        if USE_TABPFN_EMB_PCA:
            preprocessings.extend([f"tabpfn_embedding_pca_{d}" for d in TABPFN_EMB_PCA_DIMS])

    return preprocessings


def _class_count_dict(y):
    """Small helper used only for readable logs."""
    labels, counts = np.unique(y, return_counts=True)
    return {str(k): int(v) for k, v in zip(labels, counts)}


def _class_prop_dict(y):
    """Small helper used only for readable logs."""
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {str(k): round(float(v / total), 4) for k, v in zip(labels, counts)}


def sample_train_fraction(X, y, groups, frac, random_state=42):
    """
    Group-aware stratified subsampling of the TRAINING portion of each outer fold.

    Why this is better than simple row-level sampling:
    - the external train/test split is already StratifiedGroupKFold;
    - this function keeps the selected training fraction group-aware as well;
    - it tries to preserve the original class proportions inside the current
      training fold for every fraction: 0.10, 0.20, 0.50, 0.75, 1.00.

    Note:
    Exact class proportions are not always mathematically possible when groups
    are indivisible and group sizes/classes are unbalanced. The greedy procedure
    below minimizes deviation from the target class counts while selecting
    whole groups.
    """
    if frac >= 0.999:
        return X.reset_index(drop=True), y, groups

    rng = check_random_state(random_state)
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)

    df_idx = pd.DataFrame({
        "idx": np.arange(len(y)),
        "group": groups,
        "y": y,
    })

    class_labels = np.array(sorted(np.unique(y)))
    global_counts = (
        df_idx["y"]
        .value_counts()
        .reindex(class_labels, fill_value=0)
        .astype(float)
        .to_numpy()
    )
    global_props = global_counts / global_counts.sum()

    target_n = max(len(class_labels), int(round(len(y) * frac)))
    target_counts = global_counts * frac

    # Rows = groups, columns = classes. Each entry is the number of rows of that
    # class in the group. This supports both one-row-per-patient and repeated
    # measures/session-level data.
    group_counts = (
        df_idx.groupby("group")["y"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=class_labels, fill_value=0)
    )
    group_sizes = group_counts.sum(axis=1)

    selected = []
    selected_set = set()
    current_counts = np.zeros(len(class_labels), dtype=float)
    current_n = 0

    def add_group(g):
        nonlocal current_counts, current_n
        if g in selected_set:
            return
        selected.append(g)
        selected_set.add(g)
        current_counts += group_counts.loc[g].to_numpy(dtype=float)
        current_n += int(group_sizes.loc[g])

    # First guarantee that every class present in the training fold is represented
    # in the sampled training fraction, if possible.
    for j, c in enumerate(class_labels):
        candidates = group_counts.index[group_counts[c] > 0].difference(selected_set)
        if len(candidates) == 0:
            continue

        # Prefer groups that are more pure for that class; tie-break randomly.
        purity = group_counts.loc[candidates, c] / group_sizes.loc[candidates]
        best_purity = purity.max()
        best_candidates = purity[purity == best_purity].index.to_numpy()
        add_group(rng.choice(best_candidates))

    def score_after_adding(g):
        new_counts = current_counts + group_counts.loc[g].to_numpy(dtype=float)
        new_n = current_n + int(group_sizes.loc[g])

        # Class-count target error. This preserves proportions at the row level
        # as much as possible while selecting whole groups.
        count_error = np.sum(
            np.abs(new_counts - target_counts) / np.maximum(target_counts, 1.0)
        )

        # Proportion error, useful when the target fraction is small.
        prop_error = np.sum((new_counts / max(new_n, 1) - global_props) ** 2)

        # Size error, to stay close to the requested training fraction.
        size_error = abs(new_n - target_n) / max(target_n, 1)

        # Small random term avoids deterministic tie artifacts.
        return count_error + prop_error + 0.25 * size_error + rng.uniform(0, 1e-9)

    # Greedily add groups until the requested fraction is reached or slightly
    # exceeded. For small/rare datasets, overshooting is preferable to losing a
    # class entirely.
    while current_n < target_n:
        remaining = [g for g in group_counts.index if g not in selected_set]
        if len(remaining) == 0:
            break

        scores = np.array([score_after_adding(g) for g in remaining])
        best_group = remaining[int(np.argmin(scores))]
        add_group(best_group)

    sampled_idx = df_idx.loc[df_idx["group"].isin(selected_set), "idx"].to_numpy()
    sampled_idx = np.sort(sampled_idx)

    # Final safety: if something unexpected happened, fall back to the full train
    # fold rather than producing an invalid single-class training set.
    if len(np.unique(y[sampled_idx])) < len(class_labels):
        return X.reset_index(drop=True), y, groups

    return (
        X.iloc[sampled_idx].reset_index(drop=True),
        y[sampled_idx],
        groups[sampled_idx],
    )


def safe_predict_proba(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.c_[1 - p1, p1]

        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return None


def compute_classification_metrics(y_true, y_pred, y_proba, class_labels):
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }

    result["auroc_ovr_macro"] = np.nan
    result["auprc_macro"] = np.nan

    if y_proba is None:
        return result

    y_proba = np.asarray(y_proba)
    n_classes = len(class_labels)

    try:
        if n_classes == 2:
            # caso binario
            if y_proba.ndim == 2:
                if y_proba.shape[1] == 2:
                    pos_proba = y_proba[:, 1]
                elif y_proba.shape[1] == 1:
                    pos_proba = y_proba[:, 0]
                else:
                    return result
            elif y_proba.ndim == 1:
                pos_proba = y_proba
            else:
                return result

            result["auroc_ovr_macro"] = float(roc_auc_score(y_true, pos_proba))
            result["auprc_macro"] = float(average_precision_score(y_true, pos_proba))

        else:
            # caso multiclasse
            if y_proba.ndim != 2 or y_proba.shape[1] != n_classes:
                return result

            y_bin = label_binarize(y_true, classes=class_labels)

            result["auroc_ovr_macro"] = float(
                roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            )
            result["auprc_macro"] = float(
                average_precision_score(y_bin, y_proba, average="macro")
            )

    except Exception as e:
        log(f"[DEBUG] AUROC/AUPRC failed: {e}")

    return result


def to_dataframe(X_array, prefix="f"):
    X_array = np.asarray(X_array)
    cols = [f"{prefix}_{i}" for i in range(X_array.shape[1])]
    return pd.DataFrame(X_array, columns=cols)


# ============================================================
# CUSTOM PREPROCESSING
# ============================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns.tolist()

        for col in self.columns_:
            freq = X[col].astype(str).value_counts(normalize=True, dropna=False)
            self.freq_maps_[col] = freq.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)

        for col in self.columns_:
            fmap = self.freq_maps_[col]
            out[col] = X[col].astype(str).map(fmap).fillna(0.0)

        return out.to_numpy()


def make_hybrid_preprocessor(X_train_raw: pd.DataFrame):
    cat_cols_all = X_train_raw.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    num_cols = X_train_raw.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    if len(cat_cols_all) > 0:
        card = X_train_raw[cat_cols_all].nunique(dropna=False)
        low_card = card[card <= LOW_CARD_MAX_UNIQUE].index.tolist()
        high_card = card[card > LOW_CARD_MAX_UNIQUE].index.tolist()
    else:
        low_card, high_card = [], []

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "low_cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                low_card,
            ),
            (
                "high_cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("freq", FrequencyEncoder()),
                ]),
                high_card,
            ),
        ],
        remainder="drop",
    )
    return pre, num_cols, low_card, high_card


# ============================================================
# AUTOENCODER
# ============================================================
if AE_AVAILABLE:
    class AutoEncoder(nn.Module):
        def __init__(self, d_in: int, d_lat: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d_in, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, d_lat),
            )
            self.decoder = nn.Sequential(
                nn.Linear(d_lat, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, d_in),
            )

        def forward(self, x):
            z = self.encoder(x)
            xhat = self.decoder(z)
            return xhat, z


def fit_ae_embeddings(Xtr_df, Xte_df, latent_dim):
    if not AE_AVAILABLE:
        raise RuntimeError("PyTorch non disponibile.")

    pre, _, _, _ = make_hybrid_preprocessor(Xtr_df)
    Xtr = pre.fit_transform(Xtr_df)
    Xte = pre.transform(Xte_df)

    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    if hasattr(Xte, "toarray"):
        Xte = Xte.toarray()

    model = AutoEncoder(Xtr.shape[1], latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=AE_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(AE_EPOCHS):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            xhat, _ = model(xb)
            loss = loss_fn(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        log(f"[AE d={latent_dim}] recon_mse={total/len(ds):.6f}")

    model.eval()
    with torch.no_grad():
        Ztr = model.encoder(
            torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
        Zte = model.encoder(
            torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    return to_dataframe(Ztr, f"ae{latent_dim}"), to_dataframe(Zte, f"ae{latent_dim}")


# ============================================================
# TABPFN EMBEDDINGS
# ============================================================
def fit_tabpfn_embeddings(Xtr_df, ytr, Xte_df, pca_dim=None):
    if not TABPFN_EMB_AVAILABLE:
        raise RuntimeError("tabpfn-extensions non disponibile.")

    pre, _, _, _ = make_hybrid_preprocessor(Xtr_df)
    Xtr = pre.fit_transform(Xtr_df)
    Xte = pre.transform(Xte_df)

    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    if hasattr(Xte, "toarray"):
        Xte = Xte.toarray()

    clf = TabPFNExtClassifier(
        n_estimators=TABPFN_EMB_N_ESTIMATORS,
        device=DEVICE,
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
    )

    emb = TabPFNEmbedding(tabpfn_clf=clf, n_fold=TABPFN_EMB_N_FOLD)
    emb.fit(Xtr, ytr)

    Ztr = emb.get_embeddings(Xtr, ytr, Xtr, data_source="train")[0]
    Zte = emb.get_embeddings(Xtr, ytr, Xte, data_source="test")[0]

    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
        Ztr = pca.fit_transform(Ztr)
        Zte = pca.transform(Zte)
        return (
            to_dataframe(Ztr, f"tpfnpca{pca_dim}"),
            to_dataframe(Zte, f"tpfnpca{pca_dim}")
        )

    return to_dataframe(Ztr, "tpfnemb"), to_dataframe(Zte, "tpfnemb")


# ============================================================
# PREPROCESSING ROUTER
# ============================================================
def apply_preprocessing(strategy, Xtr_df, ytr, Xte_df):
    if strategy == "auto":
        return Xtr_df.copy(), Xte_df.copy(), {"embedding_dim": None}

    if strategy == "hybrid_ohe_freq":
        pre, num_cols, low_card, high_card = make_hybrid_preprocessor(Xtr_df)
        Xtr = pre.fit_transform(Xtr_df)
        Xte = pre.transform(Xte_df)

        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray()
        if hasattr(Xte, "toarray"):
            Xte = Xte.toarray()

        meta = {
            "embedding_dim": Xtr.shape[1],
            "num_cols": len(num_cols),
            "low_card_cols": len(low_card),
            "high_card_cols": len(high_card),
        }
        log(
            f"[hybrid] num={len(num_cols)} low_card={len(low_card)} "
            f"high_card={len(high_card)} final_dim={Xtr.shape[1]}"
        )
        return to_dataframe(Xtr, "hyb"), to_dataframe(Xte, "hyb"), meta

    if strategy.startswith("ae_embedding_"):
        latent_dim = int(strategy.split("_")[-1])
        Xtr_p, Xte_p = fit_ae_embeddings(Xtr_df, Xte_df, latent_dim=latent_dim)
        return Xtr_p, Xte_p, {"embedding_dim": latent_dim}

    if strategy == "tabpfn_embedding":
        Xtr_p, Xte_p = fit_tabpfn_embeddings(Xtr_df, ytr, Xte_df, pca_dim=None)
        return Xtr_p, Xte_p, {"embedding_dim": Xtr_p.shape[1]}

    if strategy.startswith("tabpfn_embedding_pca_"):
        pca_dim = int(strategy.split("_")[-1])
        Xtr_p, Xte_p = fit_tabpfn_embeddings(Xtr_df, ytr, Xte_df, pca_dim=pca_dim)
        return Xtr_p, Xte_p, {"embedding_dim": pca_dim}

    raise ValueError(f"Unknown preprocessing strategy: {strategy}")


# ============================================================
# TABTUNE
# ============================================================
def build_tuning_params(strategy, finetune_mode=None):
    if strategy == "inference":
        return deepcopy(INFERENCE_PARAMS)

    if strategy == "base-ft":
        return deepcopy(BASE_FT_PARAMS)

    if strategy == "peft":
        return deepcopy(PEFT_FT_PARAMS)

    if strategy == "finetune":
        params = {
            "device": DEVICE,
        }

        if finetune_mode is not None:
            params["finetune_mode"] = finetune_mode

        return params

    raise ValueError(strategy)


def _validate_native_probabilities(y_proba, n_classes, context=""):
    """
    Validate native probabilities for uncertainty/calibration analysis.

    We deliberately do NOT convert decision_function scores with sigmoid/softmax.
    If a model cannot provide native predict_proba output, the configuration is
    skipped rather than treating arbitrary scores as calibrated probabilities.
    """
    if y_proba is None:
        raise RuntimeError(f"{context}: predict_proba returned None")

    p = np.asarray(y_proba, dtype=float)

    if p.ndim == 1:
        if n_classes != 2:
            raise ValueError(
                f"{context}: 1D predict_proba is only interpretable for binary tasks; "
                f"n_classes={n_classes}"
            )
        p = np.c_[1.0 - p, p]

    if p.ndim != 2:
        raise ValueError(f"{context}: expected 2D probabilities, got shape={p.shape}")

    if p.shape[1] != n_classes:
        raise ValueError(
            f"{context}: probability columns={p.shape[1]} but training classes={n_classes}"
        )

    if not np.all(np.isfinite(p)):
        raise ValueError(f"{context}: predict_proba contains NaN/inf")

    # Allow very small floating-point excursions only.
    tol = 1e-6
    if np.any(p < -tol) or np.any(p > 1.0 + tol):
        raise ValueError(
            f"{context}: predict_proba contains values outside [0, 1] "
            f"(min={p.min()}, max={p.max()})"
        )

    p = np.clip(p, 0.0, 1.0)
    row_sums = p.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError(f"{context}: at least one probability row sums to <= 0")

    # Renormalize tiny numerical deviations, but reject outputs that are clearly
    # not probability distributions.
    if not np.allclose(row_sums, 1.0, atol=1e-4, rtol=1e-4):
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"{context}: rows do not sum to 1 (max deviation={max_dev:.6g}); "
            "not treating these outputs as probabilities"
        )

    p = p / row_sums[:, None]
    return p


def fit_and_eval_tabtune(
    model_name,
    tuning_strategy,
    X_train,
    y_train,
    X_test,
    y_test,
    finetune_mode=None,
):
    tuning_params = build_tuning_params(
        strategy=tuning_strategy,
        finetune_mode=finetune_mode,
    )

    pipeline = TabularPipeline(
        model_name=model_name,
        task_type="classification",
        tuning_strategy=tuning_strategy,
        tuning_params=tuning_params,
    )

    pipeline.fit(X_train, y_train)

    y_pred = np.asarray(pipeline.predict(X_test)).reshape(-1)
    class_labels = np.array(sorted(np.unique(y_train)))

    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError(
            f"TabTune pipeline for {model_name}/{tuning_strategy} has no predict_proba; "
            "uncertainty analysis requires native probabilities"
        )

    y_proba = pipeline.predict_proba(X_test)
    y_proba = _validate_native_probabilities(
        y_proba,
        n_classes=len(class_labels),
        context=f"{model_name}/{tuning_strategy}/{finetune_mode}",
    )

    log(
        f"[DEBUG] model={model_name} strategy={tuning_strategy} "
        f"finetune_mode={finetune_mode} "
        f"train_classes={class_labels} test_classes={np.unique(y_test)} "
        f"y_proba_shape={y_proba.shape}"
    )

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
    )

    return metrics, y_pred, y_proba, class_labels


def build_ml_classifier(model_name):
    """Construct one classification baseline with deterministic settings."""
    if model_name == "LogReg":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,  # GridSearchCV performs the outer parallelization
        )

    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if model_name == "HistGB":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

    if model_name == "SVM_RBF":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "KNN":
        return KNeighborsClassifier()

    if model_name == "MLP":
        return MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )

    if model_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not installed")
        return CatBoostClassifier(
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=RANDOM_STATE,
        )

    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not installed")
        return XGBClassifier(
            eval_metric="mlogloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

    if model_name == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not installed")
        return LGBMClassifier(
            class_weight="balanced",
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            verbosity=-1,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown ML baseline: {model_name}")


def fit_and_eval_ml_baseline(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    groups_train=None,
):
    """
    Fit/tune the extended ML baseline exactly as in the extended benchmark,
    then return native class probabilities for uncertainty analysis.
    """
    model = build_ml_classifier(model_name)
    param_grid = deepcopy(ML_PARAM_GRIDS[model_name])

    # Avoid invalid KNN candidates in very small folds.
    if model_name == "KNN":
        max_neighbors = max(1, len(X_train) - 1)
        valid_neighbors = [
            k for k in param_grid["n_neighbors"] if k <= max_neighbors
        ]
        param_grid["n_neighbors"] = valid_neighbors or [1]
        model.set_params(n_neighbors=min(5, max_neighbors))

    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = int(class_counts.min())
    inner_splits = min(ML_GRIDSEARCH_CV_SPLITS, min_class_count)

    if inner_splits < 2:
        log(
            f"[WARNING] model={model_name} GridSearch skipped: "
            f"not enough samples per class. Fitting default model."
        )
        model.fit(X_train, y_train)
        best_params = {}
        best_score = np.nan
        best_estimator = model

    else:
        inner_cv = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=ML_GRIDSEARCH_SCORING,
            cv=inner_cv,
            n_jobs=ML_GRIDSEARCH_N_JOBS,
            refit=True,
            verbose=0,
            error_score=np.nan,
            return_train_score=False,
        )

        if groups_train is None:
            groups_train = np.arange(len(X_train)).astype(str)

        grid.fit(X_train, y_train, groups=groups_train)

        if not hasattr(grid, "best_estimator_"):
            raise RuntimeError(f"Grid search failed for {model_name}")

        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_

        log(
            f"[GRIDSEARCH] model={model_name} "
            f"best_score={best_score:.4f} "
            f"best_params={best_params}"
        )

    y_pred = np.asarray(best_estimator.predict(X_test)).reshape(-1)

    if not hasattr(best_estimator, "classes_"):
        raise RuntimeError(
            f"ML estimator {model_name} has no classes_ attribute after fitting"
        )
    class_labels = np.asarray(best_estimator.classes_)

    if not hasattr(best_estimator, "predict_proba"):
        raise RuntimeError(
            f"ML estimator {model_name} has no native predict_proba; "
            "uncertainty analysis requires native probabilities"
        )

    y_proba = best_estimator.predict_proba(X_test)
    y_proba = _validate_native_probabilities(
        y_proba,
        n_classes=len(class_labels),
        context=model_name,
    )

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
    )

    metrics["ml_best_cv_score"] = (
        float(best_score) if not pd.isna(best_score) else np.nan
    )
    metrics["ml_best_params"] = json.dumps(best_params)

    return metrics, y_pred, y_proba, class_labels


def get_model_strategy_mode_pairs(model_name):
    """
    Returns a list of (tuning_strategy, finetune_mode) pairs.

    For most foundation models:
        inference, base-ft, peft

    For TabPFN-like models:
        inference, base-ft, peft
        finetune + allowed finetune modes

    For ML baselines:
        inference only
    """

    if model_name in ML_BASELINES:
        return [("inference", None)]

    pairs = [(s, None) for s in STANDARD_TUNING_STRATEGIES]

    if model_name in TABPFN_FINETUNE_MODELS:
        for mode in TABPFN_FINETUNE_MODES_BY_MODEL[model_name]:
            pairs.append(("finetune", mode))

    return pairs


# ============================================================
# UNCERTAINTY / CALIBRATION / SELECTIVE-PREDICTION HELPERS
# ============================================================
def get_model_family(model_name):
    return "TFM" if model_name in FOUNDATION_MODELS else "ML"


def get_strategy_group(model_name, tuning_strategy, finetune_mode=None):
    if model_name in ML_BASELINES:
        return "ML"

    if tuning_strategy == "inference":
        return "TFM In-Context"
    if tuning_strategy == "base-ft":
        return "TFM Base-FT"
    if tuning_strategy == "peft":
        return "TFM PEFT"
    if tuning_strategy == "finetune":
        mode = finetune_mode if finetune_mode is not None else "unspecified"
        return f"TFM Fine-tune ({mode})"

    return f"TFM {tuning_strategy}"


def make_configuration_label(model_name, tuning_strategy, finetune_mode=None):
    if finetune_mode is None or (isinstance(finetune_mode, float) and np.isnan(finetune_mode)):
        return f"{model_name} | {tuning_strategy}"
    return f"{model_name} | {tuning_strategy} | {finetune_mode}"


def compute_sample_uncertainty(y_true, y_pred, y_proba, class_labels):
    """Return sample-level confidence/uncertainty quantities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p = np.asarray(y_proba, dtype=float)
    class_labels = np.asarray(class_labels)

    n, k = p.shape
    if k < 2:
        raise ValueError("Uncertainty analysis requires at least two classes")

    eps = np.finfo(float).eps
    p_clip = np.clip(p, eps, 1.0)

    confidence = np.max(p, axis=1)
    normalized_confidence = (confidence - 1.0 / k) / (1.0 - 1.0 / k)
    normalized_confidence = np.clip(normalized_confidence, 0.0, 1.0)

    entropy = -np.sum(p_clip * np.log(p_clip), axis=1)
    normalized_entropy = entropy / np.log(k)

    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    class_to_pos = {cls: j for j, cls in enumerate(class_labels)}
    true_pos = []
    for yy in y_true:
        if yy not in class_to_pos:
            raise ValueError(
                f"True class {yy!r} is not present in probability class order {class_labels.tolist()}"
            )
        true_pos.append(class_to_pos[yy])
    true_pos = np.asarray(true_pos, dtype=int)
    true_class_probability = p[np.arange(n), true_pos]

    correct = (y_pred == y_true)

    return {
        "correct": correct,
        "error": (~correct).astype(int),
        "predicted_confidence": confidence,
        "normalized_confidence": normalized_confidence,
        "uncertainty_1_minus_confidence": 1.0 - confidence,
        "predictive_entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top1_top2_margin": margin,
        "uncertainty_1_minus_margin": 1.0 - margin,
        "true_class_probability": true_class_probability,
    }


def make_prediction_rows(
    base_meta,
    sample_indices,
    group_ids,
    y_true,
    y_pred,
    y_proba,
    class_labels,
):
    uncertainty = compute_sample_uncertainty(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
    )

    out = pd.DataFrame({
        "sample_index": np.asarray(sample_indices),
        "group_id": np.asarray(group_ids).astype(str),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        **uncertainty,
    })

    for key, value in base_meta.items():
        out[key] = value

    class_order = [x.item() if hasattr(x, "item") else x for x in np.asarray(class_labels)]
    out["probability_class_order"] = json.dumps(class_order)
    out["n_classes"] = len(class_order)

    p = np.asarray(y_proba, dtype=float)
    for j in range(p.shape[1]):
        out[f"prob_class_{j}"] = p[:, j]

    return out


def expected_calibration_error(confidence, correct, n_bins=10):
    """
    Top-label ECE: bins max-class confidence and compares it with empirical
    correctness. Works for binary and multiclass classification.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, edges[1:-1], right=True)

    rows = []
    ece = 0.0
    n = len(confidence)

    for b in range(n_bins):
        mask = bin_ids == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append({
                "bin_id": b,
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "n": 0,
                "mean_confidence": np.nan,
                "empirical_accuracy": np.nan,
                "abs_calibration_gap": np.nan,
            })
            continue

        mean_conf = float(np.mean(confidence[mask]))
        accuracy = float(np.mean(correct[mask]))
        gap = abs(accuracy - mean_conf)
        ece += (n_bin / n) * gap

        rows.append({
            "bin_id": b,
            "bin_lower": edges[b],
            "bin_upper": edges[b + 1],
            "n": n_bin,
            "mean_confidence": mean_conf,
            "empirical_accuracy": accuracy,
            "abs_calibration_gap": gap,
        })

    return float(ece), pd.DataFrame(rows)


def risk_coverage_metrics(errors, confidence, coverage_levels=None):
    """
    Compute selective risk after sorting samples from highest to lowest confidence.

    AURC is computed as the mean selective risk across all attainable prefix
    coverages, matching the common empirical finite-sample definition.
    """
    errors = np.asarray(errors, dtype=float)
    confidence = np.asarray(confidence, dtype=float)

    order = np.argsort(-confidence, kind="mergesort")
    e_sorted = errors[order]
    c_sorted = confidence[order]

    n = len(errors)
    accepted = np.arange(1, n + 1)
    cumulative_errors = np.cumsum(e_sorted)
    selective_risk = cumulative_errors / accepted
    aurc = float(np.mean(selective_risk))

    if coverage_levels is None:
        coverage_levels = RISK_COVERAGE_LEVELS

    rows = []
    for cov in coverage_levels:
        k = max(1, min(n, int(np.ceil(float(cov) * n))))
        rows.append({
            "coverage": k / n,
            "requested_coverage": float(cov),
            "accepted_n": k,
            "risk": float(np.mean(e_sorted[:k])),
            "accuracy_at_coverage": float(1.0 - np.mean(e_sorted[:k])),
            "confidence_threshold": float(c_sorted[k - 1]),
        })

    return aurc, pd.DataFrame(rows)


def _extract_probability_matrix(group_df):
    prob_cols = [c for c in group_df.columns if c.startswith("prob_class_")]
    prob_cols = sorted(prob_cols, key=lambda x: int(x.rsplit("_", 1)[1]))

    # Different datasets may have different numbers of classes, so global CSVs
    # can contain trailing all-NaN probability columns. Keep only columns that
    # actually belong to this configuration.
    prob_cols = [c for c in prob_cols if group_df[c].notna().any()]
    p = group_df[prob_cols].to_numpy(dtype=float)

    class_order = json.loads(group_df["probability_class_order"].iloc[0])
    if p.shape[1] != len(class_order):
        raise ValueError(
            f"Probability matrix has {p.shape[1]} columns but class_order has {len(class_order)}"
        )

    # Convert JSON numeric labels back to numpy-friendly scalars. Strings remain strings.
    class_labels = np.asarray(class_order)
    return p, class_labels


def summarize_configuration_predictions(pred_df):
    """Summarize all 10 OOF test folds for one dataset/model configuration."""
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    errors = pred_df["error"].astype(int).to_numpy()
    correct = pred_df["correct"].astype(bool).to_numpy()
    confidence = pred_df["predicted_confidence"].to_numpy(dtype=float)
    entropy_norm = pred_df["normalized_entropy"].to_numpy(dtype=float)

    p, class_labels = _extract_probability_matrix(pred_df)

    # CSV reload can make numeric labels strings/floats depending on pandas
    # inference. Match y_true to class labels robustly by string representation
    # only when direct matching fails.
    direct_ok = set(np.unique(y_true)).issubset(set(class_labels.tolist()))
    if not direct_ok:
        mapping = {str(c): c for c in class_labels}
        try:
            y_true_for_prob = np.asarray([mapping[str(v)] for v in y_true])
        except KeyError as exc:
            raise ValueError(
                f"Could not align y_true with probability class order: {class_labels.tolist()}"
            ) from exc
    else:
        y_true_for_prob = y_true

    n = len(pred_df)
    n_errors = int(errors.sum())
    n_correct = int(correct.sum())

    wrong_conf = confidence[errors == 1]
    right_conf = confidence[errors == 0]
    wrong_ent = entropy_norm[errors == 1]
    right_ent = entropy_norm[errors == 0]

    error_detection_auroc = np.nan
    error_detection_auprc = np.nan
    if len(np.unique(errors)) == 2:
        uncertainty_score = 1.0 - confidence
        error_detection_auroc = float(roc_auc_score(errors, uncertainty_score))
        error_detection_auprc = float(average_precision_score(errors, uncertainty_score))

    ece, cal_df = expected_calibration_error(
        confidence=confidence,
        correct=correct,
        n_bins=ECE_N_BINS,
    )

    # Generalized multiclass Brier score: mean sum_k (p_k - y_k)^2.
    class_to_pos = {cls: j for j, cls in enumerate(class_labels)}
    y_onehot = np.zeros_like(p, dtype=float)
    for i, yy in enumerate(y_true_for_prob):
        y_onehot[i, class_to_pos[yy]] = 1.0
    multiclass_brier = float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))

    try:
        ll = float(log_loss(y_true_for_prob, p, labels=class_labels))
    except Exception:
        ll = np.nan

    aurc, rc_df = risk_coverage_metrics(
        errors=errors,
        confidence=confidence,
        coverage_levels=RISK_COVERAGE_LEVELS,
    )

    def safe_mean(x):
        return float(np.mean(x)) if len(x) else np.nan

    def safe_median(x):
        return float(np.median(x)) if len(x) else np.nan

    summary = {
        "n_oof_predictions": n,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "error_rate": float(n_errors / n),
        "balanced_accuracy_oof": float(balanced_accuracy_score(y_true, y_pred)),
        "mean_confidence_all": float(np.mean(confidence)),
        "median_confidence_all": float(np.median(confidence)),
        "mean_confidence_errors": safe_mean(wrong_conf),
        "median_confidence_errors": safe_median(wrong_conf),
        "mean_confidence_correct": safe_mean(right_conf),
        "median_confidence_correct": safe_median(right_conf),
        "confidence_correct_minus_error": (
            safe_median(right_conf) - safe_median(wrong_conf)
            if len(right_conf) and len(wrong_conf)
            else np.nan
        ),
        "mean_normalized_entropy_errors": safe_mean(wrong_ent),
        "median_normalized_entropy_errors": safe_median(wrong_ent),
        "mean_normalized_entropy_correct": safe_mean(right_ent),
        "median_normalized_entropy_correct": safe_median(right_ent),
        "error_detection_auroc_confidence": error_detection_auroc,
        "error_detection_auprc_confidence": error_detection_auprc,
        "overconfident_error_fraction_ge_090": (
            float(np.mean(wrong_conf >= OVERCONFIDENCE_THRESHOLD)) if len(wrong_conf) else np.nan
        ),
        "ece_toplabel": ece,
        "multiclass_brier": multiclass_brier,
        "log_loss": ll,
        "aurc": aurc,
    }

    # Useful working points for a clinical selective-prediction interpretation.
    for cov in [0.80, 0.90, 0.95, 1.00]:
        rc_row = rc_df.iloc[(rc_df["requested_coverage"] - cov).abs().argmin()]
        tag = int(round(cov * 100))
        summary[f"risk_at_{tag}pct_coverage"] = float(rc_row["risk"])
        summary[f"accuracy_at_{tag}pct_coverage"] = float(rc_row["accuracy_at_coverage"])

    return summary, cal_df, rc_df


def summarize_dataset_predictions(dataset_pred_df):
    config_cols = [
        "dataset",
        "model_family",
        "strategy_group",
        "model_name",
        "tuning_strategy",
        "finetune_mode",
        "preprocessing",
        "embedding_dim",
        "train_fraction",
        "configuration_label",
    ]

    metric_rows = []
    calibration_rows = []
    rc_rows = []

    for keys, group in dataset_pred_df.groupby(config_cols, dropna=False, sort=False):
        meta = dict(zip(config_cols, keys if isinstance(keys, tuple) else (keys,)))
        summary, cal_df, rc_df = summarize_configuration_predictions(group)
        metric_rows.append({**meta, **summary})

        for c, v in meta.items():
            cal_df[c] = v
            rc_df[c] = v
        calibration_rows.append(cal_df)
        rc_rows.append(rc_df)

    metrics_df = pd.DataFrame(metric_rows)
    calibration_df = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    rc_df = pd.concat(rc_rows, ignore_index=True) if rc_rows else pd.DataFrame()
    return metrics_df, calibration_df, rc_df


def holm_adjust(p_values):
    """Holm step-down family-wise error-rate correction without extra deps."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out

    pv = p[valid]
    order_local = np.argsort(pv)
    sorted_p = pv[order_local]
    m = len(sorted_p)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, val in enumerate(sorted_p):
        candidate = (m - i) * val
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(1.0, running_max)

    adjusted_local = np.empty(m, dtype=float)
    adjusted_local[order_local] = adjusted_sorted
    out[valid] = adjusted_local
    return out


def build_model_and_family_summaries(config_metrics_df):
    """
    Equal-weight hierarchical aggregation:
      config/strategy -> model -> family.

    This prevents TabPFN (which can have extra fine-tuning modes) or TFMs in
    general from receiving extra weight merely because more configurations were run.
    """
    metric_cols = [
        c for c in [
            "error_rate",
            "balanced_accuracy_oof",
            "mean_confidence_errors",
            "median_confidence_errors",
            "mean_confidence_correct",
            "median_confidence_correct",
            "confidence_correct_minus_error",
            "mean_normalized_entropy_errors",
            "median_normalized_entropy_errors",
            "mean_normalized_entropy_correct",
            "median_normalized_entropy_correct",
            "error_detection_auroc_confidence",
            "error_detection_auprc_confidence",
            "overconfident_error_fraction_ge_090",
            "ece_toplabel",
            "multiclass_brier",
            "log_loss",
            "aurc",
            "risk_at_80pct_coverage",
            "risk_at_90pct_coverage",
            "risk_at_95pct_coverage",
            "risk_at_100pct_coverage",
        ] if c in config_metrics_df.columns
    ]

    model_summary = (
        config_metrics_df
        .groupby(["dataset", "model_family", "model_name"], as_index=False, dropna=False)[metric_cols]
        .median(numeric_only=True)
    )

    family_summary = (
        model_summary
        .groupby(["dataset", "model_family"], as_index=False, dropna=False)[metric_cols]
        .median(numeric_only=True)
    )

    # Strategy-level descriptive summary (equal weight per model within each strategy).
    strategy_model = (
        config_metrics_df
        .groupby(
            ["dataset", "strategy_group", "model_family", "model_name"],
            as_index=False,
            dropna=False,
        )[metric_cols]
        .median(numeric_only=True)
    )
    strategy_summary = (
        strategy_model
        .groupby(["dataset", "strategy_group", "model_family"], as_index=False, dropna=False)[metric_cols]
        .median(numeric_only=True)
    )

    return model_summary, family_summary, strategy_summary


def paired_family_wilcoxon(family_summary_df, metrics, analysis_group):
    rows = []

    for metric in metrics:
        if metric not in family_summary_df.columns:
            continue

        pivot = family_summary_df.pivot(index="dataset", columns="model_family", values=metric)
        if "TFM" not in pivot.columns or "ML" not in pivot.columns:
            continue

        paired = pivot[["TFM", "ML"]].dropna()
        n = len(paired)

        tfm_vals = paired["TFM"].to_numpy(dtype=float)
        ml_vals = paired["ML"].to_numpy(dtype=float)
        diff = tfm_vals - ml_vals

        if n < 2 or np.allclose(diff, 0.0):
            stat = np.nan
            p_value = 1.0 if n >= 1 and np.allclose(diff, 0.0) else np.nan
        else:
            try:
                test = wilcoxon(
                    tfm_vals,
                    ml_vals,
                    alternative="two-sided",
                    method="auto",
                    zero_method="wilcox",
                )
                stat = float(test.statistic)
                p_value = float(test.pvalue)
            except Exception as exc:
                log(f"[WARNING] Wilcoxon failed for {metric}: {exc}")
                stat = np.nan
                p_value = np.nan

        rows.append({
            "analysis_group": analysis_group,
            "metric": metric,
            "n_paired_datasets": n,
            "tfm_median_across_datasets": float(np.median(tfm_vals)) if n else np.nan,
            "ml_median_across_datasets": float(np.median(ml_vals)) if n else np.nan,
            "median_difference_tfm_minus_ml": float(np.median(diff)) if n else np.nan,
            "wilcoxon_statistic": stat,
            "p_raw": p_value,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_raw"].to_numpy())
    return out


def _save_boxplot(df, category_col, value_col, title, ylabel, output_path, order=None):
    plot_df = df[[category_col, value_col]].dropna()
    if plot_df.empty:
        log(f"[WARNING] Plot skipped (no data): {output_path}")
        return

    if order is None:
        order = plot_df[category_col].drop_duplicates().tolist()
    else:
        order = [x for x in order if x in set(plot_df[category_col])]

    values = [plot_df.loc[plot_df[category_col] == cat, value_col].to_numpy() for cat in order]
    if not values:
        return

    width = max(8, 1.15 * len(order))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.boxplot(values, labels=order, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_uncertainty_figures(model_summary_df, family_summary_df, strategy_summary_df, rc_df):
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    model_order = FOUNDATION_MODELS + ML_BASELINES
    _save_boxplot(
        model_summary_df,
        "model_name",
        "median_confidence_errors",
        "Confidence of wrong predictions by model (100% training, OOF)",
        "Median max predicted probability on errors",
        os.path.join(fig_dir, "01_error_confidence_by_model_dataset_level.png"),
        order=model_order,
    )

    _save_boxplot(
        family_summary_df,
        "model_family",
        "median_confidence_errors",
        "TFM vs ML: confidence when predictions are wrong",
        "Median confidence on errors (lower = less overconfident)",
        os.path.join(fig_dir, "02_error_confidence_TFM_vs_ML.png"),
        order=["TFM", "ML"],
    )

    _save_boxplot(
        family_summary_df,
        "model_family",
        "median_normalized_entropy_errors",
        "TFM vs ML: predictive entropy when predictions are wrong",
        "Median normalized entropy on errors (higher = more uncertain)",
        os.path.join(fig_dir, "03_error_entropy_TFM_vs_ML.png"),
        order=["TFM", "ML"],
    )

    _save_boxplot(
        family_summary_df,
        "model_family",
        "error_detection_auroc_confidence",
        "TFM vs ML: ability of uncertainty to detect errors",
        "Error-detection AUROC (higher = better)",
        os.path.join(fig_dir, "04_error_detection_AUROC_TFM_vs_ML.png"),
        order=["TFM", "ML"],
    )

    _save_boxplot(
        family_summary_df,
        "model_family",
        "aurc",
        "TFM vs ML: selective prediction risk-coverage performance",
        "AURC (lower = better)",
        os.path.join(fig_dir, "05_AURC_TFM_vs_ML.png"),
        order=["TFM", "ML"],
    )

    strategy_order = [
        "TFM In-Context",
        "TFM Base-FT",
        "TFM PEFT",
        "TFM Fine-tune (meta-learning)",
        "TFM Fine-tune (sft)",
        "TFM Fine-tune (native)",
        "ML",
    ]
    _save_boxplot(
        strategy_summary_df,
        "strategy_group",
        "median_confidence_errors",
        "Confidence of wrong predictions by adaptation strategy",
        "Median confidence on errors (lower = less overconfident)",
        os.path.join(fig_dir, "06_error_confidence_by_strategy.png"),
        order=strategy_order,
    )

    if rc_df is not None and len(rc_df):
        # Equal-weight aggregation: configuration -> model -> family -> datasets.
        rc_model = (
            rc_df
            .groupby(
                ["dataset", "model_family", "model_name", "requested_coverage"],
                as_index=False,
                dropna=False,
            )["risk"]
            .median()
        )
        rc_family_dataset = (
            rc_model
            .groupby(["dataset", "model_family", "requested_coverage"], as_index=False)["risk"]
            .median()
        )
        rc_family = (
            rc_family_dataset
            .groupby(["model_family", "requested_coverage"], as_index=False)["risk"]
            .median()
            .sort_values(["model_family", "requested_coverage"])
        )
        rc_family.to_csv(
            os.path.join(OUTPUT_DIR, "family_risk_coverage_summary.csv"),
            index=False,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        for family in ["TFM", "ML"]:
            sub = rc_family[rc_family["model_family"] == family]
            if len(sub):
                ax.plot(sub["requested_coverage"], sub["risk"], marker="o", label=family)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk")
        ax.set_title("Risk-coverage curve: TFM vs ML")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(fig_dir, "07_risk_coverage_TFM_vs_ML.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _concat_existing_dataset_files(dataset_configs, suffix):
    frames = []
    for ds_cfg in dataset_configs:
        if str(ds_cfg["name"]).lower() in EXCLUDED_DATASETS:
            continue
        path = os.path.join(OUTPUT_DIR, f"{ds_cfg['name']}{suffix}")
        if os.path.exists(path):
            try:
                frames.append(pd.read_csv(path))
            except Exception as exc:
                log(f"[WARNING] Could not read {path}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_configs = load_dataset_configs(YAML_PATH)
    preprocessings = build_preprocessing_list()

    if START_FROM_DATASET_IDX < 0 or START_FROM_DATASET_IDX >= len(dataset_configs):
        raise ValueError(
            f"START_FROM_DATASET_IDX={START_FROM_DATASET_IDX} fuori range. "
            f"Valori validi: 0 .. {len(dataset_configs)-1}"
        )

    log(
        f"[INFO] Uncertainty analysis: 100% training fraction only | "
        f"starting dataset index={START_FROM_DATASET_IDX} "
        f"({dataset_configs[START_FROM_DATASET_IDX]['name']})"
    )

    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "analysis": "classification uncertainty / confidence / calibration / selective prediction",
            "yaml_path": YAML_PATH,
            "foundation_models": FOUNDATION_MODELS,
            "ml_baselines": ML_BASELINES,
            "models": MODELS,
            "ml_gridsearch_cv_splits": ML_GRIDSEARCH_CV_SPLITS,
            "ml_gridsearch_scoring": ML_GRIDSEARCH_SCORING,
            "ml_param_grids": {k: v for k, v in ML_PARAM_GRIDS.items() if k in ML_BASELINES},
            "optional_packages": {
                "catboost": CATBOOST_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "lightgbm": LIGHTGBM_AVAILABLE,
            },
            "standard_tuning_strategies": STANDARD_TUNING_STRATEGIES,
            "tabpfn_finetune_modes_by_model": TABPFN_FINETUNE_MODES_BY_MODEL,
            "preprocessings": preprocessings,
            "train_fractions": TRAIN_FRACTIONS,
            "excluded_datasets": sorted(EXCLUDED_DATASETS),
            "n_splits": N_SPLITS,
            "device": DEVICE,
            "start_from_dataset_idx": START_FROM_DATASET_IDX,
            "skip_completed_datasets": SKIP_COMPLETED_DATASETS,
            "ece_n_bins": ECE_N_BINS,
            "risk_coverage_levels": RISK_COVERAGE_LEVELS,
            "overconfidence_threshold": OVERCONFIDENCE_THRESHOLD,
            "probability_policy": "native predict_proba only; no decision_function conversion",
            "primary_uncertainty_metrics": PRIMARY_UNCERTAINTY_METRICS,
            "supplementary_calibration_metrics": SUPPLEMENTARY_CALIBRATION_METRICS,
        }, f, indent=2)

    for ds_idx, ds_cfg in enumerate(dataset_configs):
        if ds_idx < START_FROM_DATASET_IDX:
            log(f"[SKIP] dataset index {ds_idx} | name={ds_cfg['name']}")
            continue

        dataset_name = ds_cfg["name"]

        if str(dataset_name).lower() in EXCLUDED_DATASETS:
            log(f"[EXCLUDED] dataset={dataset_name}")
            continue

        pred_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_uncertainty_predictions.csv")
        config_metrics_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_uncertainty_config_metrics.csv")
        calibration_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_calibration_bins.csv")
        rc_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_risk_coverage.csv")
        fold_perf_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_fold_performance.csv")

        if SKIP_COMPLETED_DATASETS and all(
            os.path.exists(p)
            for p in [pred_path, config_metrics_path, calibration_path, rc_path]
        ):
            log(f"[SKIP COMPLETED] {dataset_name}")
            continue

        log("=" * 100)
        log(f"DATASET IDX={ds_idx} | DATASET: {dataset_name}")

        try:
            X, y, groups = load_single_dataset(ds_cfg)
        except Exception as e:
            log(f"[WARNING] Failed loading dataset {dataset_name}: {e}")
            continue

        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        log(f"{dataset_name} | shape={X.shape} | class_counts={class_counts}")

        if len(np.unique(y)) < 2:
            log(f"[WARNING] Skipping {dataset_name}: only one class found")
            continue

        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        dataset_prediction_frames = []
        dataset_fold_performance = []

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            log("-" * 80)
            log(f"{dataset_name} | fold={fold}/{N_SPLITS}")

            Xtr_full = X.iloc[tr_idx].reset_index(drop=True)
            Xte = X.iloc[te_idx].reset_index(drop=True)
            ytr_full = y[tr_idx]
            yte = y[te_idx]
            gtr_full = groups[tr_idx]
            gte = groups[te_idx]

            # TRAIN_FRACTIONS contains only 1.00, kept as a loop to preserve the
            # original benchmark structure and metadata.
            for frac in TRAIN_FRACTIONS:
                Xtr, ytr, gtr = sample_train_fraction(
                    Xtr_full,
                    ytr_full,
                    gtr_full,
                    frac,
                    random_state=RANDOM_STATE + fold,
                )

                log(
                    f"{dataset_name} | fold={fold} | frac={frac:.2f} | "
                    f"n_train={len(Xtr)} | n_test={len(Xte)} | "
                    f"train_counts={_class_count_dict(ytr)} | "
                    f"train_props={_class_prop_dict(ytr)} | "
                    f"test_counts={_class_count_dict(yte)} | "
                    f"test_props={_class_prop_dict(yte)}"
                )

                if len(np.unique(ytr)) < 2:
                    log(
                        f"[WARNING] Skipping frac={frac:.2f} for {dataset_name}: "
                        "training sample lost classes"
                    )
                    continue

                for prep in preprocessings:
                    try:
                        Xtr_p, Xte_p, prep_meta = apply_preprocessing(prep, Xtr, ytr, Xte)
                    except Exception as e:
                        log(
                            f"[WARNING] dataset={dataset_name} preprocessing={prep} "
                            f"failed: {e}"
                        )
                        continue

                    for model_name in MODELS:
                        candidate_pairs = get_model_strategy_mode_pairs(model_name)

                        for tuning_strategy, finetune_mode in candidate_pairs:
                            if not is_valid_combination(
                                model_name=model_name,
                                preprocessing=prep,
                                tuning_strategy=tuning_strategy,
                                finetune_mode=finetune_mode,
                            ):
                                log(
                                    f"[SKIP] dataset={dataset_name} model={model_name} "
                                    f"strategy={tuning_strategy} finetune_mode={finetune_mode} "
                                    f"prep={prep} frac={frac:.2f} reason=invalid combination"
                                )
                                continue

                            try:
                                if model_name in FOUNDATION_MODELS:
                                    metrics, y_pred, y_proba, class_labels = fit_and_eval_tabtune(
                                        model_name=model_name,
                                        tuning_strategy=tuning_strategy,
                                        finetune_mode=finetune_mode,
                                        X_train=Xtr_p,
                                        y_train=ytr,
                                        X_test=Xte_p,
                                        y_test=yte,
                                    )
                                else:
                                    metrics, y_pred, y_proba, class_labels = fit_and_eval_ml_baseline(
                                        model_name=model_name,
                                        X_train=Xtr_p,
                                        y_train=ytr,
                                        X_test=Xte_p,
                                        y_test=yte,
                                        groups_train=gtr,
                                    )

                                model_family = get_model_family(model_name)
                                strategy_group = get_strategy_group(
                                    model_name,
                                    tuning_strategy,
                                    finetune_mode,
                                )
                                config_label = make_configuration_label(
                                    model_name,
                                    tuning_strategy,
                                    finetune_mode,
                                )

                                base_meta = {
                                    "dataset": dataset_name,
                                    "fold": fold,
                                    "train_fraction": frac,
                                    "preprocessing": prep,
                                    "embedding_dim": prep_meta.get("embedding_dim"),
                                    "model_family": model_family,
                                    "strategy_group": strategy_group,
                                    "model_name": model_name,
                                    "tuning_strategy": tuning_strategy,
                                    "finetune_mode": finetune_mode,
                                    "configuration_label": config_label,
                                }

                                pred_rows = make_prediction_rows(
                                    base_meta=base_meta,
                                    sample_indices=te_idx,
                                    group_ids=gte,
                                    y_true=yte,
                                    y_pred=y_pred,
                                    y_proba=y_proba,
                                    class_labels=class_labels,
                                )
                                dataset_prediction_frames.append(pred_rows)

                                perf_row = {**base_meta, **metrics}
                                if "ml_best_cv_score" not in perf_row:
                                    perf_row["ml_best_cv_score"] = np.nan
                                if "ml_best_params" not in perf_row:
                                    perf_row["ml_best_params"] = None
                                dataset_fold_performance.append(perf_row)

                                n_fold_errors = int(np.sum(np.asarray(y_pred) != np.asarray(yte)))
                                mean_wrong_conf = (
                                    float(pred_rows.loc[pred_rows["error"] == 1, "predicted_confidence"].mean())
                                    if n_fold_errors > 0
                                    else np.nan
                                )

                                log(
                                    f"{dataset_name} | {model_name} | {tuning_strategy} | "
                                    f"finetune_mode={finetune_mode} | {prep} | frac={frac:.2f} | "
                                    f"BA={metrics['balanced_accuracy']:.4f} | errors={n_fold_errors} | "
                                    f"mean_conf_wrong={mean_wrong_conf:.4f}"
                                )

                            except Exception as e:
                                log(
                                    f"[WARNING] dataset={dataset_name} model={model_name} "
                                    f"strategy={tuning_strategy} finetune_mode={finetune_mode} "
                                    f"prep={prep} frac={frac:.2f} failed: {e}"
                                )

        if not dataset_prediction_frames:
            log(f"[WARNING] No uncertainty predictions produced for dataset {dataset_name}")
            continue

        dataset_pred_df = pd.concat(dataset_prediction_frames, ignore_index=True)
        dataset_pred_df.to_csv(pred_path, index=False)

        fold_perf_df = pd.DataFrame(dataset_fold_performance)
        fold_perf_df.to_csv(fold_perf_path, index=False)

        try:
            config_metrics_df, cal_df, rc_df = summarize_dataset_predictions(dataset_pred_df)
            config_metrics_df.to_csv(config_metrics_path, index=False)
            cal_df.to_csv(calibration_path, index=False)
            rc_df.to_csv(rc_path, index=False)
        except Exception as exc:
            log(f"[ERROR] Failed uncertainty summary for {dataset_name}: {exc}")
            raise

        log(f"Saved: {pred_path}")
        log(f"Saved: {config_metrics_path}")
        log(f"Saved: {calibration_path}")
        log(f"Saved: {rc_path}")
        log(f"Saved: {fold_perf_path}")

    # ------------------------------------------------------------------
    # Global outputs: read all per-dataset files so restart/skip mode still
    # produces a complete global analysis from previously completed datasets.
    # ------------------------------------------------------------------
    all_predictions = _concat_existing_dataset_files(
        dataset_configs,
        "_uncertainty_predictions.csv",
    )
    all_config_metrics = _concat_existing_dataset_files(
        dataset_configs,
        "_uncertainty_config_metrics.csv",
    )
    all_calibration = _concat_existing_dataset_files(
        dataset_configs,
        "_calibration_bins.csv",
    )
    all_rc = _concat_existing_dataset_files(
        dataset_configs,
        "_risk_coverage.csv",
    )
    all_fold_performance = _concat_existing_dataset_files(
        dataset_configs,
        "_fold_performance.csv",
    )

    if all_config_metrics.empty:
        log("[WARNING] No global uncertainty summaries available.")
        return

    all_predictions.to_csv(
        os.path.join(OUTPUT_DIR, "all_uncertainty_predictions.csv"),
        index=False,
    )
    all_config_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "all_uncertainty_metrics_by_configuration.csv"),
        index=False,
    )
    all_calibration.to_csv(
        os.path.join(OUTPUT_DIR, "all_calibration_bins.csv"),
        index=False,
    )
    all_rc.to_csv(
        os.path.join(OUTPUT_DIR, "all_risk_coverage.csv"),
        index=False,
    )
    all_fold_performance.to_csv(
        os.path.join(OUTPUT_DIR, "all_fold_performance.csv"),
        index=False,
    )

    model_summary, family_summary, strategy_summary = build_model_and_family_summaries(
        all_config_metrics
    )
    model_summary.to_csv(
        os.path.join(OUTPUT_DIR, "model_dataset_uncertainty_summary.csv"),
        index=False,
    )
    family_summary.to_csv(
        os.path.join(OUTPUT_DIR, "family_dataset_uncertainty_summary.csv"),
        index=False,
    )
    strategy_summary.to_csv(
        os.path.join(OUTPUT_DIR, "strategy_dataset_uncertainty_summary.csv"),
        index=False,
    )

    primary_tests = paired_family_wilcoxon(
        family_summary,
        PRIMARY_UNCERTAINTY_METRICS,
        analysis_group="primary_uncertainty",
    )
    calibration_tests = paired_family_wilcoxon(
        family_summary,
        SUPPLEMENTARY_CALIBRATION_METRICS,
        analysis_group="supplementary_calibration",
    )

    primary_tests.to_csv(
        os.path.join(OUTPUT_DIR, "TFM_vs_ML_wilcoxon_primary_uncertainty_holm.csv"),
        index=False,
    )
    calibration_tests.to_csv(
        os.path.join(OUTPUT_DIR, "TFM_vs_ML_wilcoxon_calibration_holm.csv"),
        index=False,
    )

    make_uncertainty_figures(
        model_summary_df=model_summary,
        family_summary_df=family_summary,
        strategy_summary_df=strategy_summary,
        rc_df=all_rc,
    )

    log("=" * 100)
    log("UNCERTAINTY ANALYSIS COMPLETED")
    log(f"Output directory: {OUTPUT_DIR}")
    log(
        "Primary interpretation: lower confidence on errors + higher entropy on errors + "
        "higher error-detection AUROC + lower AURC indicate better error awareness."
    )


if __name__ == "__main__":
    main()
