"""
BENCHMARK TABULAR FOUNDATION MODELS - CLASSIFICATION
===================================================

Features
--------
- Reads multiple datasets from dataset_info.yaml
- 10-fold StratifiedGroupKFold CV
- Training fractions: 10%, 20%, 50%, 75%, 100%
- Models via TabTune
- Tuning strategies:
    * inference
    * base-ft
    * peft
- Preprocessing variants:
    * auto
    * hybrid_ohe_freq
    * ae_embedding_<latent_dim>
    * tabpfn_embedding
    * tabpfn_embedding_pca_<dim>
- Metrics:
    * balanced accuracy
    * precision macro
    * recall macro
    * AUROC OvR macro
    * AUPRC macro
- Saves:
    * per-dataset fold results
    * per-dataset summary
    * global fold results
    * global summary
"""

import os
import json
import yaml
import logging
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

# =========================
# TabTune
# =========================
from tabtune.TabularPipeline.pipeline import TabularPipeline

# =========================
# Huggingface
# =========================
from huggingface_hub import login
import os

# Check token
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token)
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

TASK_NAME = "classification_benchmark_AUPRC_macro_fixed"
OUTPUT_DIR = f"/blue/salemi/simonerancati/TabProject/Results_Paper/outputs_{TASK_NAME}"
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

RANDOM_STATE = 42
N_SPLITS = 10
TRAIN_FRACTIONS = [0.10, 0.20, 0.50, 0.75, 1.00]
EXCLUDED_DATASETS = {"student_depression"}
# Inner CV for ML baseline hyperparameter tuning
ML_GRIDSEARCH_CV_SPLITS = 3
ML_GRIDSEARCH_SCORING = "balanced_accuracy"
ML_GRIDSEARCH_N_JOBS = -1
BINARY_ONLY = True

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

#if ENABLE_OPTIONAL_BOOSTERS and CATBOOST_AVAILABLE:
#    ML_BASELINES.append("CatBoost")

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
START_FROM_DATASET_IDX = 0

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


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba,
    class_labels,
):
    result = {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "auroc_ovr_macro": np.nan,
        "auprc_macro": np.nan,

        # Diagnostic values for binary datasets
        "auprc_class_0": np.nan,
        "auprc_class_1": np.nan,
        "auprc_class_0_label": np.nan,
        "auprc_class_1_label": np.nan,

        # Exact/near-exact reproduction of previous binary AUPRC logic
        "auprc_legacy": np.nan,
    }

    if y_proba is None:
        return result

    y_proba = np.asarray(y_proba)
    class_labels = np.asarray(class_labels)

    if (
        y_proba.ndim != 2
        or y_proba.shape[1] != len(class_labels)
    ):
        return result

    try:

        # =====================================================
        # NEW CONSISTENT ONE-VS-REST CALCULATION
        # =====================================================

        y_bin = np.column_stack([
            (y_true == c).astype(int)
            for c in class_labels
        ])

        aurocs = []
        auprcs = []

        for j, c in enumerate(class_labels):

            y_class = y_bin[:, j]

            if len(np.unique(y_class)) < 2:
                continue

            scores = y_proba[:, j]

            class_auroc = roc_auc_score(
                y_class,
                scores
            )

            class_auprc = average_precision_score(
                y_class,
                scores
            )

            aurocs.append(class_auroc)
            auprcs.append(class_auprc)

        if aurocs:
            result["auroc_ovr_macro"] = float(
                np.mean(aurocs)
            )

        if auprcs:
            result["auprc_macro"] = float(
                np.mean(auprcs)
            )

        # =====================================================
        # SAVE BOTH CLASS-SPECIFIC AUPRCs FOR BINARY DATASETS
        # =====================================================

        if len(class_labels) == 2:

            class0 = class_labels[0]
            class1 = class_labels[1]

            y_class0 = (y_true == class0).astype(int)
            y_class1 = (y_true == class1).astype(int)

            result["auprc_class_0_label"] = class0
            result["auprc_class_1_label"] = class1

            if len(np.unique(y_class0)) == 2:
                result["auprc_class_0"] = float(
                    average_precision_score(
                        y_class0,
                        y_proba[:, 0]
                    )
                )

            if len(np.unique(y_class1)) == 2:
                result["auprc_class_1"] = float(
                    average_precision_score(
                        y_class1,
                        y_proba[:, 1]
                    )
                )

            # =================================================
            # LEGACY AUPRC
            # Reproduces the OLD calculation as closely as possible
            # =================================================

            try:
                result["auprc_legacy"] = float(
                    average_precision_score(
                        y_true,
                        y_proba[:, 1]
                    )
                )
            except Exception:
                result["auprc_legacy"] = np.nan

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
    y_proba = safe_predict_proba(pipeline, X_test)

    log(
        f"[DEBUG] model={model_name} strategy={tuning_strategy} "
        f"finetune_mode={finetune_mode} "
        f"train_classes={np.unique(y_train)} test_classes={np.unique(y_test)}"
    )

    if y_proba is None:
        log(
            f"[DEBUG] model={model_name} strategy={tuning_strategy} "
            f"finetune_mode={finetune_mode} y_proba=None"
        )
    else:
        log(
            f"[DEBUG] model={model_name} strategy={tuning_strategy} "
            f"finetune_mode={finetune_mode} "
            f"y_proba_shape={np.asarray(y_proba).shape}"
        )

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=np.array(sorted(np.unique(y_train))),
    )

    return metrics

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
    model = build_ml_classifier(model_name)
    param_grid = deepcopy(ML_PARAM_GRIDS[model_name])

    # Avoid invalid KNN candidates in very small low-data folds.
    if model_name == "KNN":
        max_neighbors = max(1, len(X_train) - 1)
        valid_neighbors = [
            k for k in param_grid["n_neighbors"] if k <= max_neighbors
        ]
        param_grid["n_neighbors"] = valid_neighbors or [1]
        model.set_params(n_neighbors=min(5, max_neighbors))

    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = int(class_counts.min())

    # The inner CV cannot use more folds than the rarest class contains.
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
    y_proba = safe_predict_proba(best_estimator, X_test)

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=np.array(sorted(np.unique(y_train))),
    )

    metrics["ml_best_cv_score"] = (
        float(best_score) if not pd.isna(best_score) else np.nan
    )
    metrics["ml_best_params"] = json.dumps(best_params)

    return metrics


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
# MAIN
# ============================================================
def main():
    dataset_configs = load_dataset_configs(YAML_PATH)
    preprocessings = build_preprocessing_list()

    if START_FROM_DATASET_IDX < 0 or START_FROM_DATASET_IDX >= len(dataset_configs):
        raise ValueError(
            f"START_FROM_DATASET_IDX={START_FROM_DATASET_IDX} fuori range. "
            f"Valori validi: 0 .. {len(dataset_configs)-1}"
        )

    log(
        f"[INFO] Starting from dataset index {START_FROM_DATASET_IDX} "
        f"({dataset_configs[START_FROM_DATASET_IDX]['name']})"
    )

    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({
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
            "ae_latent_dims": AE_LATENT_DIMS,
            "tabpfn_emb_pca_dims": TABPFN_EMB_PCA_DIMS if USE_TABPFN_EMB_PCA else [],
            "train_fractions": TRAIN_FRACTIONS,
            "excluded_datasets": sorted(EXCLUDED_DATASETS),
            "n_splits": N_SPLITS,
            "device": DEVICE,
            "start_from_dataset_idx": START_FROM_DATASET_IDX,
        }, f, indent=2)

    all_results = []

    for ds_idx, ds_cfg in enumerate(dataset_configs):
        if ds_idx < START_FROM_DATASET_IDX:
            log(f"[SKIP] dataset index {ds_idx} | name={ds_cfg['name']}")
            continue

        dataset_name = ds_cfg["name"]

        if str(dataset_name).lower() in EXCLUDED_DATASETS:
            log(f"[EXCLUDED] dataset={dataset_name}")
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

        if BINARY_ONLY and len(np.unique(y)) != 2:
            log(
                f"[SKIP] dataset={dataset_name} | "
                f"n_classes={len(np.unique(y))} | reason=not binary"
            )
            continue

        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        dataset_results = []

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            log("-" * 80)
            log(f"{dataset_name} | fold={fold}/{N_SPLITS}")

            Xtr_full = X.iloc[tr_idx].reset_index(drop=True)
            Xte = X.iloc[te_idx].reset_index(drop=True)
            ytr_full = y[tr_idx]
            yte = y[te_idx]
            gtr_full = groups[tr_idx]

            for frac in TRAIN_FRACTIONS:
                Xtr, ytr, gtr = sample_train_fraction(
                    Xtr_full, ytr_full, gtr_full, frac, random_state=RANDOM_STATE + fold
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
                    log(f"[WARNING] Skipping frac={frac:.2f} for {dataset_name}: training sample lost classes")
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
                                    f"prep={prep} frac={frac:.2f} "
                                    f"reason=invalid combination"
                                )
                                continue

                            try:
                                if model_name in FOUNDATION_MODELS:
                                    metrics = fit_and_eval_tabtune(
                                        model_name=model_name,
                                        tuning_strategy=tuning_strategy,
                                        finetune_mode=finetune_mode,
                                        X_train=Xtr_p,
                                        y_train=ytr,
                                        X_test=Xte_p,
                                        y_test=yte,
                                    )
                                else:
                                    metrics = fit_and_eval_ml_baseline(
                                        model_name=model_name,
                                        X_train=Xtr_p,
                                        y_train=ytr,
                                        X_test=Xte_p,
                                        y_test=yte,
                                        groups_train=gtr,
                                    )

                                row = {
                                    "dataset": dataset_name,
                                    "fold": fold,
                                    "train_fraction": frac,
                                    "preprocessing": prep,
                                    "embedding_dim": prep_meta.get("embedding_dim"),
                                    "model_name": model_name,
                                    "tuning_strategy": tuning_strategy,
                                    "finetune_mode": finetune_mode,
                                    **metrics,
                                }

                                if "ml_best_cv_score" not in row:
                                    row["ml_best_cv_score"] = np.nan

                                if "ml_best_params" not in row:
                                    row["ml_best_params"] = None

                                dataset_results.append(row)
                                all_results.append(row)

                                log(
                                    f"{dataset_name} | {model_name} | {tuning_strategy} | "
                                    f"finetune_mode={finetune_mode} | {prep} | "
                                    f"frac={frac:.2f} | "
                                    f"BA={metrics['balanced_accuracy']:.4f} | "
                                    f"AUROC={metrics['auroc_ovr_macro']:.4f} | "
                                    f"AUPRC_macro={metrics['auprc_macro']:.4f} | "
                                    f"AUPRC_c0={metrics['auprc_class_0']:.4f} | "
                                    f"AUPRC_c1={metrics['auprc_class_1']:.4f} | "
                                    f"AUPRC_old={metrics['auprc_legacy']:.4f}"
                                )

                            except Exception as e:
                                log(
                                    f"[WARNING] dataset={dataset_name} model={model_name} "
                                    f"strategy={tuning_strategy} finetune_mode={finetune_mode} "
                                    f"prep={prep} frac={frac:.2f} failed: {e}"
                                )

        if len(dataset_results) == 0:
            log(f"[WARNING] No results for dataset {dataset_name}")
            continue

        dataset_df = pd.DataFrame(dataset_results)
        dataset_fold_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_fold_results.csv")
        dataset_df.to_csv(dataset_fold_path, index=False)

        dataset_summary = (
            dataset_df
            .groupby(
                [
                    "dataset",
                    "model_name",
                    "tuning_strategy",
                    "finetune_mode",
                    "preprocessing",
                    "embedding_dim",
                    "train_fraction",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                balanced_accuracy_mean=("balanced_accuracy", "mean"),
                balanced_accuracy_std=("balanced_accuracy", "std"),
                precision_macro_mean=("precision_macro", "mean"),
                precision_macro_std=("precision_macro", "std"),
                recall_macro_mean=("recall_macro", "mean"),
                recall_macro_std=("recall_macro", "std"),
                auroc_ovr_macro_mean=("auroc_ovr_macro", "mean"),
                auroc_ovr_macro_std=("auroc_ovr_macro", "std"),
                auprc_macro_mean=("auprc_macro", "mean"),
                auprc_macro_std=("auprc_macro", "std"),
                auprc_class_0_mean=("auprc_class_0", "mean"),
                auprc_class_0_std=("auprc_class_0", "std"),
                auprc_class_1_mean=("auprc_class_1", "mean"),
                auprc_class_1_std=("auprc_class_1", "std"),
                auprc_legacy_mean=("auprc_legacy", "mean"),
                auprc_legacy_std=("auprc_legacy", "std"),
                ml_best_cv_score_mean=("ml_best_cv_score", "mean"),
                ml_best_cv_score_std=("ml_best_cv_score", "std"),
            )
            .sort_values(
                ["train_fraction", "balanced_accuracy_mean"],
                ascending=[True, False]
            )
        )

        dataset_summary_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_summary.csv")
        dataset_summary.to_csv(dataset_summary_path, index=False)

        log(f"Saved: {dataset_fold_path}")
        log(f"Saved: {dataset_summary_path}")

    if len(all_results) == 0:
        log("[WARNING] No results produced across all datasets.")
        return

    all_df = pd.DataFrame(all_results)
    all_fold_path = os.path.join(OUTPUT_DIR, "all_datasets_fold_results.csv")
    all_df.to_csv(all_fold_path, index=False)

    all_summary = (
        all_df
        .groupby(
            [
                "dataset",
                "model_name",
                "tuning_strategy",
                "finetune_mode",
                "preprocessing",
                "embedding_dim",
                "train_fraction",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", "std"),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", "std"),
            auroc_ovr_macro_mean=("auroc_ovr_macro", "mean"),
            auroc_ovr_macro_std=("auroc_ovr_macro", "std"),
            auprc_macro_mean=("auprc_macro", "mean"),
            auprc_macro_std=("auprc_macro", "std"),
            auprc_class_0_mean=("auprc_class_0", "mean"),
            auprc_class_0_std=("auprc_class_0", "std"),
            auprc_class_1_mean=("auprc_class_1", "mean"),
            auprc_class_1_std=("auprc_class_1", "std"),
            auprc_legacy_mean=("auprc_legacy", "mean"),
            auprc_legacy_std=("auprc_legacy", "std"),
            ml_best_cv_score_mean=("ml_best_cv_score", "mean"),
            ml_best_cv_score_std=("ml_best_cv_score", "std"),
        )
        .sort_values(
            ["dataset", "train_fraction", "balanced_accuracy_mean"],
            ascending=[True, True, False]
        )
    )

    all_summary_path = os.path.join(OUTPUT_DIR, "all_datasets_summary.csv")
    all_summary.to_csv(all_summary_path, index=False)

    log(f"Saved: {all_fold_path}")
    log(f"Saved: {all_summary_path}")
    log("Benchmark completed.")


if __name__ == "__main__":
    main()