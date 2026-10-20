"""
BENCHMARK TABULAR FOUNDATION MODELS - REGRESSION
================================================

Features
--------
- Reads multiple datasets from dataset_info.yaml
- 10-fold GroupKFold CV
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
    * MAE
    * RMSE
    * R2
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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

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
    from tabpfn_extensions import TabPFNRegressor as TabPFNExtRegressor
    from tabpfn_extensions.embedding import TabPFNEmbedding
except Exception:
    TABPFN_EMB_AVAILABLE = False

# =========================
# Optional: strong external ML baselines
# =========================
CATBOOST_AVAILABLE = True
try:
    from catboost import CatBoostRegressor
except Exception:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

XGBOOST_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except Exception:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

LIGHTGBM_AVAILABLE = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None


# ============================================================
# CONFIG
# ============================================================
YAML_PATH = "/blue/salemi/simonerancati/TabProject/dataset_info_regression.yaml"

START_FROM_DATASET_IDX = 0
TASK_NAME = "regression_benchmark_multidataset_extended_ml"
OUTPUT_DIR = f"/blue/salemi/simonerancati/TabProject/Results_Paper/outputs_{TASK_NAME}"
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

RANDOM_STATE = 42
N_SPLITS = 10
TRAIN_FRACTIONS = [0.10, 0.20, 0.50, 0.75, 1.00]
EXCLUDED_DATASETS = {"health_insurance"}
# Inner CV for ML baseline hyperparameter tuning
ML_GRIDSEARCH_CV_SPLITS = 3
ML_GRIDSEARCH_SCORING = "neg_root_mean_squared_error"
ML_GRIDSEARCH_N_JOBS = -1

FOUNDATION_MODELS = [
    "TabPFN",
    #"TabPFNv26",
    #"TabPFNv25",
    "TabICLv2",
    "TabDPT",
    "Mitra",
    "ContextTab",
    "Limix",
]

# Extended classical/deep baselines. All are tuned with GridSearchCV.
ENABLE_NEURAL_BASELINES = True
ENABLE_DISTANCE_BASELINES = True
ENABLE_OPTIONAL_BOOSTERS = True

ML_BASELINES = [
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "RandomForest",
    "ExtraTrees",
    "HistGB",
    "GradientBoosting",
    "SVR_RBF",
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

ML_PARAM_GRIDS = {
    "LinearRegression": {},

    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },

    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    },

    "RandomForest": {
        "n_estimators": [300, 600],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", 1.0],
    },

    "ExtraTrees": {
        "n_estimators": [300, 600],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", 1.0],
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
        "loss": ["squared_error", "huber"],
    },

    "SVR_RBF": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
        "epsilon": [0.01, 0.1],
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

#MODELS = FOUNDATION_MODELS + ML_BASELINES
MODELS = ML_BASELINES


# Standard strategy for all foundation models
STANDARD_TUNING_STRATEGIES = [
    "inference",
]

# Regression fine-tuning modes.
# turn_by_turn / tbt is regression-only and is especially relevant for TabPFN-like models.
REGRESSION_FINETUNE_MODES_BY_MODEL = {
    "TabPFN": [
        "turn_by_turn",
    ],
    "TabPFNv25": [
        "turn_by_turn",
    ],
    "TabPFNv26": [
        "turn_by_turn",
        "native",
    ],
    "TabDPT": [
        "meta-learning",
    ],
    "ContextTab": [
        "meta-learning",
    ],
    "Mitra": [
        "turn_by_turn",
    ],
    "Limix": [
        "meta-learning",
    ],
}

REGRESSION_FINETUNE_MODELS = list(REGRESSION_FINETUNE_MODES_BY_MODEL.keys())

AE_LATENT_DIMS = [8, 16, 32, 64, 128]

USE_AE_EMBEDDINGS = False
USE_TABPFN_EMBEDDINGS = False
USE_TABPFN_EMB_PCA = False
TABPFN_EMB_PCA_DIMS = [32, 64, 128]

LOW_CARD_MAX_UNIQUE = 20

AE_EPOCHS = 20
AE_BATCH_SIZE = 256
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5

TABPFN_EMB_N_FOLD = 5
TABPFN_EMB_N_ESTIMATORS = 1

DEVICE = "cuda" if (AE_AVAILABLE and torch.cuda.is_available()) else "cpu"
ID_COLS = ["PATIENT_ID", "ASS_NUMBER"]

INFERENCE_PARAMS = {
    "device": DEVICE,
}

BASE_FT_PARAMS = {
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
    if path.endswith(".csv"):
        # prova a riconoscere automaticamente il separatore
        df = pd.read_csv(path, sep=None, engine="python")

        # fallback: se viene letta una sola colonna contenente ";", rilegge con sep=";"
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")

        return df

    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file extension: {path}")


def load_single_dataset(ds_cfg):
    name = ds_cfg["name"]
    path = ds_cfg["path"]
    target_col = ds_cfg["target"]

    df = read_table(path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset '{name}'")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col], errors="ignore").copy()

    valid = pd.Series(y).notna().to_numpy()
    X = X.loc[valid].reset_index(drop=True)
    y = np.asarray(y)[valid].astype(float)

    if "PATIENT_ID" in X.columns:
        groups = X["PATIENT_ID"].astype(str).to_numpy()
        X = X.drop(columns=["PATIENT_ID"], errors="ignore")
    else:
        groups = np.arange(len(X)).astype(str)

    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")
    X = X.dropna(axis=1, how="all")

    return X.reset_index(drop=True), y, groups


# ============================================================
# HELPERS
# ============================================================
def is_valid_combination(model_name, preprocessing, tuning_strategy, finetune_mode=None):
    # ML baselines: only numeric preprocessing, no auto, inference only
    if model_name in ML_BASELINES:
        if preprocessing == "auto":
            return False
        if tuning_strategy != "inference":
            return False
        if finetune_mode is not None:
            return False

    # finetune_mode only makes sense when tuning_strategy == finetune
    if finetune_mode is not None and tuning_strategy != "finetune":
        return False

    # finetune only allowed for models with an explicit regression finetune mode
    if tuning_strategy == "finetune":
        if model_name not in REGRESSION_FINETUNE_MODELS:
            return False

        allowed_modes = REGRESSION_FINETUNE_MODES_BY_MODEL.get(model_name, [])
        if finetune_mode not in allowed_modes:
            return False

    return True

def get_model_strategy_mode_pairs(model_name):
    """
    Returns a list of (tuning_strategy, finetune_mode) pairs.

    Foundation models:
        inference
        finetune + model-specific regression finetune modes

    ML baselines:
        inference only
    """

    if model_name in ML_BASELINES:
        return [("inference", None)]

    pairs = [(s, None) for s in STANDARD_TUNING_STRATEGIES]

    if model_name in REGRESSION_FINETUNE_MODELS:
        for mode in REGRESSION_FINETUNE_MODES_BY_MODEL[model_name]:
            pairs.append(("finetune", mode))

    return pairs

def build_preprocessing_list():
    preprocessings = ["hybrid_ohe_freq"]

    if USE_AE_EMBEDDINGS and AE_AVAILABLE:
        preprocessings.extend([f"ae_embedding_{d}" for d in AE_LATENT_DIMS])

    if USE_TABPFN_EMBEDDINGS and TABPFN_EMB_AVAILABLE:
        preprocessings.append("tabpfn_embedding")
        if USE_TABPFN_EMB_PCA:
            preprocessings.extend([f"tabpfn_embedding_pca_{d}" for d in TABPFN_EMB_PCA_DIMS])

    return preprocessings


def sample_train_fraction(X, y, groups, frac, random_state=42):
    if frac >= 0.999:
        return X, y, groups

    rng = check_random_state(random_state)
    idx_all = np.arange(len(X))
    n = max(2, int(np.floor(len(idx_all) * frac)))
    sampled_idx = rng.choice(idx_all, size=n, replace=False)
    sampled_idx = np.sort(sampled_idx)

    return (
        X.iloc[sampled_idx].reset_index(drop=True),
        y[sampled_idx],
        groups[sampled_idx],
    )


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


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

def build_ml_regressor(model_name):
    """Construct one regression baseline with deterministic settings."""
    if model_name == "LinearRegression":
        return LinearRegression()

    if model_name == "Ridge":
        return Ridge(random_state=RANDOM_STATE)

    if model_name == "ElasticNet":
        return ElasticNet(
            max_iter=10000,
            random_state=RANDOM_STATE,
        )

    if model_name == "RandomForest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,  # GridSearchCV performs the outer parallelization
        )

    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if model_name == "HistGB":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)

    if model_name == "SVR_RBF":
        return SVR(kernel="rbf")

    if model_name == "KNN":
        return KNeighborsRegressor()

    if model_name == "MLP":
        return MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )

    if model_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not installed")
        return CatBoostRegressor(
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=RANDOM_STATE,
        )

    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not installed")
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

    if model_name == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not installed")
        return LGBMRegressor(
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
    if model_name == "SVR_RBF" and len(X_train) > 15000:
        raise RuntimeError(
            "SVR_RBF skipped because the training set exceeds 15,000 rows; "
            "libsvm scaling would make nested grid search impractical."
        )

    model = build_ml_regressor(model_name)
    param_grid = deepcopy(ML_PARAM_GRIDS[model_name])

    # Avoid invalid KNN candidates in very small low-data folds.
    if model_name == "KNN":
        max_neighbors = max(1, len(X_train) - 1)
        valid_neighbors = [
            k for k in param_grid["n_neighbors"] if k <= max_neighbors
        ]
        param_grid["n_neighbors"] = valid_neighbors or [1]
        model.set_params(n_neighbors=min(5, max_neighbors))

    if groups_train is None:
        groups_train = np.arange(len(X_train)).astype(str)

    n_unique_groups = len(np.unique(groups_train))
    inner_splits = min(ML_GRIDSEARCH_CV_SPLITS, n_unique_groups)

    if inner_splits < 2:
        log(
            f"[WARNING] model={model_name} GridSearch skipped: "
            f"not enough groups. Fitting default model."
        )
        model.fit(X_train, y_train)
        best_estimator = model
        best_params = {}
        best_score = np.nan

    else:
        inner_cv = GroupKFold(n_splits=inner_splits)

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

    metrics = compute_regression_metrics(
        y_true=y_test,
        y_pred=y_pred,
    )

    metrics["ml_best_cv_score"] = (
        float(best_score) if not pd.isna(best_score) else np.nan
    )
    metrics["ml_best_params"] = json.dumps(best_params)

    return metrics


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

    reg = TabPFNExtRegressor(
        n_estimators=TABPFN_EMB_N_ESTIMATORS,
        device=DEVICE,
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
    )

    emb = TabPFNEmbedding(tabpfn_clf=reg, n_fold=TABPFN_EMB_N_FOLD)
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

    if strategy == "finetune":
        params = deepcopy(BASE_FT_PARAMS)

        if finetune_mode is not None:
            params["finetune_mode"] = finetune_mode

        return params

    raise ValueError(strategy)

# def get_finetune_mode(model_name, tuning_strategy, task_type="regression"):
#     if task_type == "regression" and tuning_strategy == "finetune":
#         if model_name in {"TabPFN", "Mitra", "ContextTab"}:
#             return "turn_by_turn"
#     return "meta-learning"

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
        task_type="regression",
        tuning_strategy=tuning_strategy,
        tuning_params=tuning_params,
    )

    pipeline.fit(X_train, y_train)
    y_pred = np.asarray(pipeline.predict(X_test)).reshape(-1)

    log(
        f"[DEBUG] model={model_name} strategy={tuning_strategy} "
        f"finetune_mode={finetune_mode} "
        f"y_train_min={np.min(y_train):.4f} y_train_max={np.max(y_train):.4f}"
    )

    metrics = compute_regression_metrics(
        y_true=y_test,
        y_pred=y_pred,
    )

    return metrics


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
            "standard_tuning_strategies": STANDARD_TUNING_STRATEGIES,
            "regression_finetune_modes_by_model": REGRESSION_FINETUNE_MODES_BY_MODEL,
            "ml_gridsearch_cv_splits": ML_GRIDSEARCH_CV_SPLITS,
            "ml_gridsearch_scoring": ML_GRIDSEARCH_SCORING,
            "ml_param_grids": {k: v for k, v in ML_PARAM_GRIDS.items() if k in ML_BASELINES},
            "optional_packages": {
                "catboost": CATBOOST_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "lightgbm": LIGHTGBM_AVAILABLE,
            },
            "preprocessings": preprocessings,
            "ae_latent_dims": AE_LATENT_DIMS,
            "tabpfn_emb_pca_dims": TABPFN_EMB_PCA_DIMS if USE_TABPFN_EMB_PCA else [],
            "train_fractions": TRAIN_FRACTIONS,
            "excluded_datasets": sorted(EXCLUDED_DATASETS),
            "n_splits": N_SPLITS,
            "device": DEVICE,
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
        log(f"DATASET: {dataset_name}")

        try:
            X, y, groups = load_single_dataset(ds_cfg)
        except Exception as e:
            log(f"[WARNING] Failed loading dataset {dataset_name}: {e}")
            continue

        log(
            f"{dataset_name} | shape={X.shape} | "
            f"y_min={np.min(y):.4f} | y_max={np.max(y):.4f} | y_mean={np.mean(y):.4f}"
        )

        if len(y) < N_SPLITS:
            log(f"[WARNING] Skipping {dataset_name}: too few samples for {N_SPLITS}-fold CV")
            continue

        n_unique_groups = len(np.unique(groups))
        if n_unique_groups < N_SPLITS:
            log(
                f"[WARNING] Skipping {dataset_name}: only {n_unique_groups} groups "
                f"available, but N_SPLITS={N_SPLITS}"
            )
            continue

        cv = GroupKFold(n_splits=N_SPLITS)
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
                    f"n_train={len(Xtr)} | n_test={len(Xte)}"
                )

                if len(Xtr) < 2:
                    log(f"[WARNING] Skipping frac={frac:.2f} for {dataset_name}: too few training samples")
                    continue

                for prep in preprocessings:
                    try:
                        Xtr_p, Xte_p, prep_meta = apply_preprocessing(prep, Xtr, ytr, Xte)
                    except Exception as e:
                        log(
                            f"[WARNING] dataset={dataset_name} preprocessing={prep} failed: {e}"
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
                                    f"frac={frac:.2f} | MAE={metrics['mae']:.4f} | "
                                    f"RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
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
                as_index=False
            )
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                ml_best_cv_score_mean=("ml_best_cv_score", "mean"),
                ml_best_cv_score_std=("ml_best_cv_score", "std"),

            )
            .sort_values(
                ["train_fraction", "rmse_mean"],
                ascending=[True, True]
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
            as_index=False
        )
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            ml_best_cv_score_mean=("ml_best_cv_score", "mean"),
            ml_best_cv_score_std=("ml_best_cv_score", "std"),
        )
        .sort_values(
            ["dataset", "train_fraction", "rmse_mean"],
            ascending=[True, True, True]
        )
    )

    all_summary_path = os.path.join(OUTPUT_DIR, "all_datasets_summary.csv")
    all_summary.to_csv(all_summary_path, index=False)

    log(f"Saved: {all_fold_path}")
    log(f"Saved: {all_summary_path}")
    log("Benchmark completed.")


if __name__ == "__main__":
    main()