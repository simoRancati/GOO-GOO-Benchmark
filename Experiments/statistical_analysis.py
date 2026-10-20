#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL GOO-GOO ANALYSIS — 2026-09-06
===================================

This script builds the final classification benchmark by combining:
  1) the corrected binary-classification rerun (macro-AUPRC), and
  2) the original multiclass results (whose AUPRC was already macro-OvR),
then reruns the full classification + regression analyses used in the paper.

It also performs an old-vs-fixed audit to show whether the rerun changed only
AUPRC or also predictions / balanced accuracy / AUROC / preprocessing dimension.

Expected classification tasks:
  Binary rerun (9):
    myocardial_infarction, parkinson, hepatitis, glioma, thyroid_recurrence,
    aids_death, autism_detection, cervical_cancer, radnkbench
  Multiclass retained from old results (5):
    cirrhosis, obesity_prediction, Gh_T0, Gh_T1, Gh_T2

Hard exclusions:
  classification: student_depression
  regression: health_insurance

Statistical unit:
  dataset, with Gh_T0/Gh_T1/Gh_T2 averaged into one GH cohort for inferential
  classification comparisons and correlations.

Main outputs:
  - classification_final_merged.csv
  - old_vs_fixed_binary_audit.csv
  - old_vs_final_best_BA_comparison.csv
  - tables/*.csv
  - figures/*.png
  - analysis_report.txt
  - manuscript_update_notes.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FRACTIONS = [0.10, 0.20, 0.50, 0.75, 1.00]
FRAC_LABELS = ["10%", "20%", "50%", "75%", "100%"]
SEED = 20260906
N_BOOT = 2000

EXCLUDED_CLASS = {"student_depression"}
EXCLUDED_REG = {"health_insurance"}
EXPECTED_BINARY = {
    "myocardial_infarction", "parkinson", "hepatitis", "glioma",
    "thyroid_recurrence", "aids_death", "autism_detection",
    "cervical_cancer", "radnkbench",
}
EXPECTED_MULTICLASS = {"cirrhosis", "obesity_prediction", "Gh_T0", "Gh_T1", "Gh_T2"}
GH_TASKS = {"Gh_T0", "Gh_T1", "Gh_T2"}
GH_UNIT = "GH_cohort"

TFM_CLASS = {"TabPFN", "TabDPT", "TabICL", "TabICLv2", "Mitra", "ContextTab", "OrionMSP", "OrionBix"}
TFM_REG = {"TabPFN", "TabICLv2", "TabDPT", "Mitra", "ContextTab", "Limix"}

CLASS_METRICS = ["balanced_accuracy", "precision_macro", "recall_macro", "auroc_ovr_macro", "auprc_macro"]
REG_METRICS = ["mae", "rmse", "r2"]

CLASS_DESC = [
    "n_samples", "n_features", "features_per_sample", "fraction_categorical",
    "missing_fraction", "fraction_minority_class", "imbalance_ratio_max_min",
    "mean_abs_feature_correlation", "pca_n_components_95",
    "outlier_fraction_isolationforest", "target_entropy_HY", "IG_mean", "GR_mean",
    "univariate_auc_mean", "univariate_auc_max", "n_significant_features_holm_005",
]
REG_DESC = [
    "n_samples", "n_features", "features_per_sample", "fraction_categorical",
    "missing_fraction", "mean_abs_feature_correlation", "pca_n_components_95",
    "outlier_fraction_isolationforest", "target_std", "target_skewness",
    "target_entropy_HY", "IG_mean", "GR_mean", "n_significant_features_holm_005",
]
DESC_LABEL = {
    "n_samples": "N samples", "n_features": "N features",
    "features_per_sample": "Features / samples", "fraction_categorical": "Fraction categorical",
    "missing_fraction": "Missing fraction", "fraction_minority_class": "Minority-class fraction",
    "imbalance_ratio_max_min": "Imbalance ratio", "mean_abs_feature_correlation": "Mean |feature corr.|",
    "pca_n_components_95": "PCA components (95%)", "outlier_fraction_isolationforest": "Outlier fraction",
    "target_entropy_HY": "Target entropy", "IG_mean": "Mean information gain", "GR_mean": "Mean gain ratio",
    "univariate_auc_mean": "Mean univariate AUC", "univariate_auc_max": "Maximum univariate AUC",
    "n_significant_features_holm_005": "N significant features", "target_std": "Target SD",
    "target_skewness": "Target skewness",
}

DISPLAY_DATASET = {
    "myocardial_infarction": "Myocardial infarction", "parkinson": "Parkinson",
    "hepatitis": "Hepatitis", "cirrhosis": "Cirrhosis", "glioma": "Glioma",
    "thyroid_recurrence": "Thyroid recurrence", "aids_death": "AIDS death",
    "autism_detection": "Autism detection", "obesity_prediction": "Obesity prediction",
    "Gh_T0": "GH T0", "Gh_T1": "GH T1", "Gh_T2": "GH T2",
    "cervical_cancer": "Cervical cancer", "radnkbench": "radMLBench",
    "parkinson_telemonitoring": "Parkinson telemonitoring",
    "infrared_thermography": "Infrared thermography", "cancer_regression": "Cancer regression",
    "heart_risk_prediction": "Heart risk prediction", "protein_structure": "Protein structure",
}
DISPLAY_MODEL = {
    "LogReg": "Logistic regression", "RandomForest": "Random forest",
    "HistGB": "Hist gradient boosting", "ExtraTrees": "Extra Trees",
    "GradientBoosting": "Gradient boosting", "SVM_RBF": "RBF SVM", "SVR_RBF": "RBF SVR",
    "KNN": "K-nearest neighbors", "MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
    "LightGBM": "LightGBM", "LinearRegression": "Linear regression", "Ridge": "Ridge",
    "ElasticNet": "Elastic Net", "Limix": "LimiX",
}

# -----------------------------------------------------------------------------
# Utility / statistics
# -----------------------------------------------------------------------------
def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    pv = p[mask]
    order = np.argsort(pv)
    sp = pv[order]
    m = len(sp)
    adj = np.maximum.accumulate([(m - i) * sp[i] for i in range(m)])
    adj = np.minimum(adj, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[np.where(mask)[0]] = restored
    return out


def exact_wilcoxon(values: Iterable[float]):
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    x_nz = x[x != 0]
    if len(x_nz) == 0:
        return np.nan, np.nan, "not_computable", len(x), 0
    try:
        res = stats.wilcoxon(x_nz, zero_method="wilcox", alternative="two-sided", method="exact")
        method = "exact"
    except ValueError:
        res = stats.wilcoxon(x_nz, zero_method="wilcox", alternative="two-sided", method="auto")
        method = "auto"
    return float(res.statistic), float(res.pvalue), method, len(x), len(x_nz)


def bootstrap_ci(values, statistic="mean", n_boot=N_BOOT, seed=SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(x, size=(n_boot, len(x)), replace=True)
    vals = sample.mean(axis=1) if statistic == "mean" else np.median(sample, axis=1)
    return tuple(np.quantile(vals, [0.025, 0.975]).astype(float))


def summarize(values, seed=SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    stat, p, method, n_total, n_nonzero = exact_wilcoxon(x)
    mean_ci = bootstrap_ci(x, "mean", seed=seed)
    med_ci = bootstrap_ci(x, "median", seed=seed + 1)
    return {
        "n_units": n_total, "n_nonzero": n_nonzero,
        "mean_difference": float(np.mean(x)) if len(x) else np.nan,
        "mean_ci95_low": mean_ci[0], "mean_ci95_high": mean_ci[1],
        "median_difference": float(np.median(x)) if len(x) else np.nan,
        "median_ci95_low": med_ci[0], "median_ci95_high": med_ci[1],
        "n_tfm_better": int(np.sum(x > 0)), "n_ml_better": int(np.sum(x < 0)),
        "n_ties": int(np.sum(x == 0)), "wilcoxon_statistic": stat,
        "p_raw": p, "wilcoxon_method": method,
    }


def normalize_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")

# -----------------------------------------------------------------------------
# Loading / merging / audit
# -----------------------------------------------------------------------------
def setting_label(row):
    if row["family"] == "ML":
        return "grid-search baseline"
    if row["tuning_strategy"] == "inference":
        return "inference"
    if row["tuning_strategy"] == "base-ft":
        return "fine-tune (base-ft)"
    if row["tuning_strategy"] == "peft":
        return "fine-tune (PEFT)"
    mode = str(row.get("finetune_mode", "")).replace("_", "-")
    return f"fine-tune ({mode})"


def annotate_class(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["family"] = np.where(c["model_name"].isin(TFM_CLASS), "TFM", "ML")
    def strat(r):
        if r.family == "ML": return "ML models"
        if r.tuning_strategy == "inference": return "TFM - In-Context"
        if r.tuning_strategy == "base-ft": return "TFM - base-ft"
        if r.tuning_strategy == "peft": return "TFM - PEFT"
        if r.tuning_strategy == "finetune" and str(r.finetune_mode) == "meta-learning": return "TFM - meta-learning"
        if r.tuning_strategy == "finetune" and str(r.finetune_mode) == "sft": return "TFM - SFT"
        return "TFM - other"
    c["strategy_group"] = c.apply(strat, axis=1)
    c["setting"] = c.apply(setting_label, axis=1)
    c["dataset_display"] = c["dataset"].map(DISPLAY_DATASET).fillna(c["dataset"])
    c["model_display"] = c["model_name"].map(DISPLAY_MODEL).fillna(c["model_name"])
    return c


def annotate_reg(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["family"] = np.where(r["model_name"].isin(TFM_REG), "TFM", "ML")
    r["strategy_group"] = np.where(
        r["family"].eq("ML"), "ML models",
        np.where(r["tuning_strategy"].eq("inference"), "TFM - In-Context", "TFM - fine-tuning")
    )
    r["setting"] = r.apply(setting_label, axis=1)
    r["dataset_display"] = r["dataset"].map(DISPLAY_DATASET).fillna(r["dataset"])
    r["model_display"] = r["model_name"].map(DISPLAY_MODEL).fillna(r["model_name"])
    return r


def load_and_merge(args, out_dir: Path):
    old = pd.read_csv(args.old_class)
    fixed = pd.read_csv(args.fixed_binary)
    reg = pd.read_csv(args.reg)

    old = old[~old["dataset"].astype(str).str.lower().isin(EXCLUDED_CLASS)].copy()
    reg = reg[~reg["dataset"].astype(str).str.lower().isin(EXCLUDED_REG)].copy()

    fixed_set = set(fixed["dataset"].unique())
    if fixed_set != EXPECTED_BINARY:
        raise RuntimeError(f"Fixed binary dataset set mismatch. Found={sorted(fixed_set)} expected={sorted(EXPECTED_BINARY)}")
    old_multiclass = old[old["dataset"].isin(EXPECTED_MULTICLASS)].copy()
    if set(old_multiclass["dataset"].unique()) != EXPECTED_MULTICLASS:
        raise RuntimeError("Not all expected multiclass datasets are present in old classification results")

    # Check corrected macro AUPRC for binary rows where class-specific values exist.
    if {"auprc_class_0", "auprc_class_1", "auprc_macro"}.issubset(fixed.columns):
        z = fixed[["auprc_class_0", "auprc_class_1", "auprc_macro"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(z):
            max_err = np.max(np.abs(z.auprc_macro - (z.auprc_class_0 + z.auprc_class_1) / 2.0))
            if max_err > 1e-10:
                raise RuntimeError(f"Macro-AUPRC validation failed; max error={max_err}")

    # Preserve all columns. Multiclass rows naturally have NaN for binary diagnostic fields.
    final = pd.concat([fixed, old_multiclass], ignore_index=True, sort=False)
    if set(final["dataset"].unique()) != EXPECTED_BINARY | EXPECTED_MULTICLASS:
        raise RuntimeError("Final classification dataset set is not exactly the expected 14 tasks")

    old = old.reset_index(drop=True); fixed = fixed.reset_index(drop=True); final = final.reset_index(drop=True); reg = reg.reset_index(drop=True)
    old["_source_order"] = np.arange(len(old)); fixed["_source_order"] = np.arange(len(fixed)); final["_source_order"] = np.arange(len(final)); reg["_source_order"] = np.arange(len(reg))

    old = annotate_class(old); fixed = annotate_class(fixed); final = annotate_class(final); reg = annotate_reg(reg)

    final.to_csv(out_dir / "classification_final_merged.csv", index=False)
    reg.to_csv(out_dir / "regression_final_excluding_health_insurance.csv", index=False)
    return old, fixed, final, reg


def audit_old_vs_fixed(old, fixed):
    keys = ["dataset", "fold", "train_fraction", "preprocessing", "model_name", "tuning_strategy", "finetune_mode"]
    cols_old = keys + ["embedding_dim", "balanced_accuracy", "precision_macro", "recall_macro", "auroc_ovr_macro", "auprc_macro"]
    cols_new = keys + ["embedding_dim", "balanced_accuracy", "precision_macro", "recall_macro", "auroc_ovr_macro", "auprc_macro",
                       "auprc_class_0", "auprc_class_1", "auprc_class_0_label", "auprc_class_1_label", "auprc_legacy"]
    a = old[old["dataset"].isin(EXPECTED_BINARY)][cols_old].copy()
    b = fixed[cols_new].copy()
    a = a.rename(columns={c: c + "_old" for c in cols_old if c not in keys})
    b = b.rename(columns={c: c + "_fixed" for c in cols_new if c not in keys})
    m = a.merge(b, on=keys, how="inner")
    for c in ["embedding_dim", "balanced_accuracy", "precision_macro", "recall_macro", "auroc_ovr_macro", "auprc_macro"]:
        m[f"delta_{c}_fixed_minus_old"] = pd.to_numeric(m[f"{c}_fixed"], errors="coerce") - pd.to_numeric(m[f"{c}_old"], errors="coerce")
    m["delta_fixed_macro_minus_legacy"] = pd.to_numeric(m["auprc_macro_fixed"], errors="coerce") - pd.to_numeric(m["auprc_legacy_fixed"], errors="coerce")
    m["delta_newlegacy_minus_old_auprc"] = pd.to_numeric(m["auprc_legacy_fixed"], errors="coerce") - pd.to_numeric(m["auprc_macro_old"], errors="coerce")
    m["ba_exactly_same"] = np.isclose(pd.to_numeric(m["balanced_accuracy_fixed"], errors="coerce"), pd.to_numeric(m["balanced_accuracy_old"], errors="coerce"), atol=1e-12, equal_nan=False)
    m["embedding_dim_same"] = np.isclose(pd.to_numeric(m["embedding_dim_fixed"], errors="coerce"), pd.to_numeric(m["embedding_dim_old"], errors="coerce"), atol=1e-12, equal_nan=False)

    ds = m.groupby("dataset", as_index=False).agg(
        n_common=("dataset", "size"),
        pct_BA_exactly_same=("ba_exactly_same", lambda x: 100 * np.mean(x)),
        mean_abs_BA_change=("delta_balanced_accuracy_fixed_minus_old", lambda x: np.nanmean(np.abs(x))),
        mean_abs_AUROC_change=("delta_auroc_ovr_macro_fixed_minus_old", lambda x: np.nanmean(np.abs(x))),
        mean_macro_minus_legacy_AUPRC=("delta_fixed_macro_minus_legacy", "mean"),
        median_macro_minus_legacy_AUPRC=("delta_fixed_macro_minus_legacy", "median"),
        mean_abs_newlegacy_vs_old_AUPRC=("delta_newlegacy_minus_old_auprc", lambda x: np.nanmean(np.abs(x))),
        pct_embedding_dim_same=("embedding_dim_same", lambda x: 100 * np.mean(x)),
    )
    return m, ds

# -----------------------------------------------------------------------------
# Configuration summaries / selection
# -----------------------------------------------------------------------------
def config_summary(df: pd.DataFrame, metrics):
    keys = ["dataset", "dataset_display", "train_fraction", "model_name", "model_display",
            "tuning_strategy", "finetune_mode", "setting", "family", "strategy_group"]
    agg = {"source_order": ("_source_order", "min"), "n_folds": ("fold", "nunique")}
    for m in metrics:
        agg[m] = (m, "mean"); agg[m + "_sd"] = (m, "std"); agg[m + "_n"] = (m, "count")
    out = df.groupby(keys, dropna=False, sort=False).agg(**agg).reset_index()
    out = out.sort_values("source_order", kind="stable").reset_index(drop=True)
    out["configuration_order"] = np.arange(len(out))
    return out


def select_best(summary, metric, maximize, group_cols):
    rows = []
    for _, g in summary.dropna(subset=[metric]).groupby(group_cols, sort=False, dropna=False):
        best = g[metric].max() if maximize else g[metric].min()
        tied = g[g[metric] == best].sort_values("configuration_order", kind="stable")
        row = tied.iloc[0].copy()
        row["n_exact_ties"] = len(tied)
        row["exact_tie_configurations"] = " | ".join(tied.apply(lambda x: f"{x.model_name}::{x.setting}", axis=1))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def family_diff(summary, metric, maximize):
    best = select_best(summary, metric, maximize, ["dataset", "dataset_display", "train_fraction", "family"])
    w = best.pivot(index=["dataset", "dataset_display", "train_fraction"], columns="family", values=metric).reset_index()
    w = w.rename(columns={"ML": "ml_score", "TFM": "tfm_score"})
    w["metric"] = metric
    w["tfm_advantage"] = (w.tfm_score - w.ml_score) if maximize else (w.ml_score - w.tfm_score)
    return w, best


def aggregate_gh_diff(diff):
    d = diff.copy()
    d["statistical_unit"] = np.where(d["dataset"].isin(GH_TASKS), GH_UNIT, d["dataset"])
    return d.groupby(["train_fraction", "statistical_unit"], sort=False).agg(
        tfm_advantage=("tfm_advantage", "mean"), n_component_tasks=("dataset", "nunique")
    ).reset_index()

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
def analyze_classification(c):
    cfg = config_summary(c, CLASS_METRICS)

    curve = c.groupby(["strategy_group", "train_fraction"], sort=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        precision_macro_mean=("precision_macro", "mean"),
        recall_macro_mean=("recall_macro", "mean"),
        auroc_mean=("auroc_ovr_macro", "mean"), auprc_mean=("auprc_macro", "mean"),
        n_rows=("balanced_accuracy", "size"), n_datasets=("dataset", "nunique")
    ).reset_index()

    dataset_curve = c.groupby(["dataset", "strategy_group", "train_fraction"], sort=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"), precision_macro=("precision_macro", "mean"),
        recall_macro=("recall_macro", "mean"), auroc=("auroc_ovr_macro", "mean"), auprc=("auprc_macro", "mean")
    ).reset_index()
    dataset_balanced_curve = dataset_curve.groupby(["strategy_group", "train_fraction"], sort=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"), precision_macro_mean=("precision_macro", "mean"),
        recall_macro_mean=("recall_macro", "mean"), auroc_mean=("auroc", "mean"), auprc_mean=("auprc", "mean"),
        n_datasets=("dataset", "nunique")
    ).reset_index()

    best_overall = select_best(cfg, "balanced_accuracy", True, ["dataset", "dataset_display", "train_fraction"])
    winner_fraction = best_overall.groupby(["train_fraction", "family"]).size().rename("n_selected").reset_index()
    winner_model = best_overall.groupby(["model_name", "model_display", "family"]).size().rename("n_selected").reset_index().sort_values("n_selected", ascending=False)
    winner_model["percentage_of_family_wins"] = winner_model.groupby("family")["n_selected"].transform(lambda x: 100 * x / x.sum())
    tfm_setting = best_overall[best_overall.family == "TFM"].groupby(["setting"]).size().rename("n_selected").reset_index().sort_values("n_selected", ascending=False)
    tfm_setting["percentage_of_TFM_wins"] = 100 * tfm_setting.n_selected / tfm_setting.n_selected.sum()

    diffs = []; best_family = []; stat_tables = []
    for i, (m, maximize) in enumerate([("balanced_accuracy", True), ("auroc_ovr_macro", True), ("auprc_macro", True)]):
        diff, best = family_diff(cfg, m, maximize); diffs.append(diff); best_family.append(best.assign(metric=m))
        units = aggregate_gh_diff(diff)
        rows = []
        for j, f in enumerate(FRACTIONS):
            vals = units[np.isclose(units.train_fraction, f)].tfm_advantage.to_numpy()
            s = summarize(vals, seed=SEED + i * 10000 + j * 100); s.update(metric=m, train_fraction=f); rows.append(s)
        st = pd.DataFrame(rows); st["p_holm"] = holm_adjust(st.p_raw); stat_tables.append(st)
    diffs = pd.concat(diffs, ignore_index=True); best_family = pd.concat(best_family, ignore_index=True); stats_all = pd.concat(stat_tables, ignore_index=True)

    # Best TFM inference / fine-tuning vs best ML for each metric (plots)
    strategy_metric = {}
    for m in ["balanced_accuracy", "auprc_macro", "auroc_ovr_macro"]:
        ml = select_best(cfg[cfg.family == "ML"], m, True, ["dataset", "dataset_display", "train_fraction"])
        parts = []
        for filt, name in [
            ((cfg.family == "TFM") & (cfg.tuning_strategy == "inference"), "TFM In-Context"),
            ((cfg.family == "TFM") & (cfg.tuning_strategy != "inference"), "TFM fine-tuning"),
        ]:
            tf = select_best(cfg[filt], m, True, ["dataset", "dataset_display", "train_fraction"])
            z = tf[["dataset", "dataset_display", "train_fraction", m]].rename(columns={m: "tfm_score"}).merge(
                ml[["dataset", "train_fraction", m]].rename(columns={m: "ml_score"}), on=["dataset", "train_fraction"]
            )
            z["strategy"] = name; z["advantage"] = z.tfm_score - z.ml_score; parts.append(z)
        strategy_metric[m] = pd.concat(parts, ignore_index=True)

    # Paper Table 4: select by BA, then report metrics from the same configuration
    table4 = best_overall[["dataset", "dataset_display", "train_fraction", "model_name", "model_display", "setting", "family",
                           "balanced_accuracy", "precision_macro", "auroc_ovr_macro", "auprc_macro", "n_folds"]].copy()
    table4 = table4.sort_values(["dataset_display", "train_fraction"])

    return {
        "classification_configuration_summary": cfg,
        "classification_strategy_curve": curve,
        "classification_strategy_curve_dataset_balanced": dataset_balanced_curve,
        "classification_oracle_best_overall_ba": best_overall,
        "classification_oracle_winners_by_fraction": winner_fraction,
        "classification_oracle_winners_by_model": winner_model,
        "classification_oracle_tfm_setting_counts": tfm_setting,
        "classification_oracle_differences": diffs,
        "classification_oracle_best_by_family": best_family,
        "classification_oracle_statistics": stats_all,
        "classification_strategy_advantage_ba": strategy_metric["balanced_accuracy"],
        "classification_strategy_advantage_auprc": strategy_metric["auprc_macro"],
        "classification_strategy_advantage_auroc": strategy_metric["auroc_ovr_macro"],
        "Table_4_best_classification_by_BA": table4,
    }

# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------
def analyze_regression(r):
    cfg = config_summary(r, REG_METRICS)
    strategy_dataset = r.groupby(["dataset", "dataset_display", "train_fraction", "strategy_group"], sort=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean")
    ).reset_index()

    rank_rows = []
    for f in FRACTIONS:
        p = strategy_dataset[np.isclose(strategy_dataset.train_fraction, f)]
        ranks = []
        for metric, asc in [("mae", True), ("rmse", True), ("r2", False)]:
            w = p.pivot(index="dataset", columns="strategy_group", values=metric)
            long = w.rank(axis=1, ascending=asc, method="average").stack().rename(metric + "_rank").reset_index()
            ranks.append(long)
        m = ranks[0].merge(ranks[1], on=["dataset", "strategy_group"]).merge(ranks[2], on=["dataset", "strategy_group"])
        m["train_fraction"] = f; rank_rows.append(m)
    ranks = pd.concat(rank_rows, ignore_index=True)
    mean_ranks = ranks.groupby(["train_fraction", "strategy_group"], sort=False).agg(
        mean_mae_rank=("mae_rank", "mean"), mean_rmse_rank=("rmse_rank", "mean"),
        mean_r2_rank=("r2_rank", "mean"), n_datasets=("dataset", "nunique")
    ).reset_index()

    best_overall = select_best(cfg, "rmse", False, ["dataset", "dataset_display", "train_fraction"])
    winner_model = best_overall.groupby(["model_name", "model_display", "family"]).size().rename("n_selected").reset_index().sort_values("n_selected", ascending=False)

    best_family = select_best(cfg, "rmse", False, ["dataset", "dataset_display", "train_fraction", "family"])
    sel = best_family[["dataset", "dataset_display", "train_fraction", "family", "model_name", "model_display", "setting", "mae", "rmse", "r2", "n_exact_ties"]]
    w = sel.pivot(index=["dataset", "dataset_display", "train_fraction"], columns="family", values=["mae", "rmse", "r2"]).reset_index()
    w.columns = ["dataset", "dataset_display", "train_fraction"] + [f"{a}_{b}" for a, b in w.columns[3:]]
    model = sel.pivot(index=["dataset", "dataset_display", "train_fraction"], columns="family", values="model_name").reset_index().rename(columns={"ML": "ml_model", "TFM": "tfm_model"})
    setting = sel.pivot(index=["dataset", "dataset_display", "train_fraction"], columns="family", values="setting").reset_index().rename(columns={"ML": "ml_setting", "TFM": "tfm_setting"})
    diff = w.merge(model, on=["dataset", "dataset_display", "train_fraction"]).merge(setting, on=["dataset", "dataset_display", "train_fraction"])
    diff["relative_rmse_reduction_pct"] = (diff.rmse_ML - diff.rmse_TFM) / diff.rmse_ML * 100
    diff["relative_mae_reduction_pct"] = (diff.mae_ML - diff.mae_TFM) / diff.mae_ML * 100
    diff["r2_advantage_tfm_minus_ml"] = diff.r2_TFM - diff.r2_ML

    summary = diff.groupby("train_fraction", sort=False).agg(
        median_relative_rmse_reduction_pct=("relative_rmse_reduction_pct", "median"),
        mean_relative_rmse_reduction_pct=("relative_rmse_reduction_pct", "mean"),
        median_relative_mae_reduction_pct=("relative_mae_reduction_pct", "median"),
        mean_relative_mae_reduction_pct=("relative_mae_reduction_pct", "mean"),
        mean_r2_tfm=("r2_TFM", "mean"), mean_r2_ml=("r2_ML", "mean"),
        mean_r2_advantage=("r2_advantage_tfm_minus_ml", "mean"), n_datasets=("dataset", "nunique")
    ).reset_index()

    stats_rows = []
    for i, col in enumerate(["relative_rmse_reduction_pct", "relative_mae_reduction_pct", "r2_advantage_tfm_minus_ml"]):
        rows = []
        for j, f in enumerate(FRACTIONS):
            vals = diff[np.isclose(diff.train_fraction, f)][col].to_numpy()
            s = summarize(vals, seed=SEED + 60000 + i * 10000 + j * 100); s.update(metric=col, train_fraction=f); rows.append(s)
        st = pd.DataFrame(rows); st["p_holm"] = holm_adjust(st.p_raw); stats_rows.append(st)
    stats_all = pd.concat(stats_rows, ignore_index=True)

    ml = select_best(cfg[cfg.family == "ML"], "rmse", False, ["dataset", "dataset_display", "train_fraction"])
    parts = []
    for filt, name in [
        ((cfg.family == "TFM") & (cfg.tuning_strategy == "inference"), "TFM In-Context"),
        ((cfg.family == "TFM") & (cfg.tuning_strategy != "inference"), "TFM fine-tuning"),
    ]:
        tf = select_best(cfg[filt], "rmse", False, ["dataset", "dataset_display", "train_fraction"])
        z = tf[["dataset", "dataset_display", "train_fraction", "mae", "rmse", "r2"]].rename(columns={"mae": "tfm_mae", "rmse": "tfm_rmse", "r2": "tfm_r2"}).merge(
            ml[["dataset", "train_fraction", "mae", "rmse", "r2"]].rename(columns={"mae": "ml_mae", "rmse": "ml_rmse", "r2": "ml_r2"}), on=["dataset", "train_fraction"]
        )
        z["strategy"] = name
        z["relative_rmse_reduction_pct"] = (z.ml_rmse - z.tfm_rmse) / z.ml_rmse * 100
        z["relative_mae_reduction_pct"] = (z.ml_mae - z.tfm_mae) / z.ml_mae * 100
        z["r2_advantage"] = z.tfm_r2 - z.ml_r2
        parts.append(z)
    strategy_adv = pd.concat(parts, ignore_index=True)

    table5 = best_overall[["dataset", "dataset_display", "train_fraction", "model_name", "model_display", "setting", "family", "mae", "rmse", "r2", "n_folds"]].copy().sort_values(["dataset_display", "train_fraction"])

    return {
        "regression_configuration_summary": cfg,
        "regression_strategy_by_dataset": strategy_dataset,
        "regression_strategy_ranks": ranks,
        "regression_strategy_mean_ranks": mean_ranks,
        "regression_oracle_best_overall_rmse": best_overall,
        "regression_oracle_winners_by_model": winner_model,
        "regression_oracle_best_by_family_rmse": best_family,
        "regression_oracle_differences": diff,
        "regression_oracle_summary": summary,
        "regression_oracle_statistics": stats_all,
        "regression_strategy_advantage": strategy_adv,
        "Table_5_best_regression_by_RMSE": table5,
    }

# -----------------------------------------------------------------------------
# Model-level Friedman + pairwise Wilcoxon/Holm
# -----------------------------------------------------------------------------
def aggregate_gh_model_table(b):
    z = b.copy()
    z["statistical_unit"] = np.where(z.dataset.isin(GH_TASKS), GH_UNIT, z.dataset)
    value_cols = [c for c in ["balanced_accuracy", "auroc_ovr_macro", "auprc_macro"] if c in z.columns]
    return z.groupby(["statistical_unit", "train_fraction", "model_name"], as_index=False)[value_cols].mean()


def model_level_tests(cfg, task="classification"):
    if task == "classification":
        metric_spec = [("balanced_accuracy", True), ("auroc_ovr_macro", True), ("auprc_macro", True)]
    else:
        metric_spec = [("mae", False), ("rmse", False), ("r2", True)]
    omnibus = []; pairwise_all = []
    for metric, maximize in metric_spec:
        best_model = select_best(cfg, metric, maximize, ["dataset", "train_fraction", "model_name"])
        if task == "classification":
            b = aggregate_gh_model_table(best_model)
            unit_col = "statistical_unit"
        else:
            b = best_model[["dataset", "train_fraction", "model_name", metric]].copy(); unit_col = "dataset"
        for f in FRACTIONS:
            p = b[np.isclose(b.train_fraction, f)].pivot(index=unit_col, columns="model_name", values=metric)
            p = p.dropna(axis=1, how="any").dropna(axis=0, how="any")
            if p.shape[1] < 3 or p.shape[0] < 2:
                continue
            fr = stats.friedmanchisquare(*[p[c].to_numpy(float) for c in p.columns])
            omnibus.append({"task": task, "metric": metric, "train_fraction": f, "n_units": p.shape[0], "n_models": p.shape[1],
                            "friedman_statistic": float(fr.statistic), "friedman_p": float(fr.pvalue), "models": " | ".join(p.columns)})
            if fr.pvalue < 0.05:
                local = []
                cols = list(p.columns)
                for i, a in enumerate(cols):
                    for bname in cols[i + 1:]:
                        diff = p[a].to_numpy(float) - p[bname].to_numpy(float)
                        stat, pv, method, n_total, n_nonzero = exact_wilcoxon(diff)
                        local.append({"task": task, "metric": metric, "train_fraction": f, "model_a": a, "model_b": bname,
                                      "n_units": n_total, "wilcoxon_statistic": stat, "p_raw": pv, "method": method})
                q = holm_adjust([x["p_raw"] for x in local])
                for x, qq in zip(local, q):
                    x["p_holm"] = qq; x["significant_holm_0_05"] = bool(np.isfinite(qq) and qq < 0.05)
                pairwise_all.extend(local)
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise_all)

# -----------------------------------------------------------------------------
# Correlations
# -----------------------------------------------------------------------------
def gh_aggregate_characteristics(chars, descriptors):
    chars = chars.copy(); chars["unit"] = np.where(chars.dataset.isin(GH_TASKS), GH_UNIT, chars.dataset)
    agg = {d: (d, "mean") for d in descriptors}
    return chars.groupby("unit", sort=False).agg(**agg).reset_index().rename(columns={"unit": "dataset_unit"})


def corr_table(merged, outcome_cols, descriptors, analysis_name):
    rows = []
    for outcome in outcome_cols:
        for desc in descriptors:
            x = pd.to_numeric(merged[desc], errors="coerce").to_numpy(float)
            y = pd.to_numeric(merged[outcome], errors="coerce").to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
                rho = p = np.nan
            else:
                res = stats.spearmanr(x[mask], y[mask]); rho = float(res.statistic); p = float(res.pvalue)
            rows.append({"analysis": analysis_name, "outcome": outcome, "descriptor": desc, "n_units": int(mask.sum()), "spearman_rho": rho, "p_raw": p})
    tab = pd.DataFrame(rows); tab["p_holm"] = np.nan
    for _, idx in tab.groupby("outcome", sort=False).groups.items():
        tab.loc[idx, "p_holm"] = holm_adjust(tab.loc[idx, "p_raw"])
    return tab


def analyze_correlations(class_tables, reg_tables, class_char_path, reg_char_path):
    cc = pd.read_csv(class_char_path); rc = pd.read_csv(reg_char_path)
    cchars = gh_aggregate_characteristics(cc, CLASS_DESC)
    rchars = rc[["dataset"] + REG_DESC].rename(columns={"dataset": "dataset_unit"})

    cd = class_tables["classification_oracle_differences"]
    cd = cd[np.isclose(cd.train_fraction, 1.0)].copy()
    cd["dataset_unit"] = np.where(cd.dataset.isin(GH_TASKS), GH_UNIT, cd.dataset)
    cd = cd.groupby(["dataset_unit", "metric"], sort=False).agg(tfm_advantage=("tfm_advantage", "mean")).reset_index()
    cwide = cd.pivot(index="dataset_unit", columns="metric", values="tfm_advantage").reset_index().rename(columns={
        "balanced_accuracy": "BA advantage", "auroc_ovr_macro": "AUROC advantage", "auprc_macro": "AUPRC advantage"
    })
    cmerge = cwide.merge(cchars, on="dataset_unit", how="inner")
    ctab = corr_table(cmerge, ["BA advantage", "AUROC advantage", "AUPRC advantage"], CLASS_DESC, "classification_family_advantage")

    rd = reg_tables["regression_oracle_differences"]
    rd = rd[np.isclose(rd.train_fraction, 1.0)].copy().rename(columns={"dataset": "dataset_unit"})
    rmerge = rd.merge(rchars, on="dataset_unit", how="inner")
    rtab = corr_table(rmerge, ["relative_rmse_reduction_pct", "relative_mae_reduction_pct", "r2_advantage_tfm_minus_ml"], REG_DESC, "regression_family_advantage")

    # Supplementary absolute performance correlations at full data
    class_cfg = class_tables["classification_configuration_summary"]; class_abs = []
    for metric, maximize in [("balanced_accuracy", True), ("auroc_ovr_macro", True), ("auprc_macro", True)]:
        for label, filt in [
            ("ML", class_cfg.family == "ML"),
            ("TFM In-Context", (class_cfg.family == "TFM") & (class_cfg.tuning_strategy == "inference")),
            ("TFM fine-tuning", (class_cfg.family == "TFM") & (class_cfg.tuning_strategy != "inference")),
        ]:
            b = select_best(class_cfg[filt], metric, maximize, ["dataset", "train_fraction"])
            b = b[np.isclose(b.train_fraction, 1.0)][["dataset", metric]]
            b["group"] = label; b["metric"] = metric; b = b.rename(columns={metric: "performance"}); class_abs.append(b)
    ca = pd.concat(class_abs); ca["dataset_unit"] = np.where(ca.dataset.isin(GH_TASKS), GH_UNIT, ca.dataset)
    ca = ca.groupby(["dataset_unit", "metric", "group"]).performance.mean().reset_index(); ca["outcome"] = ca.metric + " | " + ca.group
    caw = ca.pivot(index="dataset_unit", columns="outcome", values="performance").reset_index().merge(cchars, on="dataset_unit")
    catab = corr_table(caw, [c for c in caw.columns if " | " in c], CLASS_DESC, "classification_absolute_performance")

    reg_cfg = reg_tables["regression_configuration_summary"]; reg_abs = []
    for metric, maximize in [("rmse", False), ("mae", False), ("r2", True)]:
        for label, filt in [
            ("ML", reg_cfg.family == "ML"),
            ("TFM In-Context", (reg_cfg.family == "TFM") & (reg_cfg.tuning_strategy == "inference")),
            ("TFM fine-tuning", (reg_cfg.family == "TFM") & (reg_cfg.tuning_strategy != "inference")),
        ]:
            b = select_best(reg_cfg[filt], metric, maximize, ["dataset", "train_fraction"])
            b = b[np.isclose(b.train_fraction, 1.0)][["dataset", metric]]; b["outcome"] = metric + " | " + label; b = b.rename(columns={metric: "performance"}); reg_abs.append(b)
    ra = pd.concat(reg_abs); raw = ra.pivot(index="dataset", columns="outcome", values="performance").reset_index().rename(columns={"dataset": "dataset_unit"}).merge(rchars, on="dataset_unit")
    ratab = corr_table(raw, [c for c in raw.columns if " | " in c], REG_DESC, "regression_absolute_performance")

    return {
        "classification_advantage_correlations": ctab,
        "regression_advantage_correlations": rtab,
        "classification_absolute_correlations": catab,
        "regression_absolute_correlations": ratab,
        "classification_correlation_merged_data": cmerge,
        "regression_correlation_merged_data": rmerge,
    }

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

ML_COLOR = "#f28e2b"
TFM_SINGLE_BLUE = "#3182bd"
TFM_BLUE_SHADES = {
    "TFM - In-Context": "#9ecae1",
    "TFM - base-ft": "#6baed6",
    "TFM - PEFT": "#3182bd",
    "TFM - meta-learning": "#08519c",
    "TFM - SFT": "#2171b5",
    "TFM - fine-tuning": "#3182bd",
    "TFM In-Context": "#9ecae1",
    "TFM fine-tuning": "#3182bd",
    "TFM": "#3182bd",
    "ML": ML_COLOR,
    "ML models": ML_COLOR,
}

def color_for_label(label):
    return TFM_BLUE_SHADES.get(label, None)


def plot_strategy_curves(curve, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = ["ML models", "TFM - In-Context", "TFM - base-ft", "TFM - PEFT", "TFM - meta-learning", "TFM - SFT"]
    markers = ["o", "s", "^", "D", "P", "X"]
    for ax, col, title, ylabel in [
        (axes[0], "balanced_accuracy_mean", "A. Balanced accuracy", "Mean balanced accuracy"),
        (axes[1], "auprc_mean", "B. Macro-AUPRC", "Mean macro-AUPRC"),
    ]:
        for s, mk in zip(order, markers):
            g = curve[curve.strategy_group == s].sort_values("train_fraction")
            if len(g):
                line_color = ML_COLOR if s == "ML models" else TFM_SINGLE_BLUE
                ax.plot(g.train_fraction, g[col], marker=mk, label=s, color=line_color, linewidth=2)
        ax.set_title(title); ax.set_xlabel("Training fraction"); ax.set_ylabel(ylabel); ax.set_xticks(FRACTIONS, FRAC_LABELS); ax.grid(alpha=.25)
    axes[1].legend(fontsize=7, frameon=False)
    savefig(fig, fig_dir / "Figure_2_classification_performance.png")


def plot_winner_counts(winner_fraction, fig_dir):
    p = winner_fraction.pivot(index="train_fraction", columns="family", values="n_selected").fillna(0).reindex(FRACTIONS)
    x = np.arange(len(FRACTIONS)); width=.36
    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.bar(x-width/2, p.get("TFM", pd.Series(0,index=p.index)), width, label="TFM")
    ax.bar(x+width/2, p.get("ML", pd.Series(0,index=p.index)), width, label="ML")
    ax.set_xticks(x, FRAC_LABELS); ax.set_xlabel("Training fraction"); ax.set_ylabel("Number of datasets"); ax.set_title("Best-performing family by balanced accuracy"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
    savefig(fig, fig_dir / "Figure_3_classification_winner_counts.png")


def plot_dataset_adv(table, fractions, ylabel, title, path):
    fig, axes = plt.subplots(len(fractions), 1, figsize=(12, 4.0*len(fractions)))
    if len(fractions)==1: axes=[axes]
    for ax, f in zip(axes, fractions):
        d=table[np.isclose(table.train_fraction,f)]
        datasets=list(dict.fromkeys(d.dataset_display.tolist())); x=np.arange(len(datasets))
        for strategy,marker in [("TFM In-Context","o"),("TFM fine-tuning","s")]:
            g=d[d.strategy==strategy].set_index("dataset_display").reindex(datasets)
            ax.plot(x,g.advantage,marker=marker,label=strategy)
        ax.axhline(0,linestyle="--",linewidth=1); ax.set_xticks(x,datasets,rotation=45,ha="right",fontsize=8); ax.set_ylabel(ylabel); ax.set_title(f"{title} — {int(f*100)}% training"); ax.grid(axis="y",alpha=.25); ax.legend(frameon=False)
    savefig(fig,path)


def plot_rank_panels(mean_ranks, fig_dir):
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for ax,col,title in [(axes[0],"mean_rmse_rank","A. RMSE rank"),(axes[1],"mean_r2_rank","B. R² rank")]:
        for s,mk in [("ML models","o"),("TFM - In-Context","s"),("TFM - fine-tuning","D")]:
            g=mean_ranks[mean_ranks.strategy_group==s].sort_values("train_fraction")
            ax.plot(g.train_fraction,g[col],marker=mk,label=s)
        ax.set_title(title); ax.set_xlabel("Training fraction"); ax.set_ylabel("Mean rank (1 = best)"); ax.set_xticks(FRACTIONS,FRAC_LABELS); ax.invert_yaxis(); ax.grid(alpha=.25)
    axes[1].legend(frameon=False,fontsize=8); savefig(fig,fig_dir/"Figure_6_regression_ranks.png")


def plot_regression_oracle(diff, fig_dir):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ds,g in diff.groupby("dataset_display",sort=False):
        g=g.sort_values("train_fraction"); axes[0].plot(g.train_fraction,g.relative_rmse_reduction_pct,marker="o",label=ds); axes[1].plot(g.train_fraction,g.r2_advantage_tfm_minus_ml,marker="o",label=ds)
    axes[0].set_title("A. Relative RMSE reduction"); axes[0].set_ylabel("Relative RMSE reduction vs best ML (%)")
    axes[1].set_title("B. R² advantage"); axes[1].set_ylabel("R² difference (TFM - ML)")
    for ax in axes: ax.axhline(0,linewidth=1); ax.set_xlabel("Training fraction"); ax.set_xticks(FRACTIONS,FRAC_LABELS); ax.grid(alpha=.25)
    axes[0].legend(frameon=False,fontsize=7); savefig(fig,fig_dir/"Figure_7_regression_oracle.png")


def grouped_boxplot_two_panels(tab_a, col_a, y_a, tab_b, col_b, y_b, title, path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8)); pos=np.arange(1,6); off=.18; width=.30
    for ax,tab,col,ylabel,panel in [(axes[0],tab_a,col_a,y_a,"A"),(axes[1],tab_b,col_b,y_b,"B")]:
        handles=[]
        for k,strategy in enumerate(["TFM In-Context","TFM fine-tuning"]):
            data=[tab[np.isclose(tab.train_fraction,f)&(tab.strategy==strategy)][col].dropna().to_numpy() for f in FRACTIONS]
            bp=ax.boxplot(data,positions=pos+(-off if k==0 else off),widths=width,patch_artist=True,manage_ticks=False)
            fc = color_for_label(strategy)
            for patch in bp["boxes"]:
                patch.set_facecolor(fc); patch.set_alpha(0.85)
            for med in bp["medians"]:
                med.set_color("#7f7f7f")
            handles.append(bp["boxes"][0])
        ax.axhline(0,linewidth=1); ax.set_xticks(pos,FRAC_LABELS); ax.set_xlabel("Training fraction"); ax.set_ylabel(ylabel); ax.set_title(panel); ax.grid(axis="y",alpha=.25)
        ax.legend(handles,["TFM In-Context","TFM fine-tuning"],frameon=False,fontsize=8)
    fig.suptitle(title); savefig(fig,path)


def plot_heatmap(corr, outcomes, descriptors, title, path):
    pivot=corr[corr.outcome.isin(outcomes)&corr.descriptor.isin(descriptors)].pivot(index="descriptor",columns="outcome",values="spearman_rho").reindex(descriptors)[outcomes]
    ptab=corr[corr.outcome.isin(outcomes)&corr.descriptor.isin(descriptors)].pivot(index="descriptor",columns="outcome",values="p_holm").reindex(descriptors)[outcomes]
    arr=pivot.to_numpy(float); fig,ax=plt.subplots(figsize=(max(6,2.6*len(outcomes)),.42*len(descriptors)+2.2)); im=ax.imshow(arr,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_xticks(np.arange(len(outcomes)),outcomes,rotation=25,ha="right"); ax.set_yticks(np.arange(len(descriptors)),[DESC_LABEL.get(d,d) for d in descriptors]); ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val=arr[i,j]; star="*" if np.isfinite(ptab.iloc[i,j]) and ptab.iloc[i,j]<.05 else ""
            ax.text(j,i,"NA" if not np.isfinite(val) else f"{val:.2f}{star}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="Spearman rho"); savefig(fig,path)


def plot_absolute_heatmap(corr, metrics, groups, descriptors, title, path):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig,axes=plt.subplots(1,len(metrics),figsize=(5*len(metrics)+0.6,.4*len(descriptors)+2.5),sharey=True)
    if len(metrics)==1: axes=[axes]
    for ax,metric in zip(axes,metrics):
        outcomes=[metric+" | "+g for g in groups]
        p=corr[corr.outcome.isin(outcomes)].pivot(index="descriptor",columns="outcome",values="spearman_rho").reindex(descriptors).reindex(columns=outcomes)
        arr=p.to_numpy(float); im=ax.imshow(arr,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm")
        ax.set_xticks(range(len(groups)),groups,rotation=30,ha="right")
        ax.set_title(metric.replace("_"," ").upper())
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v=arr[i,j]
                ax.text(j,i,"NA" if not np.isfinite(v) else f"{v:.2f}",ha="center",va="center",fontsize=7)
    axes[0].set_yticks(range(len(descriptors)),[DESC_LABEL.get(d,d) for d in descriptors])
    fig.suptitle(title)
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="4%", pad=0.15)
    fig.colorbar(im,cax=cax,label="Spearman rho")
    savefig(fig,path)


def generate_figures(ct, rt, corr, fig_dir):
    plot_strategy_curves(ct["classification_strategy_curve"], fig_dir)
    plot_winner_counts(ct["classification_oracle_winners_by_fraction"], fig_dir)
    plot_dataset_adv(ct["classification_strategy_advantage_ba"],[.10,1.0],"Balanced-accuracy advantage","TFM vs best ML",fig_dir/"Figure_4_classification_BA_advantage_10_100.png")
    plot_rank_panels(rt["regression_strategy_mean_ranks"],fig_dir)
    plot_regression_oracle(rt["regression_oracle_differences"],fig_dir)
    grouped_boxplot_two_panels(ct["classification_strategy_advantage_ba"],"advantage","Balanced-accuracy advantage vs best ML",
                               ct["classification_strategy_advantage_auprc"],"advantage","Macro-AUPRC advantage vs best ML",
                               "Dataset-level TFM advantage in classification",fig_dir/"Figure_8_classification_oracle_boxplots.png")
    grouped_boxplot_two_panels(rt["regression_strategy_advantage"],"r2_advantage","R² advantage vs best ML",
                               rt["regression_strategy_advantage"],"relative_rmse_reduction_pct","Relative RMSE reduction vs best ML (%)",
                               "Dataset-level TFM advantage in regression",fig_dir/"Figure_9_regression_oracle_boxplots.png")
    plot_heatmap(corr["classification_advantage_correlations"],["BA advantage","AUROC advantage","AUPRC advantage"],CLASS_DESC,"Dataset characteristics and classification family advantage (100%)",fig_dir/"Figure_10_classification_advantage_correlations.png")
    plot_heatmap(corr["regression_advantage_correlations"],["relative_rmse_reduction_pct","relative_mae_reduction_pct","r2_advantage_tfm_minus_ml"],REG_DESC,"Dataset characteristics and regression family advantage (100%)",fig_dir/"Figure_11_regression_advantage_correlations.png")

    # Supplementary S1-S8
    curve=ct["classification_strategy_curve"]
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for s,mk in [("ML models","o"),("TFM - In-Context","s"),("TFM - base-ft","^"),("TFM - PEFT","D"),("TFM - meta-learning","P"),("TFM - SFT","X")]:
        g=curve[curve.strategy_group==s].sort_values("train_fraction")
        if len(g): ax.plot(g.train_fraction,g.auroc_mean,marker=mk,label=s,color=color_for_label(s), linewidth=2)
    ax.set_title("Mean AUROC"); ax.set_xlabel("Training fraction"); ax.set_ylabel("Mean AUROC"); ax.set_xticks(FRACTIONS,FRAC_LABELS); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=7); savefig(fig,fig_dir/"Figure_S1_classification_AUROC.png")

    plot_dataset_adv(ct["classification_strategy_advantage_ba"],[.20,.75],"Balanced-accuracy advantage","TFM vs best ML",fig_dir/"Figure_S2_classification_BA_advantage_20_75.png")

    mr=rt["regression_strategy_mean_ranks"]; fig,ax=plt.subplots(figsize=(7.5,4.5))
    for s,mk in [("ML models","o"),("TFM - In-Context","s"),("TFM - fine-tuning","D")]:
        g=mr[mr.strategy_group==s].sort_values("train_fraction"); ax.plot(g.train_fraction,g.mean_mae_rank,marker=mk,label=s)
    ax.set_title("Mean within-dataset MAE rank"); ax.set_xlabel("Training fraction"); ax.set_ylabel("Mean rank (1 = best)"); ax.set_xticks(FRACTIONS,FRAC_LABELS); ax.invert_yaxis(); ax.grid(alpha=.25); ax.legend(frameon=False); savefig(fig,fig_dir/"Figure_S3_regression_MAE_rank.png")

    d=rt["regression_oracle_differences"]; fig,ax=plt.subplots(figsize=(9,5))
    for ds,g in d.groupby("dataset_display",sort=False): g=g.sort_values("train_fraction"); ax.plot(g.train_fraction,g.relative_mae_reduction_pct,marker="o",label=ds)
    ax.axhline(0,linewidth=1); ax.set_title("Relative MAE reduction by regression dataset"); ax.set_xlabel("Training fraction"); ax.set_ylabel("Relative MAE reduction vs best ML (%)"); ax.set_xticks(FRACTIONS,FRAC_LABELS); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8); savefig(fig,fig_dir/"Figure_S4_regression_MAE_by_dataset.png")

    # S5 AUROC boxplot
    tab=ct["classification_strategy_advantage_auroc"]; fig,ax=plt.subplots(figsize=(8,4.8)); pos=np.arange(1,6); off=.18; width=.30; handles=[]
    for k,strategy in enumerate(["TFM In-Context","TFM fine-tuning"]):
        data=[tab[np.isclose(tab.train_fraction,f)&(tab.strategy==strategy)].advantage.dropna().to_numpy() for f in FRACTIONS]
        bp=ax.boxplot(data,positions=pos+(-off if k==0 else off),widths=width,patch_artist=True,manage_ticks=False); handles.append(bp["boxes"][0])
    ax.legend(handles,["TFM In-Context","TFM fine-tuning"],frameon=False); ax.axhline(0,linewidth=1); ax.set_xticks(pos,FRAC_LABELS); ax.set_xlabel("Training fraction"); ax.set_ylabel("AUROC advantage vs best ML"); ax.set_title("TFM advantage across training fractions (AUROC)"); ax.grid(axis="y",alpha=.25); savefig(fig,fig_dir/"Figure_S5_classification_AUROC_boxplot.png")

    tab=rt["regression_strategy_advantage"]; fig,ax=plt.subplots(figsize=(8,4.8)); handles=[]
    for k,strategy in enumerate(["TFM In-Context","TFM fine-tuning"]):
        data=[tab[np.isclose(tab.train_fraction,f)&(tab.strategy==strategy)].relative_mae_reduction_pct.dropna().to_numpy() for f in FRACTIONS]
        bp=ax.boxplot(data,positions=pos+(-off if k==0 else off),widths=width,patch_artist=True,manage_ticks=False); handles.append(bp["boxes"][0])
    ax.legend(handles,["TFM In-Context","TFM fine-tuning"],frameon=False); ax.axhline(0,linewidth=1); ax.set_xticks(pos,FRAC_LABELS); ax.set_xlabel("Training fraction"); ax.set_ylabel("Relative MAE reduction vs best ML (%)"); ax.set_title("Dataset-level TFM advantage in regression (MAE)"); ax.grid(axis="y",alpha=.25); savefig(fig,fig_dir/"Figure_S6_regression_MAE_boxplot.png")

    plot_absolute_heatmap(corr["classification_absolute_correlations"],["balanced_accuracy","auroc_ovr_macro","auprc_macro"],["ML","TFM In-Context","TFM fine-tuning"],CLASS_DESC,"Absolute classification performance correlations at full data",fig_dir/"Figure_S7_classification_absolute_correlations.png")
    plot_absolute_heatmap(corr["regression_absolute_correlations"],["r2","rmse","mae"],["ML","TFM In-Context","TFM fine-tuning"],REG_DESC,"Absolute regression performance correlations at full data",fig_dir/"Figure_S8_regression_absolute_correlations.png")

# -----------------------------------------------------------------------------
# Report / manuscript update notes
# -----------------------------------------------------------------------------
def compare_old_new_best(old_ct, new_ct):
    a=old_ct["classification_oracle_best_overall_ba"][["dataset","train_fraction","model_name","family","balanced_accuracy"]].rename(columns={
        "model_name":"old_model","family":"old_family","balanced_accuracy":"old_BA"})
    b=new_ct["classification_oracle_best_overall_ba"][["dataset","train_fraction","model_name","family","balanced_accuracy"]].rename(columns={
        "model_name":"new_model","family":"new_family","balanced_accuracy":"new_BA"})
    z=a.merge(b,on=["dataset","train_fraction"]); z["delta_BA_new_minus_old"]=z.new_BA-z.old_BA; z["model_changed"]=z.old_model.ne(z.new_model); z["family_changed"]=z.old_family.ne(z.new_family)
    return z


def val_at(curve, strategy, f, col):
    x=curve[(curve.strategy_group==strategy)&np.isclose(curve.train_fraction,f)]
    return float(x.iloc[0][col]) if len(x) else np.nan


def build_report(old, fixed, final, reg, old_ct, ct, rt, corr, audit_ds, best_compare, model_omni_class, model_pairs_class, model_omni_reg, model_pairs_reg):
    lines=[]
    lines.append("GOO-GOO FINAL REANALYSIS — 2026-09-06")
    lines.append("="*78)
    lines.append(f"Final data: {final.dataset.nunique()} classification tasks + {reg.dataset.nunique()} regression tasks")
    lines.append("Classification final merge: corrected binary rerun + original multiclass tasks; student_depression excluded.")
    lines.append("Regression: health_insurance excluded.\n")

    # Audit
    common_n=int(audit_ds.n_common.sum()); weighted_same=float(np.average(audit_ds.pct_BA_exactly_same,weights=audit_ds.n_common))
    lines.append("IMPORTANT OLD-vs-FIXED BINARY AUDIT")
    lines.append(f"- Common fold/configuration rows: {common_n}")
    lines.append(f"- Balanced accuracy exactly unchanged in {weighted_same:.1f}% of common rows; therefore the rerun changed predictions/results, not only AUPRC.")
    lines.append(f"- Best-BA model identity changed in {int(best_compare.model_changed.sum())}/70 dataset-fraction cells; family changed in {int(best_compare.family_changed.sum())}/70.")
    lines.append("- In many changed datasets the fixed run has one fewer preprocessed feature than the old run; audit table exported.\n")

    curve=ct["classification_strategy_curve"]; old_curve=old_ct["classification_strategy_curve"]
    lines.append("CLASSIFICATION STRATEGY MEANS")
    for s in ["TFM - PEFT","ML models","TFM - In-Context"]:
        old10=val_at(old_curve,s,.1,"balanced_accuracy_mean"); old100=val_at(old_curve,s,1,"balanced_accuracy_mean")
        new10=val_at(curve,s,.1,"balanced_accuracy_mean"); new100=val_at(curve,s,1,"balanced_accuracy_mean")
        lines.append(f"- {s}: BA old {old10:.3f}->{old100:.3f}; final {new10:.3f}->{new100:.3f}")

    oldbest=old_ct["classification_oracle_best_overall_ba"]; newbest=ct["classification_oracle_best_overall_ba"]
    ov=oldbest.family.value_counts(); nv=newbest.family.value_counts()
    lines.append(f"\nClassification BA oracle: OLD TFM {ov.get('TFM',0)}/70 vs ML {ov.get('ML',0)}/70; FINAL TFM {nv.get('TFM',0)}/70 vs ML {nv.get('ML',0)}/70")
    wf=ct["classification_oracle_winners_by_fraction"]
    for f in FRACTIONS:
        g=wf[np.isclose(wf.train_fraction,f)].set_index("family").n_selected
        lines.append(f"- {int(f*100)}%: TFM {int(g.get('TFM',0))}, ML {int(g.get('ML',0))}")
    st=ct["classification_oracle_statistics"]
    lines.append("\nFinal classification family tests (GH cohort combined; Wilcoxon two-sided; Holm across fractions):")
    for metric in ["balanced_accuracy","auroc_ovr_macro","auprc_macro"]:
        lines.append(metric)
        for _,x in st[st.metric==metric].iterrows():
            lines.append(f"  {int(x.train_fraction*100)}%: mean diff={x.mean_difference:+.4f}; p={x.p_raw:.5f}; Holm={x.p_holm:.5f}; TFM/ML/tie={int(x.n_tfm_better)}/{int(x.n_ml_better)}/{int(x.n_ties)}")

    lines.append("\nClassification correlations with Holm p<0.05:")
    sig=corr["classification_advantage_correlations"]
    sig=sig[sig.p_holm<.05]
    lines.append("- None" if len(sig)==0 else "\n".join(f"- {r.outcome} vs {r.descriptor}: rho={r.spearman_rho:.2f}, Holm={r.p_holm:.4g}" for _,r in sig.iterrows()))

    lines.append("\nModel-level classification Friedman tests: all fraction/metric omnibus tests significant = " + str(bool((model_omni_class.friedman_p<.05).all())))
    n_pair=int(model_pairs_class.significant_holm_0_05.sum()) if len(model_pairs_class) else 0
    lines.append(f"Holm-significant model-level pairwise tests: {n_pair}")

    lines.append("\nREGRESSION (unchanged relative to manuscript values)")
    for _,x in rt["regression_oracle_summary"].iterrows():
        lines.append(f"- {int(x.train_fraction*100)}%: median RMSE reduction={x.median_relative_rmse_reduction_pct:.1f}%; median MAE reduction={x.median_relative_mae_reduction_pct:.1f}%; mean R2 advantage={x.mean_r2_advantage:.3f}")
    wins=rt["regression_oracle_best_overall_rmse"].family.value_counts(); lines.append(f"- RMSE oracle: TFM {wins.get('TFM',0)}/25; ML {wins.get('ML',0)}/25")
    rs=rt["regression_oracle_statistics"]; rr=rs[rs.metric=="relative_rmse_reduction_pct"]
    lines.append("- RMSE Wilcoxon p_raw / Holm: " + "; ".join(f"{int(x.train_fraction*100)}% {x.p_raw:.4f}/{x.p_holm:.4f}" for _,x in rr.iterrows()))
    rsig=corr["regression_advantage_correlations"]; rsig=rsig[rsig.p_holm<.05]; lines.append("- Regression descriptor correlations after Holm: none" if len(rsig)==0 else f"- Regression significant correlations: {len(rsig)}")
    lines.append("\nConfidence analysis is NOT recomputed by this script because fold-summary CSVs do not contain per-sample prediction probabilities. If the confidence analysis was generated from the old pipeline, rerun it using the final preprocessing/results pipeline before submission.")
    return "\n".join(lines)


def build_manuscript_notes(ct, rt, corr):
    curve=ct["classification_strategy_curve"]
    peft10=val_at(curve,"TFM - PEFT",.1,"balanced_accuracy_mean"); peft100=val_at(curve,"TFM - PEFT",1,"balanced_accuracy_mean")
    ml10=val_at(curve,"ML models",.1,"balanced_accuracy_mean"); ml100=val_at(curve,"ML models",1,"balanced_accuracy_mean")
    icl10=val_at(curve,"TFM - In-Context",.1,"balanced_accuracy_mean"); icl100=val_at(curve,"TFM - In-Context",1,"balanced_accuracy_mean")
    best=ct["classification_oracle_best_overall_ba"]; vc=best.family.value_counts(); tfm=int(vc.get("TFM",0)); ml=int(vc.get("ML",0))
    setting=ct["classification_oracle_tfm_setting_counts"].set_index("setting").n_selected
    icl_w=int(setting.get("inference",0)); peft_w=int(setting.get("fine-tune (PEFT)",0))
    st=ct["classification_oracle_statistics"]
    ba=st[st.metric=="balanced_accuracy"].sort_values("train_fraction")
    au=st[st.metric=="auroc_ovr_macro"].sort_values("train_fraction")
    ap=st[st.metric=="auprc_macro"].sort_values("train_fraction")
    wf=ct["classification_oracle_winners_by_fraction"]
    counts=[]
    for f in FRACTIONS:
        g=wf[np.isclose(wf.train_fraction,f)].set_index("family").n_selected
        counts.append((int(g.get("TFM",0)),int(g.get("ML",0))))
    corr_class=corr["classification_advantage_correlations"]
    imb=corr_class[(corr_class.outcome=="AUPRC advantage")&(corr_class.descriptor=="imbalance_ratio_max_min")].iloc[0]

    txt=f"""# Manuscript updates after final binary rerun / corrected macro-AUPRC\n\n## 1. Methods — AUPRC definition\nReplace the current binary/multiclass wording with:\n\n> For both binary and multiclass classification tasks, AUROC and AUPRC were calculated in a one-vs-rest framework. Each class was treated in turn as the positive class against all remaining classes, and the class-specific scores were macro-averaged, assigning equal weight to each class regardless of class frequency.\n\n## 2. Abstract — classification results\nSuggested replacement:\n\n> Parameter-efficient fine-tuning (PEFT) showed the strongest overall mean balanced-accuracy profile among the evaluated adaptation strategies, increasing from {peft10:.3f} at 10% training data to {peft100:.3f} at full data. In the balanced-accuracy oracle analysis, a TFM was selected in {tfm} of 70 dataset-fraction combinations and an ML baseline in {ml}. For regression, a TFM was selected in all 25 RMSE comparisons; the median relative RMSE reduction versus the best ML baseline ranged from {rt['regression_oracle_summary'].iloc[0].median_relative_rmse_reduction_pct:.1f}% to {rt['regression_oracle_summary'].iloc[-1].median_relative_rmse_reduction_pct:.1f}%, while the mean R² advantage decreased from {rt['regression_oracle_summary'].iloc[0].mean_r2_advantage:.3f} to {rt['regression_oracle_summary'].iloc[-1].mean_r2_advantage:.3f}.\n\n## 3. Results — strategy means\nFinal values:\n- PEFT BA: {peft10:.3f} -> {peft100:.3f}\n- ML BA: {ml10:.3f} -> {ml100:.3f}\n- TFM In-Context BA: {icl10:.3f} -> {icl100:.3f}\n\nNote: PEFT remains strongest overall across fractions, but meta-learning/SFT are slightly higher than PEFT at 20%, so avoid saying PEFT is the top strategy at every fraction.\n\n## 4. Results — BA oracle\nFinal family wins: TFM {tfm}/70 ({100*tfm/70:.1f}%), ML {ml}/70 ({100*ml/70:.1f}%).\nBy fraction: 10% {counts[0][0]} vs {counts[0][1]}; 20% {counts[1][0]} vs {counts[1][1]}; 50% {counts[2][0]} vs {counts[2][1]}; 75% {counts[3][0]} vs {counts[3][1]}; 100% {counts[4][0]} vs {counts[4][1]} (TFM vs ML).\nAmong TFM wins, In-Context = {icl_w}/{tfm} ({100*icl_w/tfm:.1f}%), PEFT = {peft_w}/{tfm} ({100*peft_w/tfm:.1f}%).\n\nCohort-adjusted mean BA differences (TFM - ML): {', '.join(f'{x.mean_difference:+.3f}' for _,x in ba.iterrows())} at 10%, 20%, 50%, 75%, and 100%, respectively. None is significant after Holm correction.\n\n## 5. AUROC/AUPRC inferential results\nThe old statements about significant AUROC at 10% and significant AUPRC at 20%, 50%, and 100% must be removed.\n\nFinal Holm-adjusted AUROC p-values: {', '.join(f'{x.p_holm:.3f}' for _,x in au.iterrows())}.\nFinal Holm-adjusted macro-AUPRC p-values: {', '.join(f'{x.p_holm:.3f}' for _,x in ap.iterrows())}.\nNo AUROC or AUPRC family comparison remains significant after Holm correction.\n\n## 6. Dataset-characteristic correlations\nThe previous AUPRC-imbalance result (rho ~0.86, Holm p ~0.005) does not survive the corrected analysis.\nFinal AUPRC advantage vs imbalance ratio: rho = {imb.spearman_rho:.2f}, Holm p = {imb.p_holm:.3f}.\nNo classification family-advantage correlation remains significant after Holm correction.\n\nTherefore remove the sentence saying that class imbalance was the only significant descriptor, and remove every reference to a 'planned AUPRC audit': the audit is now complete.\n\n## 7. Regression\nRegression results are unchanged: all 25/25 RMSE oracle comparisons favor a TFM; median relative RMSE reduction = 12.6% -> 16.1%; mean R² advantage = 0.099 -> 0.061; exact Wilcoxon p = 0.0625 and Holm p = 0.3125 at every fraction.\n\n## 8. Important consistency check\nThe corrected binary rerun changed balanced accuracy and AUROC for many matched fold/configuration rows, not only AUPRC. Therefore Table 3, Table 4, Figures 2-4, 8, 10, and the corresponding Results/Discussion text must use the final merged dataset generated by this script. Do not patch only the AUPRC column onto the old Table 4.\n\nBecause the prediction-confidence analysis uses prediction-level probabilities, verify that it was generated with the same final preprocessing pipeline. If it came from the old run, rerun it before submission.\n"""
    return txt

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--old-class", default="/mnt/data/all_datasets_fold_results_class(20260906-065606).csv")
    p.add_argument("--fixed-binary", default="/mnt/data/all_datasets_fold_results_fixed_auprc.csv")
    p.add_argument("--reg", default="/mnt/data/all_datasets_fold_results_reg(9).csv")
    p.add_argument("--class-char", default="/mnt/data/dataset_characteristics_class.csv")
    p.add_argument("--reg-char", default="/mnt/data/dataset_characteristics_reg.csv")
    p.add_argument("--out-dir", default="/mnt/data/GOOGOO_final_analysis_20260906")
    return p.parse_args()


def main():
    args=parse_args(); out_dir=Path(args.out_dir); table_dir=out_dir/"tables"; fig_dir=out_dir/"figures"; table_dir.mkdir(parents=True,exist_ok=True); fig_dir.mkdir(parents=True,exist_ok=True)
    old,fixed,final,reg=load_and_merge(args,out_dir)

    audit_rows,audit_ds=audit_old_vs_fixed(old,fixed); audit_rows.to_csv(out_dir/"old_vs_fixed_binary_audit_rows.csv",index=False); audit_ds.to_csv(out_dir/"old_vs_fixed_binary_audit_summary.csv",index=False)

    old_ct=analyze_classification(old); ct=analyze_classification(final); rt=analyze_regression(reg)
    corr=analyze_correlations(ct,rt,args.class_char,args.reg_char)

    omni_c,pairs_c=model_level_tests(ct["classification_configuration_summary"],"classification")
    omni_r,pairs_r=model_level_tests(rt["regression_configuration_summary"],"regression")
    ct["classification_model_level_friedman"]=omni_c; ct["classification_model_level_pairwise_wilcoxon_holm"]=pairs_c
    rt["regression_model_level_friedman"]=omni_r; rt["regression_model_level_pairwise_wilcoxon_holm"]=pairs_r

    best_compare=compare_old_new_best(old_ct,ct); best_compare.to_csv(out_dir/"old_vs_final_best_BA_comparison.csv",index=False)

    # AUPRC diagnostic summary from the corrected binary run
    diag_cols=[c for c in ["auprc_macro","auprc_class_0","auprc_class_1","auprc_legacy"] if c in fixed.columns]
    diag=fixed.groupby("dataset",as_index=False)[diag_cols].mean(numeric_only=True)
    if {"auprc_macro","auprc_legacy"}.issubset(diag.columns): diag["mean_macro_minus_legacy"]=diag.auprc_macro-diag.auprc_legacy
    diag.to_csv(out_dir/"fixed_binary_AUPRC_diagnostic_means.csv",index=False)

    all_tables={**ct,**rt,**corr}
    for name,df in all_tables.items():
        if isinstance(df,pd.DataFrame): df.to_csv(table_dir/(name+".csv"),index=False)

    generate_figures(ct,rt,corr,fig_dir)
    report=build_report(old,fixed,final,reg,old_ct,ct,rt,corr,audit_ds,best_compare,omni_c,pairs_c,omni_r,pairs_r)
    (out_dir/"analysis_report.txt").write_text(report,encoding="utf-8")
    (out_dir/"manuscript_update_notes.md").write_text(build_manuscript_notes(ct,rt,corr),encoding="utf-8")

    metadata={
        "final_classification_datasets":sorted(final.dataset.unique()),
        "final_regression_datasets":sorted(reg.dataset.unique()),
        "binary_rerun_datasets":sorted(fixed.dataset.unique()),
        "classification_models":sorted(final.model_name.unique()),
        "regression_models":sorted(reg.model_name.unique()),
        "excluded_classification":sorted(EXCLUDED_CLASS),
        "excluded_regression":sorted(EXCLUDED_REG),
    }
    (out_dir/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(report)
    print(f"\nOutputs: {out_dir}")

if __name__=="__main__":
    main()
