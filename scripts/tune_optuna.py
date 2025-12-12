#!/usr/bin/env python3
"""
scripts/tune_optuna.py

Lightweight, robust Optuna tuning for CatBoost / LightGBM / XGBoost.

- Loads processed train/val/test from data/processed/*.csv.gz
- Tunes each requested model for n_trials (default small)
- Saves best params and final fitted model to models/tuned/
- Defensive about package versions and early_stopping compatibility.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# optional imports
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier, Pool as CatPool
    HAS_CAT = True
except Exception:
    CatBoostClassifier = None
    CatPool = None
    HAS_CAT = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False


OUT_DIR = Path("models/tuned")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_processed_data(processed_dir: Path):
    train_p = processed_dir / "train_processed.csv.gz"
    val_p = processed_dir / "val_processed.csv.gz"
    test_p = processed_dir / "test_processed.csv.gz"
    for p in (train_p, val_p, test_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file: {p}")
    train = pd.read_csv(train_p, compression="gzip")
    val = pd.read_csv(val_p, compression="gzip")
    test = pd.read_csv(test_p, compression="gzip")
    return train, val, test


def prepare_Xy(df):
    drop = ["y", "loan_status", "issue_d_parsed"]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    y = df["y"].values
    return X, y


# ---------------------------
# CatBoost objective & final fit
# ---------------------------
def objective_cat(trial, Xtr, ytr, Xv, yv, cat_indices):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "depth": trial.suggest_int("depth", 4, 9),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 10.0),
        "random_seed": 42,
        "verbose": False,
        "auto_class_weights": "Balanced"
    }
    model = CatBoostClassifier(**params)
    try:
        # CatPool is generally safe on most installs
        pool_tr = CatPool(Xtr, label=ytr, cat_features=cat_indices)
        pool_val = CatPool(Xv, label=yv, cat_features=cat_indices)
        model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=50, verbose=False)
    except Exception:
        # fallback: fit without early stopping if compatibility issues occur
        model.fit(pool_tr, eval_set=pool_val, verbose=False)
    p = model.predict_proba(Xv)[:, 1]
    return float(roc_auc_score(yv, p)), model


def tune_catboost(Xtr, ytr, Xv, yv, cat_indices, n_trials=20):
    if not HAS_CAT:
        log("CatBoost not installed; skipping.")
        return None, None
    if not HAS_OPTUNA:
        log("Optuna not installed; performing single default fit for CatBoost.")
        # default fit
        model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, auto_class_weights="Balanced", verbose=100)
        pool_tr = CatPool(Xtr, label=ytr, cat_features=cat_indices)
        pool_val = CatPool(Xv, label=yv, cat_features=cat_indices)
        model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=50, verbose=False)
        p = model.predict_proba(Xv)[:, 1]
        return {"iterations": 1000, "learning_rate": 0.03, "depth": 6}, model
    # use optuna
    study = optuna.create_study(direction="maximize")
    best_model = None
    best_score = 0.0

    def _obj(trial):
        nonlocal best_model, best_score
        try:
            score, model = objective_cat(trial, Xtr, ytr, Xv, yv, cat_indices)
            if score > best_score:
                best_score = score
                best_model = model
            return score
        except Exception as e:
            # return poor score on exception but don't crash
            log(f"[WARN] CatBoost trial error: {e}")
            return 0.0

    study.optimize(_obj, n_trials=n_trials)
    log(f"CatBoost best AUC (val) = {best_score:.4f}")
    # best_model is kept from trials; if not, refit with best params
    if best_model is None:
        best_params = study.best_params
        best_params.update({"iterations": study.best_params.get("iterations", 1000), "auto_class_weights": "Balanced"})
        model = CatBoostClassifier(**best_params)
        pool_tr = CatPool(Xtr, label=ytr, cat_features=cat_indices)
        pool_val = CatPool(Xv, label=yv, cat_features=cat_indices)
        model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=50, verbose=False)
        best_model = model
        best_params = model.get_params()
    else:
        best_params = study.best_params
    return best_params, best_model


# ---------------------------
# LightGBM objective & final fit
# ---------------------------
def objective_lgb(trial, Xtr, ytr, Xv, yv):
    param = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "verbosity": -1,
        "is_unbalance": True
    }
    try:
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xv, label=yv)
        # try with early stopping; if fails, fallback
        bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dval], early_stopping_rounds=50, verbose_eval=False)
    except Exception:
        bst = lgb.train(param, dtrain, num_boost_round=300, valid_sets=[dval], verbose_eval=False)
    p = bst.predict(Xv)
    return float(roc_auc_score(yv, p)), bst


def tune_lgb(Xtr, ytr, Xv, yv, n_trials=20):
    if not HAS_LGB:
        log("LightGBM not installed; skipping.")
        return None, None
    if not HAS_OPTUNA:
        log("Optuna not installed; performing single default fit for LightGBM.")
        params = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 31, "is_unbalance": True}
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xv, label=yv)
        bst = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval], early_stopping_rounds=30, verbose_eval=False)
        return params, bst
    study = optuna.create_study(direction="maximize")
    best_score = 0.0
    best_model = None
    best_params = None

    def _obj(trial):
        nonlocal best_score, best_model, best_params
        try:
            score, model = objective_lgb(trial, Xtr, ytr, Xv, yv)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = trial.params
            return score
        except Exception as e:
            log(f"[WARN] LightGBM trial error: {e}")
            return 0.0

    study.optimize(_obj, n_trials=n_trials)
    log(f"LightGBM best AUC (val) = {best_score:.4f}")
    return best_params or study.best_params, best_model


# ---------------------------
# XGBoost objective & final fit
# ---------------------------
def objective_xgb(trial, Xtr, ytr, Xv, yv):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False,
        "verbosity": 0
    }
    try:
        model = xgb.XGBClassifier(**param)
        # Avoid passing early_stopping_rounds inside optuna objective to remain compatible with various xgboost builds
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        p = model.predict_proba(Xv)[:, 1]
    except Exception:
        # fallback: fit with fewer rounds without eval_set if something errors
        model = xgb.XGBClassifier(**param)
        model.fit(Xtr, ytr, verbose=False)
        p = model.predict_proba(Xv)[:, 1]
    return float(roc_auc_score(yv, p)), model


def tune_xgb(Xtr, ytr, Xv, yv, n_trials=20):
    if not HAS_XGB:
        log("XGBoost not installed; skipping.")
        return None, None
    if not HAS_OPTUNA:
        log("Optuna not installed; performing single default fit for XGBoost.")
        model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, use_label_encoder=False, verbosity=0)
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        return {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6}, model
    study = optuna.create_study(direction="maximize")
    best_score = 0.0
    best_model = None
    best_params = None

    def _obj(trial):
        nonlocal best_score, best_model, best_params
        try:
            score, model = objective_xgb(trial, Xtr, ytr, Xv, yv)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = trial.params
            return score
        except Exception as e:
            log(f"[WARN] XGBoost trial error: {e}")
            return 0.0

    study.optimize(_obj, n_trials=n_trials)
    log(f"XGBoost best AUC (val) = {best_score:.4f}")
    return best_params or study.best_params, best_model


# ---------------------------
# Utility: save results
# ---------------------------
def save_model_and_params(prefix: str, model_obj, params):
    # model_obj may be sklearn estimator, lgb booster, or catboost model
    out_model = OUT_DIR / f"{prefix}_model.joblib"
    try:
        joblib.dump(model_obj, out_model)
    except Exception:
        try:
            # some models (lgb booster) pickle ok via their own save
            if hasattr(model_obj, "save_model"):
                model_obj.save_model(str(OUT_DIR / f"{prefix}_model.txt"))
                out_model = OUT_DIR / f"{prefix}_model.txt"
            else:
                raise
        except Exception as e:
            log(f"[WARN] Could not dump model with joblib: {e}")
    params_out = OUT_DIR / f"{prefix}_best_params.json"
    with open(params_out, "w") as fh:
        json.dump(params if params is not None else {}, fh, indent=2)


# ---------------------------
# CLI / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for CatBoost/LightGBM/XGBoost (robust)")
    parser.add_argument("--models", "-m", choices=["cat", "lgb", "xgb", "all"], default="all",
                        help="Which models to tune")
    parser.add_argument("--n-trials", "-n", type=int, default=20, help="Number of optuna trials per model (default 20)")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed CSVs (train/val/test)")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output folder for tuned models")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_OPTUNA:
        log("WARNING: optuna is not installed. The script will run single default fits instead of tuning.")
    log(f"Loading processed data from {processed_dir}")
    train, val, test = load_processed_data(processed_dir)
    Xtr_df, ytr = prepare_Xy(train)
    Xv_df, yv = prepare_Xy(val)
    Xt_df, yt = prepare_Xy(test)

    # Keep DataFrame copies for CatBoost; arrays for others
    Xtr_vals = Xtr_df.values
    Xv_vals = Xv_df.values
    Xt_vals = Xt_df.values

    cat_indices = []
    if HAS_CAT:
        cat_cols = [c for c in Xtr_df.columns if Xtr_df[c].dtype == "object" or Xtr_df[c].nunique(dropna=True) < 200]
        # make sure cat cols are strings
        for c in cat_cols:
            Xtr_df[c] = Xtr_df[c].astype(str).fillna("__nan__")
            Xv_df[c] = Xv_df[c].astype(str).fillna("__nan__")
            Xt_df[c] = Xt_df[c].astype(str).fillna("__nan__")
        cat_indices = [Xtr_df.columns.get_loc(c) for c in cat_cols]
        log(f"CatBoost categorical columns detected: {len(cat_indices)}")

    models_to_run = [args.models] if args.models != "all" else ["cat", "lgb", "xgb"]
    results = {}

    if "cat" in models_to_run:
        log("=== Tuning CatBoost ===")
        try:
            params, model = tune_catboost(Xtr_df, ytr, Xv_df, yv, cat_indices, n_trials=args.n_trials)
            if model is not None:
                save_model_and_params("catboost", model, params)
                results["catboost"] = {"val_auc": float(roc_auc_score(yv, model.predict_proba(Xv_df)[:,1]))}
        except Exception:
            log("CatBoost tuning failed:")
            traceback.print_exc()

    if "lgb" in models_to_run:
        log("=== Tuning LightGBM ===")
        try:
            params, model = tune_lgb(Xtr_vals, ytr, Xv_vals, yv, n_trials=args.n_trials)
            if model is not None:
                save_model_and_params("lightgbm", model, params)
                # get val preds
                try:
                    p_val = model.predict(Xv_vals)
                except Exception:
                    p_val = model.predict(Xv_vals)
                results["lightgbm"] = {"val_auc": float(roc_auc_score(yv, p_val))}
        except Exception:
            log("LightGBM tuning failed:")
            traceback.print_exc()

    if "xgb" in models_to_run:
        log("=== Tuning XGBoost ===")
        try:
            params, model = tune_xgb(Xtr_vals, ytr, Xv_vals, yv, n_trials=args.n_trials)
            if model is not None:
                save_model_and_params("xgboost", model, params)
                p_val = model.predict_proba(Xv_vals)[:,1]
                results["xgboost"] = {"val_auc": float(roc_auc_score(yv, p_val))}
        except Exception:
            log("XGBoost tuning failed:")
            traceback.print_exc()

    summary_out = out_dir / "tuning_summary.json"
    with open(summary_out, "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"Tuning complete. Summary written to {summary_out}")
    log("Saved models & params to " + str(out_dir))


if __name__ == "__main__":
    main()
