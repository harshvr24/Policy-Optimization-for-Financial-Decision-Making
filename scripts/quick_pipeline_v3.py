# scripts/quick_pipeline_v3.py
"""
Quick pipeline v3 — advanced feature engineering (V3) for strong uplift.

Outputs:
 - models/quick_model_v3.joblib
 - models/quick_model_v3_type.txt
 - reports/metrics_quick_v3.json
 - reports/feature_importance_v3.json (if available)

Run:
 python scripts/quick_pipeline_v3.py
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
import joblib
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Optional model libs
try:
    from catboost import CatBoostClassifier, Pool as CatPool
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

OUT_MODEL = Path("models")
OUT_REPORT = Path("reports")
OUT_MODEL.mkdir(parents=True, exist_ok=True)
OUT_REPORT.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------
# Feature engineering helpers (V3)
# --------------------------
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def parse_pct(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str) and "%" in x:
            return safe_numeric(x.strip().rstrip("%"))
        return safe_numeric(x)
    except Exception:
        return np.nan


def encode_emp_length(x):
    # typical LendingClub emp_length strings: "10+ years", "< 1 year", "n/a"
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ("n/a", "N/A", ""):
        return np.nan
    if s.startswith("<"):
        return 0.5
    if "+" in s:
        try:
            return float(s.split("+")[0])
        except Exception:
            return np.nan
    # numeric
    try:
        return float(s.split()[0])
    except Exception:
        return np.nan


def subgrade_to_num(s):
    # A1..G5 -> ordinal
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    try:
        letter = s[0].upper()
        num = int(s[1:])
        return (ord(letter) - ord("A")) * 5 + num
    except Exception:
        return np.nan


def grade_to_num(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(ord(str(s).strip().upper()[0]) - ord("A") + 1)
    except Exception:
        return np.nan


def months_since_date(date_series, ref_series=None):
    # return months between ref and date_series; ref default = now
    d = pd.to_datetime(date_series, errors="coerce")
    if ref_series is None:
        ref = pd.Timestamp.now()
        months = ((ref - d).dt.days / 30.0).clip(lower=0)
    else:
        r = pd.to_datetime(ref_series, errors="coerce")
        months = ((r - d).dt.days / 30.0).clip(lower=0)
    return months


def featurize_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced V3 feature engineering. Returns a dataframe with many new columns."""
    df = df.copy()

    # Basic numeric safe conversions
    num_cols = [
        "loan_amnt", "funded_amnt", "installment", "annual_inc",
        "revol_bal", "total_acc", "open_acc", "tot_cur_bal", "total_rev_hi_lim"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = safe_numeric(df[c]).fillna(0)

    # log transforms (robust)
    for c in ["loan_amnt", "funded_amnt", "installment", "revol_bal", "annual_inc", "tot_cur_bal"]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

    # revol_util convert and cap
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].apply(parse_pct).fillna(0)
        df["revol_util_capped"] = df["revol_util"].clip(upper=100)

    # Debt/income signals
    if ("annual_inc" in df.columns) and ("installment" in df.columns):
        # monthly installment relative to monthly income
        df["inst_to_monthly_income"] = df["installment"] / (df["annual_inc"] / 12.0 + 1e-8)
    if ("loan_amnt" in df.columns) and ("annual_inc" in df.columns):
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-8)
    if ("revol_bal" in df.columns) and ("annual_inc" in df.columns):
        df["revol_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1e-8)

    # credit utilization / limits
    if "total_rev_hi_lim" in df.columns:
        df["rev_bal_over_rev_lim"] = df["revol_bal"] / (df["total_rev_hi_lim"] + 1e-8)

    # fico, buckets
    if "fico_range_low" in df.columns:
        df["fico_range_low"] = safe_numeric(df["fico_range_low"]).fillna(0)
        df["fico_bucket"] = pd.cut(df["fico_range_low"], bins=[0,640,660,680,700,720,740,760,800,1000], labels=False).astype(float)

    # dti features
    if "dti" in df.columns:
        df["dti"] = safe_numeric(df["dti"]).fillna(df["dti"].median() if df["dti"].notna().any() else 0)
        df["dti_sq"] = df["dti"] ** 2
        df["dti_bin"] = pd.cut(df["dti"], bins=[-1,5,10,15,20,25,35,1000], labels=False).astype(float)

    # credit age / earliest cr line
    if "earliest_cr_line" in df.columns:
        # months credit history relative to issue_d if available
        if "issue_d" in df.columns:
            df["months_credit_history"] = months_since_date(df["earliest_cr_line"], df["issue_d"])
        else:
            df["months_credit_history"] = months_since_date(df["earliest_cr_line"])
    else:
        df["months_credit_history"] = np.nan

    # counts of derogatory events / public records
    # columns that may exist: collections_12_mths_ex_med, pub_rec, pub_rec_bankruptcies
    for c in ("collections_12_mths_ex_med", "pub_rec", "pub_rec_bankruptcies"):
        if c in df.columns:
            df[c] = safe_numeric(df[c]).fillna(0)

    # fraction of open accounts that are recent (if open_acc_6m exists)
    if "open_acc" in df.columns and "open_acc_6m" in df.columns:
        df["open_acc_recent_frac"] = df["open_acc_6m"].fillna(0) / (df["open_acc"].replace(0, np.nan) + 1e-8)

    # employment length encoding
    if "emp_length" in df.columns:
        df["emp_length_years"] = df["emp_length"].apply(encode_emp_length).fillna(0)

    # grade / subgrade encoding
    if "grade" in df.columns:
        df["grade_score"] = df["grade"].apply(grade_to_num).fillna(0)
    if "sub_grade" in df.columns:
        df["subgrade_score"] = df["sub_grade"].apply(subgrade_to_num).fillna(0)

    # loan structure signals
    if "term" in df.columns:
        try:
            df["term_m"] = safe_numeric(df["term"].str.extract(r"(\d+)")[0]).fillna(36)
        except Exception:
            df["term_m"] = np.nan

    # interactions with high signal potential
    if "log_annual_inc" in df.columns and "loan_amnt" in df.columns:
        df["income_x_loan"] = df["log_annual_inc"].fillna(0) * df["loan_amnt"].fillna(0)
    if "revol_util_capped" in df.columns and "log_annual_inc" in df.columns:
        df["revol_x_income"] = df["revol_util_capped"].fillna(0) * df["log_annual_inc"].fillna(0)

    # more engineered ratios
    if ("installment" in df.columns) and ("loan_amnt" in df.columns):
        df["inst_over_loan"] = df["installment"] / (df["loan_amnt"] + 1e-8)

    # reduce textual/noise columns
    drop_cols = ["url", "desc", "title", "emp_title", "member_id", "id", "zip_code"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors="ignore")

    # fill numeric NaNs with medians
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols_all:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

    # convert small-cardinality objects to strings for CatBoost safety
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].nunique(dropna=True) < 2000:
            df[c] = df[c].astype(str).fillna("__nan__")

    return df


# --------------------------
# Training wrappers & utils
# --------------------------
def compute_scale_pos_weight(y):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if pos == 0:
        raise RuntimeError("No positive examples in training set")
    return float(neg) / float(pos)


def tune_threshold(probas_val, y_val):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probas_val >= t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return float(best_t), float(best_f1)


def train_catboost(X_train, y_train, X_val, y_val, cat_features_idx):
    params = {
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 200,
        "auto_class_weights": "Balanced"
    }
    train_pool = CatPool(X_train, label=y_train, cat_features=cat_features_idx)
    val_pool = CatPool(X_val, label=y_val, cat_features=cat_features_idx)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=75)
    return model


def train_lgb(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "is_unbalance": True
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    bst = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval], early_stopping_rounds=75, verbose_eval=200)
    return bst


def train_xgb(X_train, y_train, X_val, y_val, scale_pos_weight):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False
    }
    model = xgb.XGBClassifier(**params)
    try:
        model.fit(X_train, y_train, early_stopping_rounds=30, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate_and_save(model_obj, model_type, X_test, y_test, threshold, feat_names=None):
    # get probabilities
    if model_type == "catboost":
        proba = model_obj.predict_proba(X_test)[:, 1]
    elif model_type == "lightgbm":
        proba = model_obj.predict(X_test)
    elif model_type == "xgboost":
        proba = model_obj.predict_proba(X_test)[:, 1]
    else:
        raise RuntimeError("unknown model_type")

    preds = (proba >= threshold).astype(int)
    metrics = {
        "model_type": model_type,
        "threshold": threshold,
        "auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

    # save model + type
    joblib.dump(model_obj, OUT_MODEL / "quick_model_v3.joblib")
    (OUT_MODEL / "quick_model_v3_type.txt").write_text(model_type)

    # feature importance (best-effort)
    try:
        feat_imp = None
        if model_type == "catboost":
            # catboost get_feature_importance returns importance array in same order as input features
            imp = model_obj.get_feature_importance()
            feat_imp = sorted(list(zip(feat_names or [], imp)), key=lambda x: -x[1])[:100]
        elif model_type == "lightgbm":
            imp = model_obj.feature_importance(importance_type="gain")
            feat_imp = sorted(list(zip(feat_names or [], imp)), key=lambda x: -x[1])[:100]
        elif model_type == "xgboost":
            try:
                imp = model_obj.feature_importances_
                feat_imp = sorted(list(zip(feat_names or [], imp)), key=lambda x: -x[1])[:100]
            except Exception:
                try:
                    booster = model_obj.get_booster()
                    score = booster.get_score(importance_type="gain")
                    feat_imp = sorted(list(score.items()), key=lambda x: -x[1])[:100]
                except Exception:
                    feat_imp = None
        if feat_imp is not None:
            with open(OUT_REPORT / "feature_importance_v3.json", "w") as fh:
                json.dump([{"feature": f, "imp": float(i)} for f, i in feat_imp], fh, indent=2)
    except Exception as e:
        log(f"[WARN] Could not extract feature importances: {e}")

    with open(OUT_REPORT / "metrics_quick_v3.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


# --------------------------
# main
# --------------------------
def main():
    t0 = time.time()
    log("Quick pipeline v3 start")

    # check processed files exist
    train_p = Path("data/processed/train_processed.csv.gz")
    val_p = Path("data/processed/val_processed.csv.gz")
    test_p = Path("data/processed/test_processed.csv.gz")
    for p in (train_p, val_p, test_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file: {p}")

    # load & featurize
    log("Loading and featurizing (V3)")
    train = pd.read_csv(train_p, compression="gzip")
    val = pd.read_csv(val_p, compression="gzip")
    test = pd.read_csv(test_p, compression="gzip")

    train_fe = featurize_v3(train)
    val_fe = featurize_v3(val)
    test_fe = featurize_v3(test)

    drop_cols = ["y", "loan_status", "issue_d_parsed"]
    X_train = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns], errors="ignore")
    y_train = train_fe["y"].values
    X_val = val_fe.drop(columns=[c for c in drop_cols if c in val_fe.columns], errors="ignore")
    y_val = val_fe["y"].values
    X_test = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns], errors="ignore")
    y_test = test_fe["y"].values

    log(f"Shapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    log(f"Computed scale_pos_weight={scale_pos_weight:.3f}")

    # CatBoost: detect categorical features conservatively
    cat_indices = []
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    X_test_cb = X_test.copy()
    if HAS_CAT:
        cat_cols = [c for c in X_train_cb.columns if X_train_cb[c].dtype == "object" or (X_train_cb[c].nunique(dropna=True) < 200)]
        # ensure these are strings (CatBoost-safe)
        for c in cat_cols:
            X_train_cb[c] = X_train_cb[c].astype(str).fillna("__nan__")
            X_val_cb[c] = X_val_cb[c].astype(str).fillna("__nan__")
            X_test_cb[c] = X_test_cb[c].astype(str).fillna("__nan__")
        cat_indices = [X_train_cb.columns.get_loc(c) for c in cat_cols]

    model_obj = None
    model_type = None

    # Try CatBoost first
    if HAS_CAT:
        try:
            log("Training CatBoost (v3)...")
            model_obj = train_catboost(X_train_cb, y_train, X_val_cb, y_val, cat_indices)
            model_type = "catboost"
        except Exception as e:
            log(f"[WARN] CatBoost failed: {e}")
            model_obj = None
            model_type = None

    # LightGBM fallback
    if model_obj is None and HAS_LGB:
        try:
            log("Training LightGBM (v3)...")
            model_obj = train_lgb(X_train.values, y_train, X_val.values, y_val)
            model_type = "lightgbm"
        except Exception as e:
            log(f"[WARN] LightGBM failed: {e}")
            model_obj = None
            model_type = None

    # XGBoost fallback
    if model_obj is None and HAS_XGB:
        try:
            log("Training XGBoost (v3 fallback)...")
            model_obj = train_xgb(X_train.values, y_train, X_val.values, y_val, scale_pos_weight)
            model_type = "xgboost"
        except Exception as e:
            log(f"[WARN] XGBoost failed: {e}")
            model_obj = None
            model_type = None

    if model_obj is None:
        raise RuntimeError("No model trained. Install catboost / lightgbm / xgboost")

    # validation probabilities for threshold tuning
    log("Computing validation probabilities for threshold tuning")
    if model_type == "catboost":
        proba_val = model_obj.predict_proba(X_val_cb)[:, 1]
    elif model_type == "lightgbm":
        proba_val = model_obj.predict(X_val.values)
    elif model_type == "xgboost":
        proba_val = model_obj.predict_proba(X_val.values)[:, 1]
    else:
        raise RuntimeError("Unexpected model type")

    best_thr, best_f1 = tune_threshold(proba_val, y_val)
    log(f"Best threshold on val => threshold={best_thr:.3f}, f1={best_f1:.4f}")

    # Evaluate on test
    log("Evaluating on test and saving artifacts")
    metrics = evaluate_and_save(
        model_obj,
        model_type,
        X_test_cb if model_type == "catboost" else X_test.values,
        y_test,
        best_thr,
        feat_names=list(X_train.columns)
    )
    log(json.dumps(metrics, indent=2))
    log(f"Quick pipeline v3 finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
