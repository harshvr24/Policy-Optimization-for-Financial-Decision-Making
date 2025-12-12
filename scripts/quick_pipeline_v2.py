# scripts/quick_pipeline_v2.py
"""
Quick pipeline v2 — improved feature engineering + same single-run behavior.

Outputs:
 - models/quick_model_v2.joblib
 - models/quick_model_v2_type.txt
 - reports/metrics_quick_v2.json

Run:
 python scripts/quick_pipeline_v2.py
"""

from __future__ import annotations
import os, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# optional libs
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
# Feature engineering (v2)
# --------------------------
def featurize_df_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- basic numeric transforms ---
    for c in ("loan_amnt","funded_amnt","installment","revol_bal","total_acc","annual_inc"):
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0))

    # revol_util: normalize strings like "12.3%"
    if "revol_util" in df.columns:
        df["revol_util"] = pd.to_numeric(df["revol_util"].astype(str).str.rstrip("%"), errors="coerce")
        df["revol_util_capped"] = df["revol_util"].fillna(0).clip(upper=100)

    # --- debt / income ratios & variants ---
    if "annual_inc" in df.columns and "installment" in df.columns:
        df["income_to_installment"] = df["annual_inc"].replace(0, np.nan) / (df["installment"].replace(0, np.nan) * 12)
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"].replace(0, np.nan) / df["annual_inc"].replace(0, np.nan)
    if "revol_bal" in df.columns and "annual_inc" in df.columns:
        df["revol_to_income"] = df["revol_bal"].replace(0, np.nan) / df["annual_inc"].replace(0, np.nan)

    # dti bins and polynomial
    if "dti" in df.columns:
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce").fillna(df["dti"].median() if df["dti"].notna().any() else 0)
        df["dti_sq"] = df["dti"] ** 2
        df["dti_bin"] = pd.cut(df["dti"], bins=[-1,5,10,15,20,25,35,1000], labels=False).astype(float)

    # --- fico / credit age ---
    if "fico_range_low" in df.columns:
        df["fico_range_low"] = pd.to_numeric(df["fico_range_low"], errors="coerce")
        df["fico_bucket"] = pd.cut(df["fico_range_low"].fillna(0), bins=[0,640,660,680,700,720,740,760,800,1000], labels=False).astype(float)
    # months since earliest credit (if earliest_cr_line exists) relative to issue_d or today
    if "earliest_cr_line" in df.columns:
        try:
            df["earliest_cr_line_dt"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
            if "issue_d" in df.columns:
                df["issue_d_dt"] = pd.to_datetime(df["issue_d"], errors="coerce")
                df["months_credit_history"] = ((df["issue_d_dt"] - df["earliest_cr_line_dt"]).dt.days / 30).clip(lower=0)
            else:
                df["months_credit_history"] = ((pd.Timestamp.now() - df["earliest_cr_line_dt"]).dt.days / 30).clip(lower=0)
        except Exception:
            df["months_credit_history"] = np.nan

    # ratios of open_acc / total_acc
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_over_total_acc"] = df["open_acc"].replace(0, np.nan) / df["total_acc"].replace(0, np.nan)

    # --- grade / subgrade encodings ---
    if "grade" in df.columns:
        grade_map = {g: i for i, g in enumerate(sorted(df["grade"].dropna().unique()), start=1)}
        df["grade_score"] = df["grade"].map(grade_map).astype(float)
    if "sub_grade" in df.columns:
        # convert A1..G5 to ordinal scale: A1=1,...,G5=35 (if full range present)
        try:
            def sub_to_num(s):
                if pd.isna(s):
                    return np.nan
                s = str(s).strip()
                letter = s[0]
                number = int(s[1:])
                letter_idx = ord(letter.upper()) - ord("A")
                return letter_idx * 5 + number
            df["subgrade_score"] = df["sub_grade"].apply(sub_to_num).astype(float)
        except Exception:
            df["subgrade_score"] = np.nan

    # --- small interactions (high signal) ---
    if "log_annual_inc" in df.columns and "loan_amnt" in df.columns:
        df["income_x_loan"] = df["log_annual_inc"].fillna(0) * df["loan_amnt"].fillna(0)
    if "revol_util_capped" in df.columns and "log_annual_inc" in df.columns:
        df["revol_x_income"] = df["revol_util_capped"].fillna(0) * df["log_annual_inc"].fillna(0)

    # --- term numeric ---
    if "term" in df.columns:
        df["term_m"] = pd.to_numeric(df["term"].str.extract(r"(\d+)")[0], errors="coerce").fillna(36)

    # --- drop obviously noisy textual / id fields to avoid leakage ---
    for drop_col in ("desc","emp_title","title","url","member_id","id","zip_code"):
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True, errors="ignore")

    # fill NA for numeric columns with medians (later models prefer no NA)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

    # convert small-cardinality objects to string for CatBoost safety
    for c in df.select_dtypes(include=["object"]).columns:
        # convert only columns with reasonable cardinality or obvious categorical semantics
        if df[c].nunique(dropna=True) < 1000:
            df[c] = df[c].astype(str).fillna("__nan__")

    return df

# --------------------------
# helpers & training wrappers
# --------------------------
def load_and_featurize(path):
    df = pd.read_csv(path, compression="gzip")
    return featurize_df_v2(df)

def compute_scale_pos_weight(y):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if pos == 0:
        raise RuntimeError("No positive examples in training set")
    return float(neg) / float(pos)

def train_catboost(X_train, y_train, X_val, y_val, cat_features_idx):
    params = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 100,
        "auto_class_weights": "Balanced"
    }
    train_pool = CatPool(X_train, label=y_train, cat_features=cat_features_idx)
    val_pool = CatPool(X_val, label=y_val, cat_features=cat_features_idx)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    return model

def train_lgb(X_train, y_train, X_val, y_val):
    params = {
        "objective":"binary",
        "metric":"auc",
        "learning_rate":0.03,
        "num_leaves":31,
        "feature_fraction":0.8,
        "bagging_fraction":0.8,
        "bagging_freq":5,
        "verbosity":-1,
        "is_unbalance": True
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    bst = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], early_stopping_rounds=50, verbose_eval=100)
    return bst

def train_xgb(X_train, y_train, X_val, y_val, scale_pos_weight):
    params = {
        "objective":"binary:logistic",
        "eval_metric":"auc",
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
        model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def tune_threshold(probas_val, y_val):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probas_val >= t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return float(best_t), float(best_f1)

def evaluate_and_save(model_obj, model_type, X_test, y_test, threshold):
    if model_type == "catboost":
        proba = model_obj.predict_proba(X_test)[:,1]
    elif model_type == "lightgbm":
        proba = model_obj.predict(X_test)
    elif model_type == "xgboost":
        proba = model_obj.predict_proba(X_test)[:,1]
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
    joblib.dump(model_obj, OUT_MODEL / "quick_model_v2.joblib")
    (OUT_MODEL / "quick_model_v2_type.txt").write_text(model_type)
    with open(OUT_REPORT / "metrics_quick_v2.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics

# --------------------------
# main pipeline
# --------------------------
def main():
    t0 = time.time()
    log("Quick pipeline v2 start")
    train_p = Path("data/processed/train_processed.csv.gz")
    val_p   = Path("data/processed/val_processed.csv.gz")
    test_p  = Path("data/processed/test_processed.csv.gz")
    for p in (train_p, val_p, test_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file: {p}")

    log("Loading & featurizing data (v2)")
    train = load_and_featurize(str(train_p))
    val   = load_and_featurize(str(val_p))
    test  = load_and_featurize(str(test_p))

    drop_cols = ["y","loan_status","issue_d_parsed"]
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    y_train = train["y"].values
    X_val = val.drop(columns=[c for c in drop_cols if c in val.columns], errors="ignore")
    y_val = val["y"].values
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")
    y_test = test["y"].values

    X_train_vals = X_train.values
    X_val_vals = X_val.values
    X_test_vals = X_test.values

    log(f"Shapes -> train: {X_train_vals.shape}, val: {X_val_vals.shape}, test: {X_test_vals.shape}")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    log(f"Computed scale_pos_weight={scale_pos_weight:.3f}")

    # cat features detection (conservative)
    cat_indices = []
    if HAS_CAT:
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object" or (X_train[c].nunique(dropna=True) < 200)]
        # ensure catboost sees strings/ints only
        for c in cat_cols:
            X_train[c] = X_train[c].astype(str).fillna("__nan__")
            X_val[c] = X_val[c].astype(str).fillna("__nan__")
            X_test[c] = X_test[c].astype(str).fillna("__nan__")
        cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    model_obj = None
    model_type = None

    # Train CatBoost
    if HAS_CAT:
        try:
            log("Training CatBoost (v2)...")
            model_obj = train_catboost(X_train, y_train, X_val, y_val, cat_indices)
            model_type = "catboost"
        except Exception as e:
            log(f"[WARN] CatBoost failed: {e}")
            model_obj = None
            model_type = None

    # LightGBM
    if model_obj is None and HAS_LGB:
        try:
            log("Training LightGBM (v2)...")
            model_obj = train_lgb(X_train_vals, y_train, X_val_vals, y_val)
            model_type = "lightgbm"
        except Exception as e:
            log(f"[WARN] LightGBM failed: {e}")
            model_obj = None
            model_type = None

    # XGBoost fallback
    if model_obj is None and HAS_XGB:
        try:
            log("Training XGBoost (v2 fallback)...")
            model_obj = train_xgb(X_train_vals, y_train, X_val_vals, y_val, scale_pos_weight)
            model_type = "xgboost"
        except Exception as e:
            log(f"[WARN] XGBoost failed: {e}")
            model_obj = None
            model_type = None

    if model_obj is None:
        raise RuntimeError("No model trained - install catboost/lightgbm/xgboost")

    # validation probabilities for threshold tuning
    log("Computing validation probabilities for threshold tuning (v2)")
    if model_type == "catboost":
        proba_val = model_obj.predict_proba(X_val)[:,1]
    elif model_type == "lightgbm":
        proba_val = model_obj.predict(X_val_vals)
    elif model_type == "xgboost":
        proba_val = model_obj.predict_proba(X_val_vals)[:,1]
    else:
        raise RuntimeError("unknown model type")

    best_thr, best_f1 = tune_threshold(proba_val, y_val)
    log(f"Best threshold on val for F1 => threshold={best_thr:.3f}, f1={best_f1:.4f}")

    metrics = evaluate_and_save(model_obj, model_type, X_test if model_type=="catboost" else X_test_vals, y_test, best_thr)
    log(json.dumps(metrics, indent=2))
    log(f"Quick pipeline v2 finished in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
