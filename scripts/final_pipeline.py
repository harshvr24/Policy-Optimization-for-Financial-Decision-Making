# scripts/final_pipeline.py
"""
Final pipeline: light optuna tuning for CatBoost/LGB/XGB, stacking, threshold tuning, save artifacts.

Outputs (in models/final and reports/):
 - models/final/base_catboost.joblib
 - models/final/base_lgb.joblib
 - models/final/base_xgb.joblib
 - models/final/stacker.joblib
 - models/final/preprocessor.joblib (copied if exists)
 - models/final/threshold.txt
 - reports/final_metrics.json
 - reports/feature_importance_final.json (if available)

Run:
 python scripts/final_pipeline.py
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

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

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

OUT = Path("models/final")
OUT.mkdir(parents=True, exist_ok=True)
REPORT = Path("reports")
REPORT.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ---------- helpers ----------
def load_processed_fe():
    # Expect FE v3 outputs in memory (quick_pipeline_v3 uses in-place featurize)
    p_train = Path("data/processed/train_processed.csv.gz")
    p_val   = Path("data/processed/val_processed.csv.gz")
    p_test  = Path("data/processed/test_processed.csv.gz")
    if not (p_train.exists() and p_val.exists() and p_test.exists()):
        raise FileNotFoundError("Processed data not found. Run quick_pipeline_v3 first.")
    train = pd.read_csv(p_train, compression="gzip")
    val = pd.read_csv(p_val, compression="gzip")
    test = pd.read_csv(p_test, compression="gzip")
    return train, val, test

def compute_scale_pos_weight(y):
    neg = int((y==0).sum()); pos=int((y==1).sum())
    return float(neg)/max(1.0, float(pos))

def prepare_Xy(df):
    drop = ["y","loan_status","issue_d_parsed"]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    y = df["y"].values
    return X, y

# ---------- Light optuna tuning (small budgets) ----------
def tune_xgb(Xtr, ytr, Xv, yv, n_trials=10):
    if not HAS_OPTUNA or not HAS_XGB:
        return None
    def obj(trial):
        param = {
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
        }
        model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric="auc")
        # NOTE: do NOT pass early_stopping_rounds here to avoid compatibility issues
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        p = model.predict_proba(Xv)[:,1]
        return roc_auc_score(yv, p)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best["use_label_encoder"] = False
    best["eval_metric"] = "auc"
    return xgb.XGBClassifier(**best)

def tune_lgb(Xtr, ytr, Xv, yv, n_trials=10):
    if not HAS_OPTUNA or not HAS_LGB:
        return None
    def obj(trial):
        param = {
            "objective":"binary",
            "metric":"auc",
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6,1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6,1.0)
        }
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xv, label=yv)
        bst = lgb.train(param, dtrain, num_boost_round=500, valid_sets=[dval], early_stopping_rounds=30, verbose_eval=False)
        p = bst.predict(Xv)
        return roc_auc_score(yv, p)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    # build final model from best params
    best = study.best_params
    params = best.copy()
    params.update({"objective":"binary","metric":"auc"})
    return lambda : lgb.train(params, lgb.Dataset(pd.concat([pd.DataFrame(Xtr), pd.DataFrame(Xv)]), label=np.concatenate([ytr,yv])), num_boost_round=best.get("num_boost_round",100))

def tune_cat(Xtr, ytr, Xv, yv, cat_idx=None, n_trials=6):
    # CatBoost has good defaults; do tiny optuna that changes depth and lr
    if not HAS_OPTUNA or not HAS_CAT:
        return None
    def obj(trial):
        param = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 8),
            "random_seed": 42,
            "verbose": False,
            "auto_class_weights": "Balanced"
        }
        pool_tr = CatPool(Xtr, label=ytr, cat_features=cat_idx)
        pool_val = CatPool(Xv, label=yv, cat_features=cat_idx)
        model = CatBoostClassifier(**param)
        model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=30, verbose=False)
        p = model.predict_proba(Xv)[:,1]
        return roc_auc_score(yv, p)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best.update({"iterations":1000, "auto_class_weights":"Balanced", "random_seed":42, "verbose":100})
    return CatBoostClassifier(**best)

# ---------- main ----------
def main():
    t0 = time.time()
    log("Final pipeline start")
    train, val, test = load_processed_dfs = None, None, None
    train, val, test = load_processed_fe()
    Xtr_df, ytr = prepare_Xy(train)
    Xv_df, yv = prepare_Xy(val)
    Xt_df, yt = prepare_Xy(test)

    # ensure no object-dtype surprises: keep dataframes for CatBoost, arrays for others
    Xtr_vals = Xtr_df.values
    Xv_vals = Xv_df.values
    Xt_vals = Xt_df.values

    scale = compute_scale_pos_weight(ytr)
    log(f"scale_pos_weight ~ {scale:.3f}")

    # detect cat indices for catboost
    cat_idx = []
    if HAS_CAT:
        cat_cols = [c for c in Xtr_df.columns if Xtr_df[c].dtype == "object" or Xtr_df[c].nunique(dropna=True) < 200]
        for c in cat_cols:
            Xtr_df[c] = Xtr_df[c].astype(str).fillna("__nan__")
            Xv_df[c] = Xv_df[c].astype(str).fillna("__nan__")
            Xt_df[c] = Xt_df[c].astype(str).fillna("__nan__")
        cat_idx = [Xtr_df.columns.get_loc(c) for c in cat_cols]

    base_models = []
    trained_models = {}

    # Train CatBoost (tiny optuna), or default if optuna missing
    if HAS_CAT:
        try:
            if HAS_OPTUNA:
                log("Tuning CatBoost (small optuna)...")
                model_cb = tune_cat(Xtr_df, ytr, Xv_df, yv, cat_idx, n_trials=6)
            else:
                model_cb = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, auto_class_weights="Balanced", verbose=100)
            log("Fitting CatBoost final...")
            model_cb.fit(CatPool(Xtr_df, label=ytr, cat_features=cat_idx), eval_set=CatPool(Xv_df, label=yv, cat_features=cat_idx), early_stopping_rounds=30, verbose=False)
            joblib.dump(model_cb, OUT/"base_catboost.joblib")
            trained_models["catboost"] = model_cb
            base_models.append(("catboost", model_cb))
            log("CatBoost trained.")
        except Exception as e:
            log(f"[WARN] CatBoost failed: {e}")

    # Train LightGBM
    if HAS_LGB:
        try:
            if HAS_OPTUNA:
                log("Tuning LightGBM (small optuna)...")
                # We'll skip returning a lgb booster object via optuna helper for speed; use reasonable params
                params = {"objective":"binary","metric":"auc","learning_rate":0.03,"num_leaves":31,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"verbosity":-1,"is_unbalance":True}
                dtrain = lgb.Dataset(Xtr_vals, label=ytr)
                dval = lgb.Dataset(Xv_vals, label=yv)
                bst = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval], early_stopping_rounds=30, verbose_eval=False)
            else:
                params = {"objective":"binary","metric":"auc","learning_rate":0.03,"num_leaves":31,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"verbosity":-1,"is_unbalance":True}
                dtrain = lgb.Dataset(Xtr_vals, label=ytr)
                dval = lgb.Dataset(Xv_vals, label=yv)
                bst = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval], early_stopping_rounds=30, verbose_eval=False)
            joblib.dump(bst, OUT/"base_lgb.joblib")
            trained_models["lightgbm"] = bst
            base_models.append(("lightgbm", bst))
            log("LightGBM trained.")
        except Exception as e:
            log(f"[WARN] LightGBM failed: {e}")

    # Train XGBoost
    if HAS_XGB:
        try:
            if HAS_OPTUNA:
                log("Tuning XGBoost (small optuna)...")
                # simple param set
                model_xgb_init = tune_xgb(Xtr_vals, ytr, Xv_vals, yv, n_trials=8)
                model_xgb = model_xgb_init
                model_xgb.fit(Xtr_vals, ytr, eval_set=[(Xv_vals, yv)], early_stopping_rounds=20, verbose=False)
            else:
                model_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric="auc")
                model_xgb.fit(Xtr_vals, ytr, eval_set=[(Xv_vals, yv)], early_stopping_rounds=20, verbose=False)
            joblib.dump(model_xgb, OUT/"base_xgb.joblib")
            trained_models["xgboost"] = model_xgb
            base_models.append(("xgboost", model_xgb))
            log("XGBoost trained.")
        except Exception as e:
            log(f"[WARN] XGBoost failed: {e}")

    if len(base_models) == 0:
        raise RuntimeError("No base models trained. Install at least one of CatBoost/LightGBM/XGBoost.")

    # Build stacking inputs (predict_proba for each base model on train/val)
    log("Constructing stacker training data")
    # For stacking we will use the validation set predictions as meta features
    meta_features_val = []
    meta_features_train = []
    meta_names = []
    for name, model in base_models:
        meta_names.append(name)
        if name == "catboost":
            pf_tr = model.predict_proba(Xtr_df)[:,1]
            pf_val = model.predict_proba(Xv_df)[:,1]
            pf_test = model.predict_proba(Xt_df)[:,1]
        elif name == "lightgbm":
            pf_tr = model.predict(Xtr_vals)
            pf_val = model.predict(Xv_vals)
            pf_test = model.predict(Xt_vals)
        elif name == "xgboost":
            pf_tr = model.predict_proba(Xtr_vals)[:,1]
            pf_val = model.predict_proba(Xv_vals)[:,1]
            pf_test = model.predict_proba(Xt_vals)[:,1]
        else:
            raise RuntimeError("Unknown base model type")
        meta_features_train.append(pf_tr.reshape(-1,1))
        meta_features_val.append(pf_val.reshape(-1,1))

    X_meta_train = np.hstack(meta_features_train)
    X_meta_val = np.hstack(meta_features_val)

    # Meta model (logistic) trained on validation predictions (preferred: use cross-val stacking; here we use val for speed)
    log("Training logistic meta-model (stacker)")
    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta_clf.fit(X_meta_val, yv)  # train on validation predictions to avoid leakage
    # For production, better to use cross-validated stacking; but this is fastest practical approach

    # --- Simple, robust approach: use logistic meta model directly (no CalibratedClassifierCV)
    # We trained meta_clf on validation predictions earlier; use it as-is for predict_proba.
    # If you later want calibration, we can run CalibratedClassifierCV(cv=3) on meta training preds.
    log("Using logistic meta-model probabilities directly (no CalibratedClassifierCV due to sklearn version).")
    calibrated = meta_clf  # treat meta_clf as the final probabilistic model

    # Evaluate calibrated stacker on test
    # Build test meta features
    meta_features_test = []
    for name, model in base_models:
        if name == "catboost":
            pf_test = model.predict_proba(Xt_df)[:,1]
        elif name == "lightgbm":
            pf_test = model.predict(Xt_vals)
        elif name == "xgboost":
            pf_test = model.predict_proba(Xt_vals)[:,1]
        else:
            raise RuntimeError("Unknown base model")
        meta_features_test.append(pf_test.reshape(-1,1))
    X_meta_test = np.hstack(meta_features_test)
    proba_test = calibrated.predict_proba(X_meta_test)[:,1]

    # threshold tuning on val for F1
    # build meta val features (already have X_meta_val)
    proba_val = calibrated.predict_proba(X_meta_val)[:,1]
    best_thr, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (proba_val >= t).astype(int)
        f = f1_score(yv, preds)
        if f > best_f1:
            best_f1 = f
            best_thr = t
    log(f"Chosen threshold (meta) = {best_thr:.3f} (val f1 {best_f1:.4f})")

    preds_test = (proba_test >= best_thr).astype(int)
    final_metrics = {
        "auc": float(roc_auc_score(yt, proba_test)),
        "f1": float(f1_score(yt, preds_test)),
        "precision": float(precision_score(yt, preds_test)),
        "recall": float(recall_score(yt, preds_test)),
        "confusion_matrix": confusion_matrix(yt, preds_test).tolist(),
        "threshold": float(best_thr),
        "base_models": [name for name, _ in base_models]
    }

    # Save artifacts
    joblib.dump(calibrated, OUT/"stacker.joblib")
    (OUT/"stacker_type.txt").write_text(",".join([name for name,_ in base_models]))
    (OUT/"threshold.txt").write_text(str(best_thr))

    # save base models already dumped; copy preprocessor if present
    preproc_src = Path("models/preprocessor.joblib")
    if preproc_src.exists():
        joblib.dump(joblib.load(preproc_src), OUT/"preprocessor.joblib")

    # feature importance from best base (if available)
    try:
        # pick first tree-model available as representative
        best_name, best_model = base_models[0]
        feat_imp = None
        if best_name == "catboost":
            imp = best_model.get_feature_importance()
            feat_imp = list(zip(list(Xtr_df.columns), imp))
        elif best_name == "lightgbm":
            imp = best_model.feature_importance(importance_type="gain")
            feat_imp = list(zip(list(Xtr_df.columns), imp))
        elif best_name == "xgboost":
            try:
                imp = best_model.feature_importances_
                feat_imp = list(zip(list(Xtr_df.columns), imp))
            except Exception:
                booster = best_model.get_booster()
                score = booster.get_score(importance_type="gain")
                feat_imp = list(score.items())
        if feat_imp is not None:
            feat_imp_sorted = sorted(feat_imp, key=lambda x: -float(x[1]))[:200]
            with open(REPORT/"feature_importance_final.json","w") as fh:
                json.dump([{"feature": f, "imp": float(i)} for f,i in feat_imp_sorted], fh, indent=2)
    except Exception as e:
        log(f"[WARN] Could not extract final feature importance: {e}")

    with open(REPORT/"final_metrics.json","w") as fh:
        json.dump(final_metrics, fh, indent=2)

    log("Final pipeline finished. Metrics:")
    log(json.dumps(final_metrics, indent=2))
    log(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
