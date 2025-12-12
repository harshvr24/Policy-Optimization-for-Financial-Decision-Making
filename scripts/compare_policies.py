#!/usr/bin/env python3
"""
scripts/compare_policies.py

Compare Deep Learning MLP classifier vs RL policy on the processed test set.

Outputs:
 - reports/policy_comparison.json
 - reports/dl_confusion.csv
 - reports/rl_confusion.csv

Usage:
  python scripts/compare_policies.py \
    --test data/processed/test_processed.csv.gz \
    --dl_model models/dl/dl_model.pt \
    --dl_preproc models/preprocessor_dl.joblib \
    --rl_policy models/rl/rl_policy.pkl \
    --val data/processed/val_processed.csv.gz  # optional, used to find DL threshold

Notes:
 - script is defensive about missing artifacts and prints clear error messages.
 - RL policy pickle may be a sklearn classifier (preferred fallback) or a wrapper with .predict.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import os
import sys
import math

# PyTorch imports for DL model
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# -----------------------------------------------------------------------------
# Helpers: reward function & DL model loader
# -----------------------------------------------------------------------------
def compute_reward_from_row(row):
    # replicate reward used when preparing RL dataset
    try:
        loan_amnt = float(row.get("loan_amnt", 0.0) or 0.0)
        int_rate = float(row.get("int_rate", 0.0) or 0.0)
    except Exception:
        return 0.0
    status = str(row.get("loan_status", "")).lower()
    paid_tokens = ["fully paid", "fully_paid", "paid", "current"]
    if any(t in status for t in paid_tokens):
        reward = loan_amnt * (int_rate / 100.0)
    else:
        reward = -loan_amnt
    return float(reward)

def load_dl_state(model_path):
    p = Path(model_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"DL model file not found or empty: {p}")
    try:
        state = torch.load(str(p), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load DL model file {p}: {e}")
    return state

class EvalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def dl_predict_proba(test_df, dl_state, preproc):
    """
    Returns probability vector for test_df rows based on dl_state and preproc.
    preproc is expected to be a dict with scaler under 'scaler' and 'features' list.
    dl_state should be the dict saved by train_dl_model.py (contains model_state_dict and args).
    """
    if "model_state_dict" not in dl_state:
        raise RuntimeError("DL state file does not contain 'model_state_dict' key")
    features = dl_state.get("scaler_features") or preproc.get("features")
    if features is None:
        raise RuntimeError("Feature list not found in model state or preprocessor")

    X_df = test_df[features].copy()
    # coerce numeric-like strings
    for c in X_df.columns:
        if X_df[c].dtype == object:
            X_df[c] = pd.to_numeric(X_df[c].astype(str).str.rstrip('%'), errors='coerce')
    med = X_df.median()
    X_df = X_df.fillna(med)
    scaler = preproc.get("scaler", None)
    if scaler is not None:
        X = scaler.transform(X_df.values)
    else:
        X = X_df.values

    input_dim = X.shape[1]
    hidden_dims = tuple(dl_state.get("args", {}).get("hidden_dims", [256,128]))
    dropout = dl_state.get("args", {}).get("dropout", 0.2)
    model = EvalMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(dl_state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype('float32')))
        probs = torch.sigmoid(logits).numpy()
    return probs.ravel()

def load_rl_policy(policy_path):
    p = Path(policy_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"RL policy file not found or empty: {p}")
    try:
        policy = joblib.load(str(p))
    except Exception as e:
        raise RuntimeError(f"Failed to load RL policy pickle {p}: {e}")
    return policy

def policy_predict(policy, X_np):
    # robustly call predict on policy object
    # if policy is sklearn estimator -> predict returns {0,1}
    # if policy is wrapper with .algo.predict -> try that
    if hasattr(policy, "predict"):
        out = policy.predict(X_np)
        out = np.asarray(out)
        # if continuous probabilities returned, threshold at 0.5
        if out.dtype.kind in ("f","d"):
            if out.ndim==2 and out.shape[1] > 1:
                # class-probs matrix -> use column 1 if present or argmax
                try:
                    return (out[:,1] >= 0.5).astype(int)
                except Exception:
                    return out.argmax(axis=1)
            return (out >= 0.5).astype(int)
        else:
            return out.astype(int)
    # fallback: try .algo.predict
    if hasattr(policy, "algo") and hasattr(policy.algo, "predict"):
        out = policy.algo.predict(X_np)
        out = np.asarray(out)
        if out.ndim==2 and out.shape[1] > 1:
            return out.argmax(axis=1)
        if out.dtype.kind in ("f","d"):
            return (out >= 0.5).astype(int)
        return out.astype(int)
    raise RuntimeError("Loaded RL policy has no usable predict method")

# -----------------------------------------------------------------------------
# Metrics & reward evaluation helpers
# -----------------------------------------------------------------------------
def classification_metrics(y_true, preds, probs=None):
    res = {}
    res["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()
    res["f1"] = float(f1_score(y_true, preds))
    res["precision"] = float(precision_score(y_true, preds))
    res["recall"] = float(recall_score(y_true, preds))
    if probs is not None and len(np.unique(y_true))>1:
        try:
            res["auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            res["auc"] = None
    else:
        res["auc"] = None
    return res

def evaluate_expected_reward(df, actions, reward_scale=None):
    # actions: np array 0/1 for each row in df
    rewards = []
    for i, row in df.iterrows():
        r = compute_reward_from_row(row)
        rewards.append(r if int(actions[i])==1 else 0.0)
    rewards = np.asarray(rewards, dtype=float)
    if reward_scale and reward_scale != 0 and abs(reward_scale-1.0)>1e-9:
        # if reward_scale was applied during RL dataset creation, and user provided it,
        # multiply back; (prepare code divided rewards by scale -> to get original multiply)
        rewards = rewards * float(reward_scale)
    return {"sum_reward": float(rewards.sum()), "avg_reward": float(rewards.mean()), "std_reward": float(rewards.std())}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/processed/test_processed.csv.gz", help="processed test CSV (gzip)")
    p.add_argument("--dl_model", default="models/dl/dl_model.pt", help="dl model state file")
    p.add_argument("--dl_preproc", default="models/preprocessor_dl.joblib", help="dl preprocessor joblib")
    p.add_argument("--rl_policy", default="models/rl/rl_policy.pkl", help="rl policy pickle (sklearn or wrapper)")
    p.add_argument("--val", default="data/processed/val_processed.csv.gz", help="optional: val CSV to pick DL threshold")
    p.add_argument("--out", default="reports/policy_comparison.json", help="output JSON report")
    args = p.parse_args()

    # load test set
    test_p = Path(args.test)
    if not test_p.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_p}")
    df_test = pd.read_csv(test_p, compression='gzip')
    if "y" not in df_test.columns:
        raise RuntimeError("Test CSV must include 'y' column")

    # 1) Load DL artifacts and predict probabilities
    dl_state = None
    dl_probs = None
    dl_threshold = 0.5
    if Path(args.dl_model).exists() and Path(args.dl_preproc).exists():
        try:
            dl_state = load_dl_state(args.dl_model)
            preproc = joblib.load(args.dl_preproc)
            # If validation provided, compute best F1 threshold on val set
            if args.val and Path(args.val).exists():
                df_val = pd.read_csv(args.val, compression='gzip')
                val_probs = dl_predict_proba(df_val, dl_state, preproc)
                # find threshold that maximizes F1 on val set
                best_thr = 0.5
                best_f1 = -1.0
                yv = df_val["y"].astype(int).values
                # try thresholds in [0.01..0.99]
                for thr in np.linspace(0.01, 0.99, 99):
                    preds = (val_probs >= thr).astype(int)
                    f1 = f1_score(yv, preds) if len(np.unique(yv))>1 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = float(thr)
                dl_threshold = best_thr
                print(f"[INFO] Selected DL threshold from val for max F1 = {dl_threshold:.3f} (val f1={best_f1:.4f})")
            # compute probabilities on test
            dl_probs = dl_predict_proba(df_test, dl_state, preproc)
        except Exception as e:
            print("[WARN] DL model predict failed:", e)
            dl_state = None
    else:
        print("[WARN] DL artifacts missing; skipping DL evaluation.")

    # 2) Load RL policy and predict actions on test set features
    rl_policy = None
    rl_actions = None
    reward_scale = None
    # try reading reward_scale from data/rl/rl_dataset_info.json if present
    info_path = Path("data/rl/rl_dataset_info.json")
    if info_path.exists():
        try:
            info = json.load(open(info_path))
            # dataset_info might not include reward_scale; we also check npz
            reward_scale = info.get("reward_scale", None)
        except Exception:
            reward_scale = None

    if Path(args.rl_policy).exists():
        try:
            rl_policy = load_rl_policy(args.rl_policy)
            # build feature matrix expected by policy: same columns as used in RL dataset if possible
            # Try reading data/rl/rl_dataset_info.json for feature_cols
            feature_cols = None
            if info_path.exists():
                try:
                    feature_cols = json.load(open(info_path)).get("feature_cols", None)
                except Exception:
                    feature_cols = None
            if feature_cols is None:
                # fallback: use all columns except y/loan_status/ids
                exclude = {"y","loan_status","issue_d_parsed","id","member_id","url","title","desc"}
                feature_cols = [c for c in df_test.columns if c not in exclude]
            X_df = df_test[feature_cols].copy()
            for c in X_df.columns:
                if X_df[c].dtype == object:
                    X_df[c] = pd.to_numeric(X_df[c].astype(str).str.rstrip('%'), errors='coerce')
            X_df = X_df.fillna(X_df.median())
            X = X_df.values.astype(np.float32)
            rl_actions = policy_predict(rl_policy, X)
            # ensure length matches
            if len(rl_actions) != len(df_test):
                # try predicting per-row
                rl_actions = np.asarray([policy_predict(rl_policy, X[i:i+1])[0] for i in range(len(X))])
        except Exception as e:
            print("[WARN] RL policy load/predict failed:", e)
            rl_policy = None
    else:
        print("[WARN] RL policy file missing; skipping RL evaluation.")

    # 3) Compute metrics
    y_true = df_test["y"].astype(int).values
    report = {"n_test": int(len(df_test))}

    # DL metrics (if available)
    if dl_probs is not None:
        dl_preds = (dl_probs >= dl_threshold).astype(int)
        m = classification_metrics(y_true, dl_preds, probs=dl_probs)
        rr = evaluate_expected_reward(df_test, dl_preds, reward_scale=reward_scale)
        m.update(rr)
        m["threshold_used"] = float(dl_threshold)
        report["dl"] = m
        # save confusion csv
        df_cm = pd.DataFrame(confusion_matrix(y_true, dl_preds), index=["true_0","true_1"], columns=["pred_0","pred_1"])
        Path("reports").mkdir(parents=True, exist_ok=True)
        df_cm.to_csv("reports/dl_confusion.csv")
    else:
        report["dl"] = None

    # RL metrics (if available)
    if rl_actions is not None:
        # rl_actions already binary
        rl_preds = np.asarray(rl_actions).astype(int)
        # RL might not have a probability vector; we leave auc None
        m2 = classification_metrics(y_true, rl_preds, probs=None)
        rr2 = evaluate_expected_reward(df_test, rl_preds, reward_scale=reward_scale)
        m2.update(rr2)
        report["rl"] = m2
        df_cm2 = pd.DataFrame(confusion_matrix(y_true, rl_preds), index=["true_0","true_1"], columns=["pred_0","pred_1"])
        Path("reports").mkdir(parents=True, exist_ok=True)
        df_cm2.to_csv("reports/rl_confusion.csv")
    else:
        report["rl"] = None

    # 4) simple comparison summary
    summary = {}
    if report.get("dl"):
        summary["dl_auc"] = report["dl"].get("auc")
        summary["dl_f1"] = report["dl"].get("f1")
        summary["dl_avg_reward"] = report["dl"].get("avg_reward")
    if report.get("rl"):
        summary["rl_f1"] = report["rl"].get("f1")
        summary["rl_avg_reward"] = report["rl"].get("avg_reward")
    report["summary"] = summary

    # Save JSON report
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(report, fh, indent=2)
    print("[DONE] Policy comparison saved to", outp)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
