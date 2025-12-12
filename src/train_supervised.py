# src/train_supervised.py
"""
Train supervised baselines (XGBoost + PyTorch MLP) on processed LendingClub data.

This script is verbose and robust:
 - computes class weight from train set and applies to XGBoost and MLP
 - creates models/ and reports/ if missing
 - skips models when libraries are not installed (with clear warnings)
 - saves metrics to reports/metrics.json always (may be empty if nothing trained)
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Optional libs
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_processed(processed_dir: str):
    train_path = os.path.join(processed_dir, "train_processed.csv.gz")
    val_path = os.path.join(processed_dir, "val_processed.csv.gz")
    test_path = os.path.join(processed_dir, "test_processed.csv.gz")

    for p in (train_path, val_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Processed file not found: {p}")

    log(f"Loading processed CSVs from {processed_dir}")
    train = pd.read_csv(train_path, compression="gzip")
    val = pd.read_csv(val_path, compression="gzip")
    test = pd.read_csv(test_path, compression="gzip")
    log(f"Loaded shapes -> train: {train.shape}, val: {val.shape}, test: {test.shape}")
    return train, val, test


def split_Xy(df: pd.DataFrame):
    if "y" not in df.columns:
        raise KeyError("'y' column missing from processed dataframe")
    y = df["y"].values
    X = df.drop(columns=["y", "loan_status", "issue_d_parsed"], errors="ignore")
    return X.values.astype(float), y.astype(int)


def eval_and_report(model_name: str, preds_proba: np.ndarray, preds_label: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    try:
        auc = float(roc_auc_score(y_true, preds_proba))
    except Exception:
        auc = None
    try:
        f1 = float(f1_score(y_true, preds_label))
    except Exception:
        f1 = None
    try:
        prec = float(precision_score(y_true, preds_label))
    except Exception:
        prec = None
    try:
        rec = float(recall_score(y_true, preds_label))
    except Exception:
        rec = None
    try:
        cm = confusion_matrix(y_true, preds_label).tolist()
    except Exception:
        cm = None
    return {
        "model": model_name,
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm
    }


# ----------------------------
# XGBoost training (with scale_pos_weight)
# ----------------------------
def train_xgb(X_train, y_train, X_val, y_val, params=None):
    """
    Train an XGBoost sklearn-wrapping classifier with robust fallback for older/newer xgboost APIs.
    This function expects scale_pos_weight already computed and passed in params if needed.
    """
    if xgb is None:
        raise RuntimeError("xgboost not available")
    if params is None:
        params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "use_label_encoder": False, "eval_metric": "auc"}

    model = xgb.XGBClassifier(**params)
    log("Fitting XGBoost (this may take a while)...")

    # Try the most featureful call first (early stopping). If it errors, gracefully fallback.
    try:
        model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError as te:
        log(f"[WARN] XGBoost fit signature didn't accept early_stopping_rounds: {te}")
        try:
            # Try with eval_set only
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError as te2:
            log(f"[WARN] XGBoost fit signature didn't accept eval_set either: {te2}")
            # Final fallback: plain fit
            model.fit(X_train, y_train, verbose=False)

    log("XGBoost training complete.")
    return model


# ----------------------------
# PyTorch MLP training (logits + BCEWithLogitsLoss)
# ----------------------------
if torch is not None and nn is not None:
    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout=0.2):
            super().__init__()
            layers = []
            d = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                d = h
            layers.append(nn.Linear(d, 1))   # output logits
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)   # logits
else:
    MLP = None


def train_mlp_torch(X_train, y_train, X_val, y_val, device="cpu", epochs=10, batch_size=1024, lr=1e-3, pos_weight: float = 1.0):
    if torch is None or MLP is None:
        raise RuntimeError("PyTorch not available")
    device = device if torch.cuda.is_available() else "cpu"
    model = MLP(input_dim=X_train.shape[1])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # pos_weight should be a torch tensor on the device
    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_auc = 0.0
    best_state = None
    log(f"Training MLP on device {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_val).float().to(device)
            logits_v = model(Xv)
            pv = torch.sigmoid(logits_v).cpu().numpy()
            try:
                auc = roc_auc_score(y_val, pv)
            except Exception:
                auc = 0.0
            if auc > best_auc:
                best_auc = auc
                best_state = model.state_dict()
        log(f"  epoch {epoch+1}/{epochs} - val_auc={auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    log("MLP training complete.")
    return model


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def copy_preprocessor(artifacts_dir: str, out_dir: str):
    src = os.path.join(artifacts_dir, "preprocessor.joblib")
    if os.path.exists(src):
        dst = os.path.join(out_dir, "preprocessor.joblib")
        joblib.dump(joblib.load(src), dst)
        log(f"Copied preprocessor to {dst}")
    else:
        log("No preprocessor artifact found to copy.")


def main(processed_dir: str = "data/processed", artifacts_dir: str = "artifacts", out_dir: str = "models"):
    start_time = time.time()
    log("Starting supervised training pipeline")
    # ensure dirs
    ensure_dirs(out_dir, "reports")
    metrics: List[Dict[str, Any]] = []

    # Load processed data
    try:
        train, val, test = load_processed(processed_dir)
    except Exception as e:
        log(f"[FATAL] Could not load processed data: {e}")
        raise

    # Split into X/y
    X_train, y_train = split_Xy(train)
    X_val, y_val = split_Xy(val)
    X_test, y_test = split_Xy(test)

    # compute class weight using training set
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        log("[FATAL] No positive examples in training data.")
        raise RuntimeError("No positive examples in training data.")
    scale_pos_weight = float(neg) / float(pos)
    log(f"Computed scale_pos_weight from training set: neg={neg}, pos={pos}, scale_pos_weight={scale_pos_weight:.3f}")

    # Copy preprocessor artifact for convenience
    copy_preprocessor(artifacts_dir, out_dir)

    # Train XGBoost if available
    if xgb is None:
        log("[WARN] xgboost not installed; skipping XGBoost training.")
    else:
        try:
            # pass class weight into params
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
            xgb_model = train_xgb(X_train, y_train, X_val, y_val, params=params)
            path = os.path.join(out_dir, "xgb_model.joblib")
            joblib.dump(xgb_model, path)
            log(f"Saved XGBoost model to {path}")
            proba = xgb_model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)
            metrics.append(eval_and_report("xgboost", proba, preds, y_test))
        except Exception as e:
            log(f"[ERROR] XGBoost training failed: {e}")

    # Train PyTorch MLP if available
    if torch is None:
        log("[WARN] PyTorch not installed; skipping MLP training.")
    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mlp = train_mlp_torch(X_train, y_train, X_val, y_val, device=device, epochs=10, pos_weight=scale_pos_weight)
            mlp_path = os.path.join(out_dir, "mlp_model.pt")
            torch.save(mlp.state_dict(), mlp_path)
            log(f"Saved MLP model to {mlp_path}")
            mlp.eval()
            with torch.no_grad():
                logits_test = mlp(torch.from_numpy(X_test).float().to(device))
                p = torch.sigmoid(logits_test).cpu().numpy()
                preds = (p >= 0.5).astype(int)
                metrics.append(eval_and_report("mlp_torch", p, preds, y_test))
        except Exception as e:
            log(f"[ERROR] MLP training failed: {e}")

    # Write metrics JSON (always write, even if empty)
    report_path = os.path.join("reports", "metrics.json")
    with open(report_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"Wrote metrics to {report_path}")

    elapsed = time.time() - start_time
    log(f"Finished supervised training pipeline in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train supervised baselines on processed data")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    try:
        rc = main(args.processed_dir, args.artifacts_dir, args.out_dir)
        sys.exit(rc if isinstance(rc, int) else 0)
    except Exception as exc:
        log(f"[FATAL] Unhandled exception: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
