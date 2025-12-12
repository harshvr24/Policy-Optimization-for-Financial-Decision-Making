#!/usr/bin/env python3
"""
src/evaluate_dl.py

Load a saved PyTorch MLP state (saved by train_dl_model.py) and preprocessor,
evaluate on test CSV, save metrics JSON.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn

def load_state(model_path: str):
    model_file = Path(model_path)
    if not model_file.exists() or model_file.stat().st_size == 0:
        raise FileNotFoundError(
            f"Model file not found or is empty: {model_path}\n"
            "This usually means training failed before saving. Re-run training and ensure it completes successfully."
        )
    try:
        state = torch.load(model_path, map_location="cpu")
    except EOFError as e:
        raise EOFError(
            f"Could not read model file {model_path} (EOF). The file may be corrupted.\n"
            "If you recently ran training and it failed, delete the partial file and re-run training.\n"
            f"Original error: {e}"
        )
    return state


class EvalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256,128), dropout=0.2):
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

def prepare_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    # coerce numeric-like columns
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c].astype(str).str.rstrip('%'), errors='coerce')
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to dl_model.pt (state dict saved by training)")
    parser.add_argument("--preproc", required=True, help="path to preprocessor joblib saved by training")
    parser.add_argument("--test", required=True, help="test processed csv (gzip)")
    parser.add_argument("--out", default="reports/dl_metrics.json", help="output json metrics path")
    parser.add_argument("--threshold", type=float, default=None, help="optional threshold override for decision")
    args = parser.parse_args()

    state = load_state(args.model)
    preproc = joblib.load(args.preproc)

    feature_cols = state.get("scaler_features") if "scaler_features" in state else preproc.get("features")
    if feature_cols is None:
        raise RuntimeError("Feature list not found in model state or preprocessor")

    hidden_dims = state.get("args", {}).get("hidden_dims", [256,128])
    dropout = state.get("args", {}).get("dropout", 0.2)

    # build model & load weights
    input_dim = len(feature_cols)
    model = EvalMLP(input_dim=input_dim, hidden_dims=tuple(hidden_dims), dropout=dropout)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # prepare test data
    df_test = pd.read_csv(args.test, compression="gzip")
    if "y" not in df_test.columns:
        raise RuntimeError("Test CSV must contain 'y' column")

    X_test_df = prepare_features(df_test, feature_cols)
    # fill missing with medians from preprocessor (scaler not stored as median; use scaler if available)
    scaler = preproc.get("scaler", None)
    if scaler is not None:
        X_test_filled = X_test_df.fillna(X_test_df.median())
        X_test = scaler.transform(X_test_filled.values)
    else:
        # fallback: fillna with median and use values
        X_test = X_test_df.fillna(X_test_df.median()).values

    y_test = df_test["y"].astype(int).values

    # predict
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test.astype('float32')))
        proba = torch.sigmoid(logits).numpy()

    threshold = args.threshold if args.threshold is not None else 0.5
    preds = (proba >= threshold).astype(int)

    metrics = {
        "model_file": args.model,
        "preprocessor_file": args.preproc,
        "threshold_used": float(threshold),
        "auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Evaluation complete. Metrics saved to", str(outp))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
