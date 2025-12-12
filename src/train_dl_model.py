#!/usr/bin/env python3
"""
src/train_dl_model.py

Train a simple MLP (PyTorch) for binary classification.
Saves:
 - models/dl/dl_model.pt          (PyTorch state + metadata json)
 - models/preprocessor_dl.joblib  (scikit-learn preprocessor with feature list)

Inputs: processed CSVs with 'y' column and other features (no post-outcome fields).
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# -------- utils --------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
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
        return self.net(x).squeeze(-1)  # logits

def load_df(path: str, nrows: int = None):
    return pd.read_csv(path, compression='gzip', nrows=nrows)

def prepare_features(df: pd.DataFrame, feature_cols=None):
    # Exclude target / known columns
    to_drop = {'y','loan_status','issue_d_parsed'}
    cols = [c for c in df.columns if c not in to_drop]
    if feature_cols is not None:
        # keep only requested columns in same order
        cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].copy()
    # numeric safe conversion for object-like numeric columns
    for c in X.columns:
        if X[c].dtype == object:
            # try to coerce percent strings etc.
            X[c] = pd.to_numeric(X[c].astype(str).str.rstrip('%'), errors='coerce')
    return X, cols

def compute_metrics(y_true, proba, threshold=0.5):
    preds = (proba >= threshold).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "f1": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds)),
        "recall": float(recall_score(y_true, preds)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist()
    }
    return metrics

# -------- main training flow --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="train processed csv (gzip)")
    parser.add_argument("--val", required=True, help="val processed csv (gzip)")
    parser.add_argument("--out_dir", default="models/dl", help="output dir to save model & preprocessor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256,128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nrows", type=int, default=None, help="optional: limit rows for quick runs")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "reports").mkdir(parents=True, exist_ok=True)

    print("Loading data")
    df_train = load_df(args.train, nrows=args.nrows)
    df_val = load_df(args.val, nrows=args.nrows)

    # Extract features and target
    X_train_df, feature_cols = prepare_features(df_train)
    X_val_df, _ = prepare_features(df_val, feature_cols=feature_cols)
    if "y" not in df_train.columns:
        raise RuntimeError("Training CSV must contain 'y' column as target")
    y_train = df_train["y"].astype(int).values
    y_val = df_val["y"].astype(int).values

    # Fill numeric NaNs with median (trained on train)
    X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce')
    X_val_df = X_val_df.apply(pd.to_numeric, errors='coerce')
    medians = X_train_df.median()
    X_train_df = X_train_df.fillna(medians)
    X_val_df = X_val_df.fillna(medians)

    # Fit scaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_df.values)
    X_va = scaler.transform(X_val_df.values)

    # Save preprocessor (scaler + feature list)
    preproc = {"scaler": scaler, "features": feature_cols}
    joblib.dump(preproc, str(Path(args.out_dir).parent / "preprocessor_dl.joblib"))
    print("Saved preprocessor to", str(Path(args.out_dir).parent / "preprocessor_dl.joblib"))

    # Prepare dataloaders
    train_ds = TabularDataset(X_tr, y_train)
    val_ds = TabularDataset(X_va, y_val)
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X_tr.shape[1], hidden_dims=tuple(args.hidden_dims), dropout=args.dropout).to(device)

    # loss with pos_weight to handle imbalance
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    if pos == 0:
        raise RuntimeError("No positive examples in training data")
    pos_weight = torch.tensor(float(neg) / float(pos), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # scheduler: use ReduceLROnPlateau; older torch versions may not accept verbose kwarg
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=False)
    except TypeError:
        # older torch: retry without verbose
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)


    best_val_auc = -1.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print("Start training on device:", device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Validation
        model.eval()
        all_logits = []
        all_y = []
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(yb.detach().cpu().numpy())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        all_logits = np.concatenate(all_logits) if len(all_logits) else np.array([])
        all_y = np.concatenate(all_y) if len(all_y) else np.array([])
        proba = 1.0 / (1.0 + np.exp(-all_logits)) if all_logits.size else np.array([])
        val_auc = float(roc_auc_score(all_y, proba)) if all_y.size and len(np.unique(all_y)) > 1 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

        scheduler.step(val_loss)

        # save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "model_state_dict": model.state_dict(),
                "scaler_features": feature_cols,
                "args": vars(args),
                "epoch": epoch,
                "val_auc": val_auc,
                "pos_weight": float(pos_weight.cpu().item())
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce any state")

    # Save model
    model_file = Path(args.out_dir) / "dl_model.pt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(model_file))
    print("Saved model state to", str(model_file))

    # Also save a small JSON with training history & metrics
    summary = {
        "best_val_auc": best_val_auc,
        "history": {k: (v if isinstance(v, list) else [v]) for k,v in history.items()},
        "model_file": str(model_file),
        "preprocessor_file": str(Path(args.out_dir).parent / "preprocessor_dl.joblib")
    }
    with open(Path(args.out_dir).parent / "dl_training_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Training complete. Summary saved.")

if __name__ == "__main__":
    main()
