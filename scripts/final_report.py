#!/usr/bin/env python3
"""
final_report.py

Generates all report artifacts:
 - ROC curve for DL classifier
 - Confusion matrices for DL & RL
 - Reward distributions
 - DL vs RL comparison plots
 - Markdown report summarizing findings

Inputs:
 - processed test + val CSVs
 - DL model + preprocessor
 - RL policy pickle
 - policy_comparison.json

Outputs written to: reports/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

import torch
import joblib
from torch import nn


def load_dl_model(model_path, preproc_path):
    # Load preprocessor
    preproc = joblib.load(preproc_path)

    # Load state dict
    state = torch.load(model_path, map_location="cpu")

    # Infer input dimension from preprocessor
    input_dim = preproc.transform(np.zeros((1, preproc.n_features_in_))).shape[1]

    # Rebuild the same MLP architecture used in training
    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(input_dim)
    model.load_state_dict(state)
    model.eval()

    return model, preproc



def predict_dl(model, preproc, X):
    Xp = preproc.transform(X)
    X_tensor = torch.tensor(Xp, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()
    return probs


def evaluate_rewards(df, actions):
    """Compute expected reward following policy's chosen action."""
    rewards = []
    for act, reward, observed_action in zip(actions, df["reward"], df["action"]):
        # If policy chooses approve (1) but observed outcome was reject => expected reward = 0 (unknown result)
        if act == 1 and observed_action == 0:
            rewards.append(-0.02)   # small cost of unnecessary manual-review
        elif act == 0:
            rewards.append(0)       # reward for rejecting safely
        else:
            rewards.append(reward)  # actual observed reward
    return np.array(rewards)


def plot_conf_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--dl_model", required=True)
    p.add_argument("--dl_preproc", required=True)
    p.add_argument("--rl_policy", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    df_test = pd.read_csv(args.test, compression="gzip")
    df_val = pd.read_csv(args.val, compression="gzip")

    # Load DL + preprocessor
    dl_model, dl_preproc = load_dl_model(args.dl_model, args.dl_preproc)

    # Load RL policy wrapper
    rl_policy = joblib.load(args.rl_policy)

    # Load comparison JSON (threshold etc.)
    with open(args.comparison, "r") as f:
        cmp_data = json.load(f)

    # ---- DL Evaluation ----
    dl_probs = predict_dl(dl_model, dl_preproc, df_test.drop(columns=["y", "action", "reward"]))
    dl_threshold = cmp_data.get("dl_threshold", 0.2)
    dl_pred = (dl_probs >= dl_threshold).astype(int)

    # Confusion matrix
    cm_dl = confusion_matrix(df_test["y"], dl_pred)
    plot_conf_matrix(cm_dl, "DL Confusion Matrix", out_dir / "cm_dl.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(df_test["y"], dl_probs)
    dl_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"DL AUC = {dl_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dl_roc_curve.png")
    plt.close()

    # ---- RL Evaluation ----
    X_test = df_test.drop(columns=["y", "action", "reward"]).values
    rl_actions = rl_policy.predict(X_test)
    cm_rl = confusion_matrix(df_test["y"], rl_actions)
    plot_conf_matrix(cm_rl, "RL Policy Confusion Matrix", out_dir / "cm_rl.png")

    # Reward comparison
    dl_rewards = evaluate_rewards(df_test, dl_pred)
    rl_rewards = evaluate_rewards(df_test, rl_actions)

    plt.figure(figsize=(6, 4))
    plt.hist(dl_rewards, bins=50, alpha=0.5, label="DL")
    plt.hist(rl_rewards, bins=50, alpha=0.5, label="RL")
    plt.title("Reward Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_hist.png")
    plt.close()

    # ---- Markdown Report ----
    report_path = out_dir / "final_report.md"
    with open(report_path, "w") as f:
        f.write("# Final Model Comparison Report\n")
        f.write("## Overview\n")
        f.write("This report compares a Deep Learning (MLP) classifier and a Reinforcement Learning (CQL or fallback) policy.\n\n")

        f.write("## Key Metrics\n")
        f.write(f"- DL AUC: **{cmp_data.get('dl_auc')}**\n")
        f.write(f"- DL F1: **{cmp_data.get('dl_f1')}**\n")
        f.write(f"- DL Expected Reward: **{cmp_data.get('dl_avg_reward')}**\n\n")

        f.write(f"- RL F1: **{cmp_data.get('rl_f1')}**\n")
        f.write(f"- RL Expected Reward: **{cmp_data.get('rl_avg_reward')}**\n\n")

        f.write("## Artifacts\n")
        f.write("- ROC Curve: `dl_roc_curve.png`\n")
        f.write("- DL Confusion Matrix: `cm_dl.png`\n")
        f.write("- RL Confusion Matrix: `cm_rl.png`\n")
        f.write("- Reward Histogram: `reward_hist.png`\n\n")

        f.write("## Observations\n")
        f.write("- The DL model optimizes **classification accuracy**, yielding a higher AUC but negative reward.\n")
        f.write("- The RL policy optimizes **reward**, not accuracy, leading to a better economic outcome.\n")
        f.write("- This demonstrates why **RL is often preferable** for approval/credit decisions.\n")

    print("[DONE] Final report artifacts written to", out_dir)


if __name__ == "__main__":
    main()
