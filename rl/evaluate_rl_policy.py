#!/usr/bin/env python3
"""
rl/evaluate_rl_policy.py

Load a saved policy (policy wrapper created by train_rl_agent.py) and evaluate its
average reward on a test set (deterministic one-step evaluation).

This uses the same reward function as prepare_rl_dataset.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import json
import sys

def compute_reward_from_row(row):
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="policy pickle (rl_policy.pkl)")
    p.add_argument("--test", required=True, help="processed test csv (gzip)")
    p.add_argument("--out", default="reports/rl_eval.json", help="output metrics json")
    p.add_argument("--nrows", type=int, default=None, help="limit test rows for quick eval")
    p.add_argument("--reward_scale", type=float, default=None, help="if dataset uses a reward scale, pass it")
    args = p.parse_args()

    policy = joblib.load(args.policy)
    df = pd.read_csv(args.test, compression="gzip", nrows=args.nrows)
    # Feature columns: try to read from data/rl/rl_dataset_info.json if exists
    info_path = Path("data/rl/rl_dataset_info.json")
    if info_path.exists():
        info = json.load(open(info_path))
        feature_cols = info.get("feature_cols", None)
    else:
        # fallback: use all except y/loan_status/ids
        exclude = {"y","loan_status","issue_d_parsed","id","member_id","url","title","desc"}
        feature_cols = [c for c in df.columns if c not in exclude]

    X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(df[feature_cols].median())
    X = X_df.values.astype(np.float32)

    # policy.predict should accept (n_samples, dim) and return array of actions or action-values
    try:
        actions = policy.predict(X)
    except Exception as e:
        # some wrappers use `algo.predict` naming - try that
        try:
            actions = policy.algo.predict(X)
        except Exception as e2:
            raise RuntimeError(f"Policy predict failed: {e}; fallback also failed: {e2}")

    # Ensure actions are integers 0/1
    actions = np.asarray(actions).ravel()
    # if actions are continuous, threshold at 0.5 else cast
    if actions.dtype.kind in ('f','d'):
        actions = (actions >= 0.5).astype(int)
    else:
        actions = actions.astype(int)

    # compute rewards array (approve action -> compute reward from history; reject -> 0)
    rewards = []
    for i, row in df.iterrows():
        r = compute_reward_from_row(row)
        act = int(actions[i])
        rewards.append(r if act == 1 else 0.0)
    rewards = np.asarray(rewards, dtype=np.float32)
    # if dataset was scaled in prepare step, and user knows reward_scale, rescale back
    if args.reward_scale:
        rewards = rewards * float(args.reward_scale)

    avg_reward = float(rewards.mean())
    std_reward = float(rewards.std())
    result = {
        "n": int(len(rewards)),
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "sum_reward": float(rewards.sum()),
        "policy_actions": {
            "approve_count": int((actions == 1).sum()),
            "reject_count": int((actions == 0).sum())
        }
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("Saved RL evaluation to", args.out)
    print(result)

if __name__ == "__main__":
    main()
