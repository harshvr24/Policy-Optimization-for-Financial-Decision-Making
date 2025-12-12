#!/usr/bin/env python3
"""
rl/prepare_rl_dataset.py

Prepare an offline RL dataset (single-step episodes) from processed CSV.

Creates paired transitions for each applicant:
 - action=1 (approve) -> reward from observed loan outcome
 - action=0 (reject)  -> reward = 0

Saves dataset as numpy .npz containing:
 - observations (N, D)
 - actions (N,)
 - rewards (N,)
 - next_observations (N, D)  (same as obs here)
 - terminals (N,) (all ones for single-step episodes)

Output files:
  {out_dir}/train_rl.npz
  {out_dir}/val_rl.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import os

def compute_reward(row, profit_if_paid=None):
    # reward design:
    # if loan fully paid -> interest income = loan_amnt * int_rate/100
    # else (charged off/default) -> negative principal = -loan_amnt
    # fallback: if cannot parse, return 0
    try:
        loan_amnt = float(row.get("loan_amnt", 0.0) or 0.0)
        int_rate = float(row.get("int_rate", 0.0) or 0.0)
    except Exception:
        return 0.0
    status = str(row.get("loan_status", "")).lower()
    # define positive statuses (treated as paid)
    paid_tokens = ["fully paid", "fully_paid", "paid", "current"]  # include 'current' conservatively
    if any(t in status for t in paid_tokens):
        # profit is interest (approx); optionally pass profit_if_paid to scale
        reward = loan_amnt * (int_rate / 100.0)
    else:
        # charged off / default -> negative principal
        reward = -loan_amnt
    if profit_if_paid is not None:
        reward = profit_if_paid if reward > 0 else reward
    return float(reward)

def prepare_split(df, feature_cols, out_npz):
    # Build arrays: for each row create two transitions (action 1 and action 0)
    obs = df[feature_cols].copy()
    # coerce numeric-like columns
    obs = obs.apply(pd.to_numeric, errors='coerce')
    med = obs.median()
    obs = obs.fillna(med)
    X = obs.values.astype(np.float32)
    N, D = X.shape

    actions = []
    rewards = []
    observations = []
    next_obs = []
    terminals = []

    for i in range(N):
        s = X[i]
        # action = 1 (approve) -> reward from history
        r1 = compute_reward(df.iloc[i])
        observations.append(s)
        next_obs.append(s)  # single-step
        actions.append(1)
        rewards.append(r1)
        terminals.append(1)

        # action = 0 (reject) -> reward = 0
        observations.append(s)
        next_obs.append(s)
        actions.append(0)
        rewards.append(0.0)
        terminals.append(1)

    observations = np.asarray(observations, dtype=np.float32)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.bool_)

    # simple reward scaling: standardize by std of abs(rewards) to keep magnitudes reasonable
    scale = max(1.0, np.std(np.abs(rewards)))
    rewards = rewards / scale

    np.savez_compressed(out_npz,
                        observations=observations,
                        next_observations=next_obs,
                        actions=actions,
                        rewards=rewards,
                        terminals=terminals,
                        feature_cols=np.array(feature_cols, dtype=object),
                        reward_scale=scale)
    return {
        "n_transitions": int(observations.shape[0]),
        "feature_dim": int(D),
        "reward_scale": float(scale)
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Processed CSV input (gzip)")
    p.add_argument("--out_dir", default="data/rl", help="output folder for RL dataset")
    p.add_argument("--val_frac", type=float, default=0.1, help="validation fraction split")
    p.add_argument("--sample_frac", type=float, default=1.0, help="random subsample fraction")
    p.add_argument("--max_rows", type=int, default=None, help="max rows to read for speed")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV:", args.input)
    df = pd.read_csv(args.input, compression='gzip', nrows=args.max_rows)
    # choose feature columns: all except y, loan_status, issue_d_parsed, id-like
    exclude = {"y", "loan_status", "issue_d_parsed", "id", "member_id", "url", "title", "desc"}
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"Detected {len(feature_cols)} feature columns for RL states.")

    # subsample if requested
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)

    # shuffle and split
    df = df.reset_index(drop=True)
    n = len(df)
    val_n = int(n * args.val_frac)
    if val_n < 1:
        val_n = max(1, int(0.1 * n))
    # simple deterministic split: last val_n rows as validation
    df_train = df.iloc[:-val_n].reset_index(drop=True)
    df_val = df.iloc[-val_n:].reset_index(drop=True)

    print("Preparing TRAIN RL dataset (this will create approve/reject transitions)...")
    meta_train = prepare_split(df_train, feature_cols, out_dir / "train_rl.npz")
    print("Prepared train_rl.npz ->", meta_train)

    print("Preparing VAL RL dataset...")
    meta_val = prepare_split(df_val, feature_cols, out_dir / "val_rl.npz")
    print("Prepared val_rl.npz ->", meta_val)

    # save some metadata
    info = {
        "input_rows": int(n),
        "train_transitions": meta_train,
        "val_transitions": meta_val,
        "feature_cols": feature_cols
    }
    with open(out_dir / "rl_dataset_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print("Saved RL dataset info to", out_dir / "rl_dataset_info.json")
    print("Done.")

if __name__ == "__main__":
    main()
