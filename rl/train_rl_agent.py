#!/usr/bin/env python3
"""
rl/train_rl_agent.py (robust + fallback)

Tries to train CQL using d3rlpy (many version signatures supported).
If any step fails, falls back to training a fast LogisticRegression policy
on the same dataset (predicts approve when expected reward > 0).

Outputs:
 - models/rl/rl_policy.pkl  (joblib-wrapped policy with predict(X))
 - if CQL succeeded: models/rl/cql_model (d3rlpy model saved), train_summary.json
 - if fallback used: models/rl/logreg_policy.pkl and train_summary.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import joblib
import os
import sys
import traceback
import warnings

# Suppress gym deprecation spam
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Optional imports
try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.algos import CQL
    HAS_D3RLPY = True
except Exception:
    d3rlpy = None
    MDPDataset = None
    CQL = None
    HAS_D3RLPY = False

# sklearn fallback imports (guarantee fast fallback)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    obs = data["observations"]
    next_obs = data.get("next_observations", None)
    actions = data["actions"]
    rewards = data["rewards"]
    terminals = data["terminals"]
    feature_cols = data.get("feature_cols", None)
    reward_scale = float(data.get("reward_scale", 1.0))
    return obs, actions, rewards, terminals, next_obs, feature_cols, reward_scale

def build_mdp_dataset_flexible(MDPDataset, obs, actions, rewards, terminals, next_obs=None):
    """
    Robust builder for d3rlpy.MDPDataset. Avoids risky positional-with-next_obs calls.
    Returns MDPDataset instance or raises.
    """
    obs = np.asarray(obs)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    terminals = np.asarray(terminals)
    n = obs.shape[0]
    if actions.shape[0] != n or rewards.shape[0] != n or terminals.shape[0] != n:
        raise ValueError("Shape mismatch in RL arrays")

    # 1) Named args including next_observations (preferred)
    if next_obs is not None:
        try:
            ds = MDPDataset(observations=obs, actions=actions, rewards=rewards,
                            terminals=terminals, next_observations=next_obs)
            print("[INFO] Built MDPDataset with named next_observations.")
            return ds
        except Exception as e:
            print("[DEBUG] Named with next_observations failed:", e)

    # 2) Named args without next_observations
    try:
        ds = MDPDataset(observations=obs, actions=actions, rewards=rewards, terminals=terminals)
        print("[INFO] Built MDPDataset with named args (no next_observations).")
        return ds
    except Exception as e:
        print("[DEBUG] Named without next_observations failed:", e)

    # 3) Positional args without next_obs
    try:
        ds = MDPDataset(obs, actions, rewards, terminals)
        print("[INFO] Built MDPDataset with positional args (no next_observations).")
        return ds
    except Exception as e:
        print("[DEBUG] Positional without next_observations failed:", e)

    # 4) from_numpy if available
    try:
        if hasattr(MDPDataset, "from_numpy"):
            ds = MDPDataset.from_numpy(observations=obs, actions=actions, rewards=rewards, terminals=terminals, next_observations=next_obs)
            print("[INFO] Built MDPDataset using from_numpy.")
            return ds
    except Exception as e:
        print("[DEBUG] from_numpy attempt failed:", e)

    raise RuntimeError("Could not construct MDPDataset for your d3rlpy version")

def train_cql(dataset_tr, dataset_val, out_dir: Path, n_epochs=50, batch_size=256, seed=42):
    # Try several constructor signatures for CQL; if all fail, raise exception
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    algo = None
    tried = []
    # Attempt 1: modern signature with many kwargs
    try:
        algo = CQL(q_func_layers=(256,256), batch_size=batch_size, n_critics=2, alpha=5.0, use_gpu=use_gpu)
        print("[INFO] Initialized CQL with common kwargs")
        tried.append("modern")
    except Exception as e:
        print("[DEBUG] CQL modern init failed:", e)
    # Attempt 2: simpler constructor
    if algo is None:
        try:
            algo = CQL()
            print("[INFO] Initialized CQL() no-args")
            tried.append("no_args")
        except Exception as e:
            print("[DEBUG] CQL() no-args failed:", e)
    # Attempt 3: try to pass minimal config if LearnableBase signature differs
    if algo is None:
        try:
            # Some d3rlpy versions expect device/config positional args -> try safe minimal
            algo = CQL(None)
            print("[INFO] Initialized CQL with single None arg")
            tried.append("single_none")
        except Exception as e:
            print("[DEBUG] CQL(None) failed:", e)

    if algo is None:
        raise RuntimeError("CQL initialization failed for all tried signatures: " + ",".join(tried))

    print("[INFO] Starting CQL.fit; use_gpu=", use_gpu)
    history = algo.fit(dataset_tr, eval_episodes=dataset_val, n_epochs=n_epochs, n_steps_per_epoch=1000, verbose=True)
    # Save model
    model_path = out_dir / "cql_model"
    try:
        algo.save_model(str(model_path))
        print("[INFO] CQL model saved to", model_path)
    except Exception as e:
        print("[WARN] Could not save CQL model via algo.save_model:", e)

    # Wrap policy
    class PolicyWrapper:
        def __init__(self, algo):
            self.algo = algo
        def predict(self, obs: np.ndarray):
            act = self.algo.predict(obs)
            act = np.asarray(act)
            if act.ndim > 1 and act.shape[1] > 1:
                return act.argmax(axis=1)
            return act.ravel()

    policy_wrapper = PolicyWrapper(algo)
    joblib.dump(policy_wrapper, out_dir / "rl_policy.pkl")
    # Save summary
    summary = {
        "mode": "cql",
        "n_epochs": n_epochs,
        "n_train": int(dataset_tr.observations.shape[0]) if hasattr(dataset_tr, "observations") else int(len(dataset_tr))
    }
    with open(out_dir / "train_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return True

def fallback_train_logreg(obs, actions, rewards, terminals, out_dir: Path):
    """
    Fast fallback: train a LogisticRegression classifier to predict whether approve yields positive reward.
    We train on only the approve (action==1) rows: label = (reward>0).

    The sklearn classifier object is saved directly (joblib.dump(clf,...)) so it can be loaded
    later by rl/evaluate_rl_policy.py which calls policy.predict(X).
    """
    # Ensure arrays
    obs = np.asarray(obs)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)

    # Select rows where action == 1 (approve transitions), these correspond to observed outcomes
    idx = (actions == 1)
    if idx.sum() == 0:
        # If no approve rows, fall back to training on all rows where reward != 0
        idx = (rewards != 0)
    X = obs[idx]
    y = (rewards[idx] > 0).astype(int)  # approve desired if reward positive

    if len(y) < 10:
        raise RuntimeError("Not enough data to train fallback policy (need >=10 samples)")

    # Train logistic regression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(X, y)

    # Save sklearn classifier itself (picklable) as the policy artifact.
    joblib.dump(clf, out_dir / "rl_policy.pkl")
    print("[INFO] Saved sklearn LogisticRegression classifier to", out_dir / "rl_policy.pkl")

    # Save model metrics on a small holdout (if possible)
    try:
        if len(X) > 10:
            split = int(0.8 * len(X))
            Xtr, Xte = X[:split], X[split:]
            ytr, yte = y[:split], y[split:]
            preds = clf.predict(Xte)
            metrics = {
                "mode": "logreg_fallback",
                "train_samples": int(len(X)),
                "accuracy": float(accuracy_score(yte, preds)) if len(yte) > 0 else None,
                "f1": float(f1_score(yte, preds)) if len(yte) > 0 else None,
                "precision": float(precision_score(yte, preds)) if len(yte) > 0 else None,
                "recall": float(recall_score(yte, preds)) if len(yte) > 0 else None
            }
        else:
            metrics = {"mode": "logreg_fallback", "train_samples": int(len(X))}
    except Exception:
        metrics = {"mode": "logreg_fallback", "train_samples": int(len(X))}

    with open(out_dir / "train_summary.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="train npz from prepare_rl_dataset.py")
    p.add_argument("--val", required=False, help="val npz")
    p.add_argument("--out", default="models/rl", help="output dir")
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading train npz:", args.train)
    obs_tr, actions_tr, rewards_tr, terminals_tr, next_obs_tr, fcols_tr, rscale = load_npz(args.train)
    dataset_tr = None
    dataset_val = None

    if HAS_D3RLPY and MDPDataset is not None:
        try:
            dataset_tr = build_mdp_dataset_flexible(MDPDataset, obs_tr, actions_tr, rewards_tr, terminals_tr, next_obs_tr)
        except Exception as e:
            print("[WARN] Could not build train MDPDataset for d3rlpy:", e)
            dataset_tr = None

    if args.val:
        try:
            obs_val, actions_val, rewards_val, terminals_val, next_obs_val, fcols_val, rscale_val = load_npz(args.val)
            if HAS_D3RLPY and MDPDataset is not None and dataset_tr is not None:
                try:
                    dataset_val = build_mdp_dataset_flexible(MDPDataset, obs_val, actions_val, rewards_val, terminals_val, next_obs_val)
                except Exception as e:
                    print("[WARN] Could not build val MDPDataset:", e)
                    dataset_val = None
        except Exception as e:
            print("[WARN] Could not load val npz:", e)
            dataset_val = None

    # Try CQL if available and dataset built
    if HAS_D3RLPY and CQL is not None and dataset_tr is not None:
        try:
            print("[INFO] Attempting CQL training (this may take a while)...")
            try:
                train_cql(dataset_tr, dataset_val, out_dir, n_epochs=args.n_epochs, batch_size=args.batch_size)
                print("[INFO] CQL training completed and policy saved.")
                return
            except Exception as e:
                print("[ERROR] CQL training failed:", e)
                traceback.print_exc()
                # fall through to fallback
        except Exception as e:
            print("[ERROR] Unexpected CQL path failure:", e)
            traceback.print_exc()

    # Fallback path: train LogisticRegression policy
    try:
        print("[WARN] Falling back to fast LogisticRegression policy (no d3rlpy required).")
        fallback_train_logreg(obs_tr, actions_tr, rewards_tr, terminals_tr, out_dir)
        print("[INFO] Fallback policy trained and saved.")
        return
    except Exception as e:
        print("[ERROR] Fallback training also failed:", e)
        traceback.print_exc()
        sys.exit(10)

if __name__ == "__main__":
    main()
