Policy Optimization for Decision Making
Supervised Deep Learning + Offline Reinforcement Learning (CQL)
📌 Project Overview

This repository implements an end-to-end decision optimization system combining:

1. A Supervised Deep Learning model (MLP)

Predicts loan repayment probability using borrower-level features.

2. An Offline Reinforcement Learning Policy (CQL / fallback)

Learns an approval strategy optimized for expected financial reward, trained only on historical logged data — no online interaction required.

Goal:

Evaluate which approach—ML or RL—leads to better real-world decision performance, not just predictive accuracy.

🚀 Key Features

Clean preprocessing pipeline with reproducible feature engineering

Deep Learning classifier with threshold tuning

Offline RL dataset construction (MDP transitions)

Conservative Q-Learning (CQL) with robust fallbacks

Complete DL vs RL policy comparison

Automatic report generation (JSON, Markdown, LaTeX option)

Fully scripted, reproducible pipeline



🧠 Data Pipeline Overview
1. Preprocessing

Cleans numeric/categorical fields

Removes post-outcome leakage features

Imputes missing values

Scales numeric columns

Saves a reusable preprocessing object (preprocessor_dl.joblib)

2. Supervised Model (MLP)

Two-layer deep neural network

Class-weighted BCE loss

Early stopping

Outputs probabilities → threshold tuned on validation set

3. Offline RL (CQL)

RL dataset built from logged historical decisions

Transition tuples: (state, action, reward, next_state, terminal)

Reward function based on financial outcomes:

repayment → + loan_amnt * int_rate  
default   → – loan_amnt  
reject    → 0  


CQL attempted first; if unavailable, fallback logistic regression policy trained

4. Policy Comparison

Both DL and RL policies are evaluated using:

Metric	Description
AUC	Prediction discrimination (DL only)
F1 / Precision / Recall	Classification quality
Expected Reward	Business value of decisions
📊 Results Summary
Model	AUC	F1	Avg Reward	Conclusion
Deep Learning (MLP)	~0.578	0.25	–0.04	Higher accuracy, poor reward
RL Policy (CQL/Fallback)	N/A	0.17	+0.036	Lower accuracy, best reward
🔥 Insight:

The RL policy outperforms ML on business value, which is the real objective of decision-making systems.

⚙️ How to Run the Entire Pipeline

Below are the exact commands used in workflow order.

1️⃣ Preprocess Data
python scripts/data_prep.py --input data/raw/loan_data.csv --out data/processed

2️⃣ Train Deep Learning Model
python scripts/train_supervised.py \
  --processed_dir data/processed \
  --artifacts_dir artifacts \
  --out_dir models/dl

3️⃣ Build RL Dataset
python scripts/prepare_rl_dataset.py \
  --processed_dir data/processed \
  --out data/rl

4️⃣ Train RL Agent (CQL or Fallback)
python scripts/train_rl_agent.py \
  --train data/rl/train_rl.npz \
  --val data/rl/val_rl.npz \
  --out models/rl \
  --n_epochs 50

5️⃣ Compare DL vs RL Policies
python scripts/compare_policies.py \
  --test data/processed/test_processed.csv.gz \
  --val data/processed/val_processed.csv.gz \
  --dl_model models/dl/dl_model.pt \
  --dl_preproc models/preprocessor_dl.joblib \
  --rl_policy models/rl/rl_policy.pkl \
  --out reports/policy_comparison.json

6️⃣ Generate Final Report
python scripts/final_report.py \
  --test data/processed/test_processed.csv.gz \
  --val data/processed/val_processed.csv.gz \
  --dl_model models/dl/dl_model.pt \
  --dl_preproc models/preprocessor_dl.joblib \
  --rl_policy models/rl/rl_policy.pkl \
  --comparison reports/policy_comparison.json \
  --out reports

📦 Requirements
python >= 3.9
numpy
pandas
torch
scikit-learn
joblib
matplotlib
d3rlpy (optional; fallback enabled)


If d3rlpy fails to install (common on Windows), the RL script automatically trains a logistic regression fallback.

📝 Key Takeaways

Classification accuracy ≠ optimal decisions

RL optimizes long-term reward, not prediction precision

Offline RL enables safe policy learning from historical logs

Threshold tuning significantly impacts DL policy reward

Combining DL probability models with RL decision logic is powerful

📚 Deliverables

This project includes:

Training scripts

Cleaned data artifacts

Trained supervised + RL models

Evaluation metrics

Final Markdown + LaTeX reports

Ready-to-run reproducible pipeline

🙋 Author

Harsh Vardhan
Policy Optimization for Decision Making — 2025
