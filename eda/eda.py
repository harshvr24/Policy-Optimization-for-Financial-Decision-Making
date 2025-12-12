# eda/eda.py
import os
import sys
import pandas as pd

# ============================================
# CONFIG – Set this to your sampled dataset
# ============================================
SAMPLE_PATH = r"C:\Users\harsh\Downloads\shodh.ai\data\sample_uniform_200k.csv.gz"

NROWS = None   # load full sample; set e.g. 20_000 for quicker debugging
SAVE_QUICK = r"C:\Users\harsh\Downloads\shodh.ai\data\sample_quick_10k.csv.gz"

# ============================================
# LOAD SAMPLE
# ============================================
if not os.path.exists(SAMPLE_PATH):
    print("ERROR: Sample not found:", SAMPLE_PATH)
    print("Run sampling/sample_lc.py first.")
    sys.exit(1)

compression = 'gzip' if SAMPLE_PATH.lower().endswith('.gz') else None

print("\n[INFO] Loading sampled dataset...")
df = pd.read_csv(SAMPLE_PATH, compression=compression, nrows=NROWS, low_memory=False)
print("[INFO] Loaded:", df.shape, "columns:", len(df.columns))

# ============================================
# BASIC PREVIEW
# ============================================
print("\n=== HEAD (first 5 rows) ===")
print(df.head().to_string())

print("\n=== loan_status distribution ===")
print(df["loan_status"].value_counts(dropna=False).to_string())

# ============================================
# MISSINGNESS SUMMARY
# ============================================
missing_pct = df.isna().mean().sort_values(ascending=False)
print("\n=== Top 25 Missing Columns ===")
print(missing_pct.head(25).to_string())

# ============================================
# NUMERIC SUMMARY
# ============================================
numeric_cols = [
    'loan_amnt','funded_amnt','int_rate','installment','annual_inc','dti',
    'fico_range_low','fico_range_high','revol_bal','revol_util','open_acc','total_acc'
]
present_numeric = [c for c in numeric_cols if c in df.columns]

print("\n=== Key Numeric Feature Summary ===")
if present_numeric:
    print(df[present_numeric].describe().T.to_string())
else:
    print("No numeric columns found in sample.")

# ============================================
# ISSUE DATE SUMMARY
# ============================================
if "issue_d" in df.columns:
    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], errors="coerce")
    print("\n=== Issue Date Range ===")
    print(df["issue_d_parsed"].min(), "→", df["issue_d_parsed"].max())

    print("\n=== Loans by Year ===")
    print(df["issue_d_parsed"].dt.year.value_counts().sort_index().to_string())

# ============================================
# CATEGORICAL SUMMARIES
# ============================================
cat_cols = ['term','grade','sub_grade','home_ownership','verification_status','purpose','emp_length','addr_state']
present_cat = [c for c in cat_cols if c in df.columns]

for col in present_cat:
    print(f"\n=== Top values for {col} ===")
    print(df[col].value_counts(dropna=False).head(10).to_string())

# ============================================
# POST-OUTCOME FIELD DETECTION
# ============================================
post_markers = [
    "total_pymnt","last_pymnt","recoveries","collection",
    "out_prncp","total_rec","last_credit","recovery"
]

post_cols = [c for c in df.columns if any(p in c.lower() for p in post_markers)]

print("\n=== Detected Post-Outcome Columns (exclude from modeling) ===")
print(len(post_cols), "columns")
print(post_cols[:25])

# ============================================
# PRE-DECISION FEATURES (candidate model inputs)
# ============================================
exclude_cols = set(post_cols) | {"id","member_id","desc","url","title"}

candidate_features = [c for c in df.columns if c not in exclude_cols]

print("\n=== Number of Candidate Pre-Decision Features ===")
print(len(candidate_features))

print("\n=== Sample Candidate Features ===")
print(candidate_features[:60])

# ============================================
# SAVE 10K QUICK SAMPLE FOR LOCAL DEBUGGING
# ============================================
try:
    quick_n = min(10000, len(df))
    quick_df = df.sample(quick_n, random_state=42)
    quick_df.to_csv(SAVE_QUICK, index=False, compression="gzip")
    print(f"\n[SUCCESS] Saved {quick_n} rows quick sample → {SAVE_QUICK}")
except Exception as e:
    print("[WARNING] Could not save quick sample:", e)

print("\n[INFO] EDA Complete.")
