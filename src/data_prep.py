# src/data_prep.py
"""
Data preparation utilities for LendingClub sample -> train/val/test + preprocessing artifact save.

Usage examples (run from project root):
python -c "from src.data_prep import main; main()"
or
python src/data_prep.py --sample_path "C:/.../sample_uniform_200k.csv.gz"

Outputs:
- data/processed/train.csv, val.csv, test.csv  (preprocessed numeric + categorical features + target + issue_d)
- artifacts/preprocessor.joblib   (fitted ColumnTransformer)
- artifacts/feature_lists.joblib  (dict with numeric/categorical/feature_columns)
"""

import os
import argparse
from typing import List, Tuple, Dict
import json

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


def load_dataframe(path: str, nrows: int | None = None) -> pd.DataFrame:
    compression = "gzip" if path.lower().endswith(".gz") else None
    df = pd.read_csv(path, compression=compression, nrows=nrows, low_memory=False)
    return df


def map_loan_status_to_binary(status: str) -> int:
    if pd.isna(status):
        return -1
    s = str(status).strip().lower()
    if s == "fully paid":
        return 0
    if s in ("charged off", "default"):
        return 1
    # ambiguous or in-flight labels
    return -1


def drop_post_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    post_markers = [
        "total_pymnt", "last_pymnt", "recoveries", "collection",
        "out_prncp", "total_rec", "last_credit", "recoveries", "collection_recovery"
    ]
    cols_to_drop = [c for c in df.columns if any(p in c.lower() for p in post_markers)]
    return df.drop(columns=cols_to_drop, errors="ignore")


def select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Return numeric_features, categorical_features, final_feature_columns.
    This uses conservative default lists (derived from typical LendingClub schema).
    Caller may override if needed.
    """
    # Conservative numeric features commonly available at application time
    default_numeric = [
        "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
        "dti", "fico_range_low", "fico_range_high", "revol_bal", "revol_util",
        "open_acc", "total_acc", "inq_last_6mths", "delinq_2yrs", "pub_rec"
    ]
    # Conservative categorical features
    default_categorical = [
        "term", "grade", "sub_grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "initial_list_status", "application_type", "addr_state"
    ]

    # Filter only present columns
    numeric = [c for c in default_numeric if c in df.columns]
    categorical = [c for c in default_categorical if c in df.columns]

    # Drop known high-cardinality or text fields (emp_title, zip_code, title, desc)
    high_cardinality_blacklist = {"emp_title", "zip_code", "title", "desc", "url", "member_id", "id"}
    categorical = [c for c in categorical if c not in high_cardinality_blacklist]

    # Final feature columns: numeric + categorical
    final_features = numeric + categorical

    # As a fallback: if final_features is empty, include any numeric cols automatically (safe fallback)
    if len(final_features) == 0:
        inferred_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        final_features = inferred_numeric
        numeric = inferred_numeric
        categorical = []

    return numeric, categorical, final_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Build OneHotEncoder in a backward/forward-compatible way
    from sklearn.preprocessing import OneHotEncoder as _OHE
    try:
        # older sklearn versions accept `sparse` argument
        cat_ohe = _OHE(handle_unknown="ignore", sparse=False)
    except TypeError:
        # newer sklearn versions use `sparse_output`
        cat_ohe = _OHE(handle_unknown="ignore", sparse_output=False)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", cat_ohe)
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor

def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__y__"] = df["loan_status"].apply(map_loan_status_to_binary)
    # Drop ambiguous labels (in-flight statuses)
    df = df[df["__y__"] != -1].reset_index(drop=True)
    return df


def add_issue_year_quarter(df: pd.DataFrame, date_col: str = "issue_d") -> pd.DataFrame:
    if date_col in df.columns:
        df["issue_d_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
        df["issue_year"] = df["issue_d_parsed"].dt.year
        df["issue_quarter"] = df["issue_d_parsed"].dt.to_period("Q").astype(str)
    else:
        df["issue_d_parsed"] = pd.NaT
        df["issue_year"] = -1
        df["issue_quarter"] = "NA"
    return df


def time_split(df: pd.DataFrame, date_col_parsed: str = "issue_d_parsed", train_end: str = "2016-12-31", val_end: str = "2017-12-31") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col_parsed] = pd.to_datetime(df[date_col_parsed], errors="coerce")
    train_mask = df[date_col_parsed] <= pd.to_datetime(train_end)
    val_mask = (df[date_col_parsed] > pd.to_datetime(train_end)) & (df[date_col_parsed] <= pd.to_datetime(val_end))
    test_mask = df[date_col_parsed] > pd.to_datetime(val_end)
    train = df[train_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)
    return train, val, test


def fit_and_transform(preprocessor: ColumnTransformer, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], numeric: List[str], categorical: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    # Fit on train only
    X_train_raw = train_df[feature_cols]
    X_val_raw = val_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Build column names for transformed dataframe
    out_cols = []
    if numeric:
        out_cols.extend(numeric)
    if categorical:
        # get categories from fitted OneHotEncoder
        cat_transformer = None
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                cat_transformer = trans.named_steps["ohe"]
                cat_cols = cols
                break
        if cat_transformer is not None:
            categories = cat_transformer.categories_
            for col, cats in zip(cat_cols, categories):
                out_cols.extend([f"{col}__{str(v)}" for v in cats])
    # Create DataFrames
    X_train_df = pd.DataFrame(X_train, columns=out_cols, index=train_df.index)
    X_val_df = pd.DataFrame(X_val, columns=out_cols, index=val_df.index)
    X_test_df = pd.DataFrame(X_test, columns=out_cols, index=test_df.index)

    # Append target and issue date info for traceability
    X_train_df["loan_status"] = train_df["loan_status"].values
    X_val_df["loan_status"] = val_df["loan_status"].values
    X_test_df["loan_status"] = test_df["loan_status"].values

    X_train_df["y"] = train_df["__y__"].values
    X_val_df["y"] = val_df["__y__"].values
    X_test_df["y"] = test_df["__y__"].values

    # Keep issue date for time-based analysis
    X_train_df["issue_d_parsed"] = train_df["issue_d_parsed"].values
    X_val_df["issue_d_parsed"] = val_df["issue_d_parsed"].values
    X_test_df["issue_d_parsed"] = test_df["issue_d_parsed"].values

    return X_train_df, X_val_df, X_test_df, preprocessor


def save_processed_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_processed.csv.gz")
    val_path = os.path.join(out_dir, "val_processed.csv.gz")
    test_path = os.path.join(out_dir, "test_processed.csv.gz")
    train_df.to_csv(train_path, index=False, compression="gzip")
    val_df.to_csv(val_path, index=False, compression="gzip")
    test_df.to_csv(test_path, index=False, compression="gzip")
    return train_path, val_path, test_path


def persist_artifacts(preprocessor: ColumnTransformer, numeric: List[str], categorical: List[str], feature_cols: List[str], artifact_dir: str = "artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)
    preproc_path = os.path.join(artifact_dir, "preprocessor.joblib")
    lists_path = os.path.join(artifact_dir, "feature_lists.joblib")
    joblib.dump(preprocessor, preproc_path)
    joblib.dump({"numeric": numeric, "categorical": categorical, "feature_cols": feature_cols}, lists_path)
    return preproc_path, lists_path


def main(
    sample_path: str = "data/sample_uniform_200k.csv.gz",
    nrows: int | None = None,
    train_end: str = "2016-12-31",
    val_end: str = "2017-12-31",
    out_processed_dir: str = "data/processed",
    artifacts_dir: str = "artifacts"
):
    print("[INFO] Loading sample:", sample_path)
    df = load_dataframe(sample_path, nrows=nrows)
    print("[INFO] Raw sample shape:", df.shape)

    # Drop post-outcome columns (prevent leakage)
    df = drop_post_outcome_columns(df)
    print("[INFO] After dropping post-outcome-like cols shape:", df.shape)

    # Label and filter
    df = filter_and_label(df)
    print("[INFO] After labeling and filtering ambiguous statuses shape:", df.shape)
    if df.empty:
        raise SystemExit("[ERROR] No labeled rows after filtering. Check sample labels.")

    # Add parsed issue date, year, quarter
    df = add_issue_year_quarter(df, date_col="issue_d")

    # Select features
    numeric, categorical, feature_cols = select_feature_columns(df)
    print(f"[INFO] Selected numeric ({len(numeric)}) and categorical ({len(categorical)}) features. Total feature cols: {len(feature_cols)}")

    # Time split
    train_df, val_df, test_df = time_split(df, date_col_parsed="issue_d_parsed", train_end=train_end, val_end=val_end)
    print("[INFO] Time split sizes: train=%d, val=%d, test=%d" % (len(train_df), len(val_df), len(test_df)))

    # If any split is empty, fallback to random split (stratified on y)
    if len(train_df) < 100 or len(val_df) < 100 or len(test_df) < 100:
        print("[WARN] One or more time splits too small. Falling back to random stratified split.")
        X = df[feature_cols]
        y = df["__y__"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        # rebuild train_df/val_df/test_df from X and y
        train_df = X_train.copy()
        train_df["__y__"] = y_train
        val_df = X_val.copy(); val_df["__y__"] = y_val
        test_df = X_test.copy(); test_df["__y__"] = y_test
        # ensure issue_d_parsed present
        train_df["issue_d_parsed"] = pd.NaT; val_df["issue_d_parsed"] = pd.NaT; test_df["issue_d_parsed"] = pd.NaT

    # Build preprocessor and fit/transform
    preprocessor = build_preprocessor(numeric, categorical)
    X_train_df, X_val_df, X_test_df, fitted_preprocessor = fit_and_transform(preprocessor, train_df, val_df, test_df, feature_cols, numeric, categorical)

    # Save processed datasets
    train_path, val_path, test_path = save_processed_datasets(X_train_df, X_val_df, X_test_df, out_dir=out_processed_dir)
    print("[INFO] Saved processed datasets:", train_path, val_path, test_path)

    # Persist artifacts
    preproc_path, lists_path = persist_artifacts(fitted_preprocessor, numeric, categorical, feature_cols, artifact_dir=artifacts_dir)
    print("[INFO] Saved artifacts:", preproc_path, lists_path)

    print("[DONE] Data preparation complete.")
    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "preproc_path": preproc_path,
        "lists_path": lists_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data prep for LendingClub sample")
    parser.add_argument("--sample_path", type=str, default="data/sample_uniform_200k.csv.gz", help="Path to sampled CSV")
    parser.add_argument("--nrows", type=int, default=None, help="Optional: only read N rows from sampled file for quick debug")
    parser.add_argument("--train_end", type=str, default="2016-12-31", help="Train split end date (inclusive)")
    parser.add_argument("--val_end", type=str, default="2017-12-31", help="Validation split end date (inclusive)")
    parser.add_argument("--out_processed_dir", type=str, default="data/processed", help="Directory to save processed CSVs")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save preprocessor and lists")
    args = parser.parse_args()
    main(
        sample_path=args.sample_path,
        nrows=args.nrows,
        train_end=args.train_end,
        val_end=args.val_end,
        out_processed_dir=args.out_processed_dir,
        artifacts_dir=args.artifacts_dir
    )
