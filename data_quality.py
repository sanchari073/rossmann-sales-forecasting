"""
data_quality.py — pandas-based DQ checks for local dev / CI testing.
Mirrors the PySpark checks in pyspark_pipeline.py so you can validate
the dataset before spinning up a cluster.

Usage:
    python data_quality.py --train ./data/train.csv --store ./data/store.csv
"""

import argparse
import sys
import pandas as pd


CHECKS_PASSED = []
CHECKS_FAILED = []


def check(label, condition_series, threshold=0.02):
    total = len(condition_series)
    passed = int(condition_series.sum())
    fail_rate = (total - passed) / total if total > 0 else 0
    status = "PASS" if fail_rate <= threshold else "FAIL"
    result = f"[DQ] {status} | {label} | {total - passed:,} bad rows / {total:,} ({fail_rate:.2%})"
    print(result)
    if status == "PASS":
        CHECKS_PASSED.append(label)
    else:
        CHECKS_FAILED.append(label)


def run(train_path, store_path):
    print("Loading data...\n")
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)
    df = train.merge(store, on="Store", how="left")

    print(f"Rows: {len(df):,} | Stores: {df['Store'].nunique():,}\n")
    print("Running data quality checks...")
    print("-" * 60)

    check("Sales >= 0",                 df["Sales"] >= 0)
    check("Closed store => Sales=0",    ~((df["Open"] == 0) & (df["Sales"] > 0)))
    check("No null Sales",              df["Sales"].notna())
    check("No null Store",              df["Store"].notna())
    check("No null Date",               df["Date"].notna())
    check("No null Open",               df["Open"].notna())
    check("Date in valid range",        df["Date"].dt.year.between(2012, 2016))
    check("Store ID in [1, 1115]",      df["Store"].between(1, 1115))
    check("DayOfWeek in [1, 7]",        df["DayOfWeek"].between(1, 7))
    check("Open is binary",             df["Open"].isin([0, 1]))

    print("-" * 60)
    print(f"\nSummary: {len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed")

    if CHECKS_FAILED:
        print(f"FAILED checks: {CHECKS_FAILED}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./data/train.csv")
    parser.add_argument("--store", default="./data/store.csv")
    args = parser.parse_args()
    run(args.train, args.store)
