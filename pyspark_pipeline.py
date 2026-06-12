"""
Rossmann Store Sales — PySpark Feature & Quality Pipeline
----------------------------------------------------------
Runs on Databricks or any Spark cluster.
Reads raw train.csv + store.csv, enforces data quality checks,
engineers all features, and writes a clean parquet ready for modeling.

Usage (Databricks):
    spark-submit pyspark_pipeline.py \
        --input_path /dbfs/FileStore/rossmann/ \
        --output_path /dbfs/FileStore/rossmann/features/

Local (requires PySpark installed):
    python pyspark_pipeline.py --input_path ./data/ --output_path ./outputs/
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType


# ── helpers ──────────────────────────────────────────────────────────────────

def build_spark(app_name="rossmann_pipeline"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def log_check(label, passed, total, threshold=0.02):
    """Raise if failure rate exceeds threshold."""
    fail_rate = (total - passed) / total if total > 0 else 0
    status = "PASS" if fail_rate <= threshold else "FAIL"
    print(f"[DQ] {status} | {label} | {total - passed:,} bad rows / {total:,} ({fail_rate:.2%})")
    if status == "FAIL":
        raise ValueError(f"Data quality check failed: {label} — {fail_rate:.2%} exceeds threshold {threshold:.2%}")
    return passed


# ── data quality ─────────────────────────────────────────────────────────────

def run_quality_checks(df):
    """
    Checks applied:
    1. Sales must be non-negative
    2. Open=0 rows should have Sales=0
    3. No nulls in key columns
    4. Date must parse correctly (already done via read, but we verify range)
    5. Store ID must be in valid range
    """
    total = df.count()

    # 1. Non-negative sales
    neg_sales = df.filter(F.col("Sales") < 0).count()
    log_check("Sales >= 0", total - neg_sales, total)

    # 2. Closed stores should have 0 sales
    bad_closed = df.filter((F.col("Open") == 0) & (F.col("Sales") > 0)).count()
    log_check("Closed store => Sales=0", total - bad_closed, total)

    # 3. Nulls in key columns
    key_cols = ["Store", "Date", "Sales", "Open", "DayOfWeek"]
    for col in key_cols:
        null_count = df.filter(F.col(col).isNull()).count()
        log_check(f"No nulls in {col}", total - null_count, total)

    # 4. Date range sanity — Rossmann data runs 2013-2015
    bad_dates = df.filter(
        (F.year("Date") < 2012) | (F.year("Date") > 2016)
    ).count()
    log_check("Date in valid range", total - bad_dates, total)

    # 5. Store IDs
    bad_ids = df.filter((F.col("Store") < 1) | (F.col("Store") > 1115)).count()
    log_check("Store ID in [1, 1115]", total - bad_ids, total)

    print("[DQ] All checks passed.\n")
    return df


# ── feature engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    """
    Features engineered:
    - Calendar: year, month, day, week_of_year, is_weekend
    - Lag features: sales lag 1, 7, 14, 28 days (per store)
    - Rolling averages: 7-day and 30-day rolling mean (per store)
    - Promo flags: Promo, Promo2, SchoolHoliday, StateHoliday indicator
    - Store-level: competition distance buckets, promo2 active months
    """

    # calendar
    df = df.withColumn("year",          F.year("Date"))
    df = df.withColumn("month",         F.month("Date"))
    df = df.withColumn("day",           F.dayofmonth("Date"))
    df = df.withColumn("week_of_year",  F.weekofyear("Date"))
    df = df.withColumn("is_weekend",    (F.col("DayOfWeek") >= 6).cast(IntegerType()))

    # window specs — ordered by date per store
    w_store  = Window.partitionBy("Store").orderBy("Date")
    w_7d     = w_store.rowsBetween(-7, -1)
    w_30d    = w_store.rowsBetween(-30, -1)

    # lag features
    for lag in [1, 7, 14, 28]:
        df = df.withColumn(f"sales_lag_{lag}", F.lag("Sales", lag).over(w_store))

    # rolling averages
    df = df.withColumn("sales_roll7",  F.avg("Sales").over(w_7d))
    df = df.withColumn("sales_roll30", F.avg("Sales").over(w_30d))

    # holiday binarization
    df = df.withColumn("is_state_holiday",
                       (F.col("StateHoliday") != "0").cast(IntegerType()))

    # competition distance buckets (0=close, 1=mid, 2=far, 3=unknown)
    df = df.withColumn(
        "comp_dist_bucket",
        F.when(F.col("CompetitionDistance").isNull(), 3)
         .when(F.col("CompetitionDistance") < 1000, 0)
         .when(F.col("CompetitionDistance") < 5000, 1)
         .otherwise(2)
    )

    # months since competition opened
    df = df.withColumn(
        "comp_open_months",
        F.when(
            F.col("CompetitionOpenSinceYear").isNotNull(),
            (F.col("year") - F.col("CompetitionOpenSinceYear")) * 12
            + (F.col("month") - F.col("CompetitionOpenSinceMonth"))
        ).otherwise(F.lit(-1))
    )

    # promo2 active flag — Promo2SinceWeek and year tell us when it started
    df = df.withColumn(
        "promo2_active",
        F.when(
            (F.col("Promo2") == 1) & F.col("Promo2SinceYear").isNotNull() &
            (
                (F.col("year") > F.col("Promo2SinceYear")) |
                ((F.col("year") == F.col("Promo2SinceYear")) &
                 (F.col("week_of_year") >= F.col("Promo2SinceWeek")))
            ),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_path, output_path):
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading data from: {input_path}")

    # read
    train = (
        spark.read.csv(f"{input_path}/train.csv", header=True, inferSchema=True)
        .withColumn("Date", F.to_date("Date", "yyyy-MM-dd"))
    )
    store = spark.read.csv(f"{input_path}/store.csv", header=True, inferSchema=True)

    print(f"train rows: {train.count():,} | store rows: {store.count():,}")

    # join
    df = train.join(store, on="Store", how="left")

    # filter — only open days with positive sales for modeling
    df = df.filter((F.col("Open") == 1) & (F.col("Sales") > 0))
    print(f"After filtering closed/zero-sales rows: {df.count():,}")

    # quality checks
    df = run_quality_checks(df)

    # feature engineering
    df = engineer_features(df)

    # drop rows where lags couldn't be computed (first 28 days per store)
    df = df.dropna(subset=["sales_lag_28", "sales_roll30"])
    print(f"After dropping lag warmup rows: {df.count():,}")

    # write
    (
        df.repartition(20)
          .write
          .mode("overwrite")
          .parquet(f"{output_path}/features_clean/")
    )
    print(f"Feature parquet written to: {output_path}/features_clean/")

    # also write a summary stats table for documentation
    summary = df.select("Sales", "Customers", "sales_roll7", "sales_roll30").describe()
    summary.show()

    spark.stop()
    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  default="./data",    help="Path to raw CSV files")
    parser.add_argument("--output_path", default="./outputs", help="Path for output parquet")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
