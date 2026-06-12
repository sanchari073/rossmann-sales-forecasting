# Retail Sales Demand Forecasting
**Dataset:** [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) (Kaggle) — 1,017,209 rows, 1,115 stores, Jan 2013–Jul 2015

---

## What this project does

Forecasts daily store-level sales using time-series feature engineering, XGBoost, and ARIMA on the Rossmann Store Sales dataset. The project is split into two layers:

**`pipeline/`** — PySpark data ingestion and feature engineering, designed to run on Databricks or any Spark 3.x cluster. Includes automated data quality checks that enforce row-level constraints before features are computed. A pandas version (`data_quality.py`) runs locally without Spark for quick testing.

**`notebooks/`** — Modeling and analysis in a Colab notebook. Covers SQL-based EDA using pandasql, XGBoost with time-series cross-validation, ARIMA on a single store, KMeans store segmentation, promo ROI analysis, and seasonality decomposition.

---

## Results

| Model | MAPE | RMSE | vs. Baseline |
|-------|------|------|--------------|
| Rolling mean (baseline) | 23.43% | 2,063 | — |
| ARIMA (Store 1) | 14.19% | — | — |
| XGBoost | **10.34%** | **961** | **−56%** |

Cross-validation (expanding window): 11.43% on 2013→2014 fold, 10.50% on 2013+2014→2015 fold.

---

## Repo structure

```
rossmann-sales-forecasting/
├── pipeline/
│   ├── pyspark_pipeline.py     # Spark pipeline: DQ checks + feature engineering
│   └── data_quality.py         # Pandas DQ for local testing (no Spark needed)
├── notebooks/
│   └── Retail_Sales_Demand_Forecasting.ipynb
├── data/
│   └── (place train.csv and store.csv here — not committed, see below)
└── README.md
```

---

## Data setup

Download `train.csv` and `store.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/rossmann-store-sales/data).

To run the notebook: open in Google Colab, run the upload cell, and select both files when the file picker opens.

To run the pipeline locally: place both files in `data/` and follow the commands below.

---

## Running the pipeline

**Local DQ check (no Spark required):**
```bash
python pipeline/data_quality.py --train data/train.csv --store data/store.csv
```

**PySpark feature pipeline (Databricks or local Spark):**
```bash
spark-submit pipeline/pyspark_pipeline.py \
    --input_path ./data \
    --output_path ./outputs
```

On Databricks, update paths to DBFS:
```bash
--input_path /dbfs/FileStore/rossmann
--output_path /dbfs/FileStore/rossmann/features
```

---

## Features engineered (30 total)

**Calendar:** Year, Month, Day, WeekOfYear, Quarter, IsWeekend, IsMonthStart, IsMonthEnd, IsDecember, IsSummer

**Lag variables:** Sales_lag_1, Sales_lag_7, Sales_lag_14, Sales_lag_28

**Rolling statistics:** Sales_roll_mean_7, Sales_roll_mean_14, Sales_roll_mean_28, Sales_roll_std_7

**Store context:** StoreType, Assortment, CompetitionDistance, CompetitionOpenMonths, SalesPerCustomer

**Promo:** Promo, Promo2, IsPromo2Active, PromoInterval

**Holiday:** StateHoliday, SchoolHoliday

---

## Data quality checks (pipeline)

9 checks enforced before features are computed. Failures above 2% threshold halt the pipeline:

- Sales must be non-negative
- Closed stores must have zero sales
- No nulls in Store, Date, Sales, Open, DayOfWeek
- Date must fall in expected range (2012–2016)
- Store IDs must be in [1, 1115]
- DayOfWeek must be in [1, 7]

---

## Key findings

- XGBoost improved on the rolling mean baseline by 56% on MAPE (10.34% vs 23.43%)
- Top features: Sales_roll_mean_14 (~0.28) and Sales_roll_mean_28 (~0.21) dominated, followed by Promo (~0.16) and Sales_lag_1 (~0.13)
- Promo lift varies by store type: Type 0 sees 43% lift, Type 1 only 18.2% despite having the highest baseline sales (~9,500)
- December seasonal index is 123.2 — the only month clearly above average. Sep (93.7) and Oct (94.5) are the weakest
- 742 of 1,115 stores (66%) fall into the Promo Dependent segment with only 2.2% organic YoY growth
- High Volume stores (32 stores, avg $8,400 daily sales) had the worst forecast accuracy at 12.18% MAPE
- Store 817 has the highest avg daily sales at 21,757 — more than 3x the network average

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data ingestion & DQ | PySpark, pandas |
| Feature engineering | PySpark (window functions), pandas |
| SQL EDA | pandasql |
| Modeling | XGBoost, ARIMA (statsmodels), scikit-learn |
| Segmentation | KMeans, StandardScaler |
| Analysis | pandas, matplotlib, seaborn |
| Cluster | Databricks-compatible; runs on any Spark 3.x environment |
