# Project 1: Customer Churn Investigation

## Executive Summary
An end-to-end, reproducible churn analytics pipeline designed for Subscription Operations stakeholders to identify churn drivers, segment high-risk cohorts, and prioritize retention interventions.

## Repository Structure
```text
project_1_customer_churn/
├── data/
│   ├── raw/                 # Raw source CSV (gitignored)
│   └── processed/           # Validated & feature-engineered dataset
├── docs/
│   └── requirements.md      # Business framing & acceptance criteria
├── reports/
│   ├── figures/             # 300 DPI publication-ready EDA & driver plots
│   ├── data_dictionary.csv  # Automated schema & null audit
│   └── model_metrics.json   # Cross-validation & test evaluation metrics
├── src/
│   ├── fetch_data.py        # Reproducible data ingestion
│   ├── pipeline.py          # Schema checks, edge-case cleaning & features
│   └── analysis.py          # EDA visualizations & predictive modeling
├── tests/
│   └── test_churn.py        # Pytest validation suite
└── main.py                  # Single-command pipeline entry point
