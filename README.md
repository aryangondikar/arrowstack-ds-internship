# Arrowstack Data Science 1-Month Internship Portfolio

Production-grade, reproducible data science pipelines built with strict schema validation, automated unit testing (`pytest`), statistical verification, and high-resolution stakeholder reporting[cite: 1, 3, 6, 9].

## Projects Included
1. **[Project 1: Customer Churn Investigation](./project_1_customer_churn/)** — Subscription cohort segmentation, edge-case cleaning, and calibrated churn driver modeling (Test ROC-AUC: `0.843`)[cite: 2].
2. **[Project 2: Retail Demand & Basket Analysis](./project_2_retail_basket/)** — Historical transaction hygiene, temporal demand heatmaps, and FP-Growth market basket association rule mining[cite: 5].
3. **[Project 3: Workforce Analytics Study](./project_3_workforce_analytics/)** — Statistical hypothesis testing (Chi-Square & Mann-Whitney U), PII hashing, and small-sample governance guardrails for HR attrition & productivity[cite: 8].

## Repository Architecture
Each project is self-contained with:
* `docs/requirements.md` — Stakeholder framing, scope, and acceptance criteria[cite: 2, 5, 8]
* `src/` — Modular ingestion, validation, feature engineering, and modeling scripts[cite: 2, 5, 8]
* `tests/` — Automated `pytest` edge-case & governance test suites[cite: 2, 5, 8]
* `reports/` — Automated data dictionaries, JSON validation metrics, and 300 DPI figures[cite: 3, 6, 9]
* `main.py` — Single-command reproducible pipeline execution[cite: 3, 6, 9]
