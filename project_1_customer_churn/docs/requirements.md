# Project 1: Customer Churn Investigation — Requirements & Framing

## 1. Stakeholder & Business Use Case
* **Primary Stakeholder:** VP of Customer Retention & Subscription Operations.
* **Decision Enabled:** Prioritize high-risk customer cohorts for targeted retention campaigns (contract upgrades, onboarding check-ins, and billing discounts) before cancellation occurs.
* **Measurable Benefit:** Reduce monthly customer churn rate by 10–15% within high-risk segments while minimizing unnecessary discount spend on loyal subscribers.

## 2. Scope & Success Criteria
* **Inputs:** Historical subscription customer dataset (`customerID`, demographics, account tenure, subscribed services, billing details, `Churn` label).
* **Outputs:**
  1. Automated Data Dictionary (`reports/data_dictionary.csv`)
  2. Cleaned & validated dataset (`data/processed/churn_cleaned.csv`)
  3. High-resolution EDA & driver visualizations (`reports/figures/*.png`)
  4. Validated churn driver model metrics (`reports/model_metrics.json`)
* **Success Metrics:**
  * Zero unhandled missing values or corrupted schema fields in processed output.
  * Predictive driver model achieves **ROC-AUC ≥ 0.80** and **Recall ≥ 0.75** on the positive churn class.
  * 100% pass rate on automated edge-case unit tests.
