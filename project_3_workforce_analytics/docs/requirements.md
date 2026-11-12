# Project 3: Workforce Analytics Study — Requirements & Framing

## 1. Stakeholder & Business Use Case
* **Primary Stakeholder:** VP of People Operations & HR Strategy.
* **Decision Enabled:** Identify structural drivers of voluntary employee attrition, evaluate onboarding/hiring tenure bottlenecks, and assess workload-to-productivity balance without relying on biased or under-sampled demographic slices.
* **Measurable Benefit:** Reduce regrettable annual attrition by 15% in high-turnover roles and improve early-tenure (first 12 months) retention.

## 2. Scope, Governance & Success Criteria
* **Inputs:** Historical HR employee records (demographics, department, job role, compensation, overtime, satisfaction scores, performance rating, tenure, `Attrition`).
* **Responsible-Use & Privacy Guardrails:**
  * Strip or hash direct employee identifiers (`EmployeeNumber`) in processed analytical exports.
  * Enforce minimum subgroup sample thresholds ($n \ge 10$) before reporting statistical claims to prevent biased HR interventions.
* **Outputs:**
  1. Automated Data Dictionary (`reports/data_dictionary.csv`)
  2. Anonymized & validated HR dataset (`data/processed/workforce_cleaned.csv`)
  3. Statistical hypothesis test & predictive driver report (`reports/workforce_analytics_report.json`)
  4. High-resolution executive visualizations (`reports/figures/*.png`)
* **Success Metrics:**
  * Zero constant/zero-variance columns or unvalidated outlier records in processed data.
  * Statistically validated attrition drivers ($p < 0.05$) with documented effect sizes and governance warnings.
  * 100% pass rate on automated unit & governance tests.
