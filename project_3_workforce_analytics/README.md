# Project 3: Workforce Analytics Study

## Executive Summary
A statistically rigorous, governance-aware HR analytics pipeline examining employee attrition across hiring cohorts, compensation bands, overtime workload strain, and performance ratings[cite: 8].

## Key Findings & Recommendations
1. **Overtime Burnout Catalyst:** Employees working overtime churn at roughly 3x the rate of non-overtime peers ($p < 0.0001$).
   * *Action:* Implement automated workload alerts when employees exceed 3 consecutive weeks of overtime in high-attrition departments (Sales & Lab Tech roles).
2. **First-Year Hiring Cliff (`0-1y` Cohort):** New hires exhibit disproportionately elevated attrition compared to core (`3-7y`) staff.
   * *Action:* Restructure 30-60-90 day onboarding milestones and assign peer mentors during the first 6 months.
3. **High-Performer Retention Risk:** Outstanding performers (`PerformanceRating = 4`) with low job satisfaction or high promotion stagnation show sharp departure risk.
   * *Action:* Audit compensation bands and career-path reviews for top-rated contributors before annual review cycles.

## Responsible-Use, Privacy & Limitations
* **Privacy & Anonymization:** Raw `EmployeeNumber` identifiers are hashed via SHA-256 (`employee_hash`) prior to analytical persistence[cite: 8].
* **Small-Sample Governance:** Automated Chi-Square guards flag any subgroup with $n < 10$ to prevent statistical over-interpretation[cite: 8].
* **Limitations:** Performance ratings in the dataset exhibit low variance (only levels 3 and 4), limiting granularity on underperformance vs. burnout; exit-interview qualitative context is absent[cite: 8].

## How to Reproduce
```bash
PYTHONPATH=. pytest -v tests/
python main.py
