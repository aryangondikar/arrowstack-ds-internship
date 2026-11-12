# Project 2: Retail Demand & Basket Analysis — Requirements & Framing

## 1. Stakeholder & Business Use Case
* **Primary Stakeholder:** Category Merchandising & Inventory Planning Lead.
* **Decision Enabled:** Optimize cross-selling product bundles, store/website placement, and seasonal inventory replenishment based on purchase frequency and basket co-occurrence.
* **Measurable Benefit:** Increase Average Order Value (AOV) by 8–12% through data-backed product bundling and reduce stockouts during peak demand windows.

## 2. Scope & Success Criteria
* **Inputs:** Historical e-commerce/retail invoice line items (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`).
* **Outputs:**
  1. Automated Data Dictionary (`reports/data_dictionary.csv`)
  2. Validated & cleaned transaction dataset (`data/processed/retail_cleaned.csv`)
  3. Demand seasonality & product association charts (`reports/figures/*.png`)
  4. Ranked Association Rules table (`reports/association_rules.csv`) & summary metrics (`reports/basket_metrics.json`)
* **Success Metrics:**
  * 100% removal of cancelled orders (`InvoiceNo` starting with `C`), non-positive quantities/prices, and non-product postage/fee codes.
  * Discovery of actionable association rules meeting **Support ≥ 0.015**, **Confidence ≥ 0.40**, and **Lift ≥ 1.50**.
  * 100% pass rate on automated edge-case unit tests.
