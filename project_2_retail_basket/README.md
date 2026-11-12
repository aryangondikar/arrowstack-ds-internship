# Project 2: Retail Demand & Basket Analysis

## Executive Summary
An end-to-end retail demand and market basket association pipeline designed for Category Merchandising teams to uncover seasonal purchase spikes and high-lift cross-sell bundles.

## Methodology & Edge-Case Handling
* **Data Hygiene:** Strips cancelled invoices (`C` prefix), negative returns, zero-price entries, and non-merchandise accounting codes (`POST`, `DOT`, `M`, `BANK CHARGES`).
* **Regional Scoping:** Scopes association mining to the primary market (`United Kingdom`) and multi-item baskets (`>= 2` distinct SKUs) to prevent sparse single-item dilution.
* **Algorithm:** Uses **FP-Growth** (`min_support=0.015`, `min_confidence=0.40`, `min_lift=1.50`) for deterministic, high-lift bundle discovery.

## Limitations & Next Steps
* **Limitations:** Historical B2B/wholesale bulk buyers can skew raw item quantities; lack of product margin data means rules prioritize revenue/lift rather than net profit margin.
* **Next Steps:** Separate wholesale vs. retail customer segments via RFM clustering and run an online cart-recommendation A/B test on the top 10 product pairs.

## How to Reproduce
```bash
PYTHONPATH=. pytest -v tests/
python main.py
