import json
import logging
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def generate_demand_charts(df: pd.DataFrame, figures_dir: Path) -> None:
    """Creates 300 DPI demand trend, peak shopping hour, and top revenue SKU charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Monthly Revenue & Demand Trend
    plt.figure(figsize=(10, 5))
    monthly = df.groupby("YearMonth")["TotalRevenue"].sum().reset_index()
    sns.lineplot(data=monthly, x="YearMonth", y="TotalRevenue", marker="o", linewidth=2.5)
    plt.title("Monthly Retail Revenue Trend", fontsize=13, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "01_monthly_revenue_trend.png", dpi=300)
    plt.close()

    # Chart 2: Top 10 Products by Revenue
    plt.figure(figsize=(10, 6))
    top_skus = (
        df.groupby("Description")["TotalRevenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    sns.barplot(data=top_skus, x="TotalRevenue", y="Description", hue="Description", palette="viridis", legend=False)
    plt.title("Top 10 Products by Total Revenue", fontsize=13, fontweight="bold")
    plt.xlabel("Total Revenue ($)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_top_10_products_revenue.png", dpi=300)
    plt.close()

    # Chart 3: Order Volume by Day of Week & Hour Heatmap
    plt.figure(figsize=(10, 5))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        df.groupby(["DayOfWeek", "HourOfDay"])["InvoiceNo"]
        .nunique()
        .unstack(fill_value=0)
        .reindex([d for d in day_order if d in df["DayOfWeek"].unique()])
    )
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3)
    plt.title("Unique Order Volume Heatmap (Day of Week vs. Hour)", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (24h)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_order_volume_heatmap.png", dpi=300)
    plt.close()

    logger.info(f"Saved 3 demand analysis figures to {figures_dir}")


def run_basket_association_analysis(
    df: pd.DataFrame,
    rules_output_path: Path,
    metrics_output_path: Path,
    figures_dir: Path,
    country_filter: str = "United Kingdom",
    min_support: float = 0.015,
    min_confidence: float = 0.40,
    min_lift: float = 1.50
) -> Dict[str, Any]:
    """Runs FP-Growth association rule mining and exports ranked rules & visuals."""
    # Scope to primary market (UK accounts for >90% of orders) to avoid regional distortion
    scoped_df = df[df["Country"] == country_filter] if country_filter in df["Country"].unique() else df

    # Filter out single-item invoices as they cannot form product associations
    invoice_sizes = scoped_df.groupby("InvoiceNo")["Description"].nunique()
    multi_item_invoices = invoice_sizes[invoice_sizes >= 2].index
    basket_df = scoped_df[scoped_df["InvoiceNo"].isin(multi_item_invoices)]

    basket_matrix = (
        basket_df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
        .gt(0)
    )

    logger.info(f"Running FP-Growth on {basket_matrix.shape[0]} multi-item baskets and {basket_matrix.shape[1]} SKUs...")
    frequent_itemsets = fpgrowth(basket_matrix, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found; lower min_support threshold.")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

    # Format frozensets to clean readable strings for CSV export
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(list(x))))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(list(x))))

    export_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"]
    rules_output_path.parent.mkdir(parents=True, exist_ok=True)
    rules[export_cols].round(4).to_csv(rules_output_path, index=False)

    # Chart 4: Support vs. Confidence Scatter sized/colored by Lift
    if not rules.empty:
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(
            rules["support"], rules["confidence"],
            c=rules["lift"], cmap="plasma", alpha=0.75, edgecolors="k", s=60
        )
        plt.colorbar(scatter, label="Lift")
        plt.title("Association Rules: Support vs. Confidence (Colored by Lift)", fontsize=12, fontweight="bold")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.tight_layout()
        plt.savefig(figures_dir / "04_association_rules_scatter.png", dpi=300)
        plt.close()

    summary_metrics = {
        "scoped_country": country_filter,
        "multi_item_baskets_analyzed": int(basket_matrix.shape[0]),
        "unique_skus_analyzed": int(basket_matrix.shape[1]),
        "frequent_itemsets_count": int(len(frequent_itemsets)),
        "qualified_rules_count": int(len(rules)),
        "max_lift": round(float(rules["lift"].max()), 4) if not rules.empty else 0.0,
        "top_5_rules": rules[export_cols].head(5).round(4).to_dict(orient="records")
    }

    with open(metrics_output_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    logger.info(f"Exported {len(rules)} association rules to {rules_output_path}")
    return summary_metrics
