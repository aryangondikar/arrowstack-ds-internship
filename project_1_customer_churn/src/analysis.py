import json
import logging
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def generate_eda_charts(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generates business-focused EDA figures saved at 300 DPI."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Churn Rate by Contract Type
    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby("Contract")["churn_binary"].mean().mul(100).reset_index()
    ax = sns.barplot(data=contract_churn, x="Contract", y="churn_binary", order=["Month-to-month", "One year", "Two year"])
    plt.title("Churn Rate (%) by Subscription Contract Type", fontsize=13, fontweight="bold")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Contract Type")
    plt.ylim(0, 55)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_churn_by_contract.png", dpi=300)
    plt.close()

    # Chart 2: Churn Rate across Customer Tenure Cohorts
    plt.figure(figsize=(9, 5))
    cohort_order = ["0-6m (Onboarding)", "6-12m (Early)", "12-24m (Mid)", "24-48m (Established)", "48m+ (Loyal)"]
    cohort_churn = df.groupby("tenure_cohort")["churn_binary"].mean().mul(100).reindex(cohort_order).reset_index()
    ax = sns.lineplot(data=cohort_churn, x="tenure_cohort", y="churn_binary", marker="o", linewidth=2.5, markersize=8)
    plt.title("Customer Churn Trajectory by Tenure Cohort", fontsize=13, fontweight="bold")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Customer Lifecycle Stage")
    plt.xticks(rotation=15)
    plt.ylim(0, 60)
    for x, y in zip(range(len(cohort_churn)), cohort_churn["churn_binary"]):
        plt.text(x, y + 1.8, f"{y:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_churn_by_tenure_cohort.png", dpi=300)
    plt.close()

    # Chart 3: Monthly Charges Distribution by Churn Status
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, alpha=0.4)
    plt.title("Monthly Charges Density: Retained vs. Churned Customers", fontsize=13, fontweight="bold")
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_monthly_charges_distribution.png", dpi=300)
    plt.close()

    logger.info(f"Saved 3 EDA figures to {figures_dir}")


def run_driver_modeling(df: pd.DataFrame, report_path: Path, figures_dir: Path) -> Dict[str, Any]:
    """Trains Logistic Regression & Random Forest pipelines and exports validation evidence."""
    drop_cols = ["customerID", "Churn", "churn_binary", "tenure_cohort"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["churn_binary"]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    }

    results: Dict[str, Any] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        cls_rep = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "test_roc_auc": round(float(test_auc), 4),
            "churn_recall": round(float(cls_rep["1"]["recall"]), 4),
            "churn_precision": round(float(cls_rep["1"]["precision"]), 4),
            "churn_f1": round(float(cls_rep["1"]["f1-score"]), 4)
        }

        # Extract top drivers from Logistic Regression coefficients for interpretability
        if name == "LogisticRegression":
            feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
            coefs = pipe.named_steps["classifier"].coef_[0]
            driver_df = pd.DataFrame({"feature": feature_names, "log_odds": coefs})
            driver_df["abs_impact"] = driver_df["log_odds"].abs()
            top_drivers = driver_df.sort_values("abs_impact", ascending=False).head(12)

            plt.figure(figsize=(9, 6))
            sns.barplot(data=top_drivers, x="log_odds", y="feature", hue="feature", palette="coolwarm", legend=False)
            plt.title("Top 12 Churn Drivers (Standardized Log-Odds Impact)", fontsize=13, fontweight="bold")
            plt.xlabel("Log-Odds Impact (+ Increases Churn Risk, - Reduces Risk)")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(figures_dir / "04_top_churn_drivers.png", dpi=300)
            plt.close()

            results["top_drivers"] = top_drivers[["feature", "log_odds"]].to_dict(orient="records")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Model evaluation evidence saved to {report_path}")
    return results
