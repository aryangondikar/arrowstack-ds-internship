import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def test_categorical_driver(
    df: pd.DataFrame,
    feature: str,
    target: str = "attrition_binary",
    min_subgroup_size: int = 10
) -> Dict[str, Any]:
    """Runs Chi-Square independence test with small-sample governance guardrails."""
    contingency = pd.crosstab(df[feature], df[target])
    group_counts = df[feature].value_counts()
    small_groups = group_counts[group_counts < min_subgroup_size].to_dict()

    warning_msg = None
    if small_groups:
        warning_msg = (
            f"Governance Warning: Subgroups {small_groups} in '{feature}' have n < {min_subgroup_size}. "
            "Do not use for automated HR decisions."
        )
        logger.warning(warning_msg)

    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    rates = df.groupby(feature)[target].mean().mul(100).round(2).to_dict()

    return {
        "feature": feature,
        "test_type": "Chi-Square",
        "statistic": round(float(chi2), 4),
        "p_value": round(float(p_val), 6),
        "significant_at_0_05": bool(p_val < 0.05),
        "attrition_rate_by_group_pct": rates,
        "governance_warning": warning_msg
    }


def test_numeric_driver(
    df: pd.DataFrame,
    feature: str,
    target: str = "attrition_binary"
) -> Dict[str, Any]:
    """Runs non-parametric Mann-Whitney U test comparing retained vs. attrited distributions."""
    retained = df[df[target] == 0][feature].dropna()
    attrited = df[df[target] == 1][feature].dropna()

    stat, p_val = stats.mannwhitneyu(retained, attrited, alternative="two-sided")
    return {
        "feature": feature,
        "test_type": "Mann-Whitney U",
        "statistic": round(float(stat), 2),
        "p_value": round(float(p_val), 6),
        "significant_at_0_05": bool(p_val < 0.05),
        "retained_median": round(float(retained.median()), 2),
        "attrited_median": round(float(attrited.median()), 2)
    }


def generate_workforce_charts(df: pd.DataFrame, figures_dir: Path) -> None:
    """Exports 4 high-resolution (300 DPI) workforce analytics figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Attrition Rate by Hiring Cohort Stage
    plt.figure(figsize=(8, 5))
    stage_order = ["0-1y (New Hire)", "1-3y (Ramp-Up)", "3-7y (Core)", "7y+ (Veteran)"]
    stage_rates = df.groupby("hiring_tenure_stage")["attrition_binary"].mean().mul(100).reindex(stage_order).reset_index()
    ax = sns.barplot(data=stage_rates, x="hiring_tenure_stage", y="attrition_binary", hue="hiring_tenure_stage", palette="Blues_d", legend=False)
    plt.title("Attrition Rate (%) Across Hiring & Tenure Stages", fontsize=13, fontweight="bold")
    plt.ylabel("Attrition Rate (%)")
    plt.xlabel("Employee Tenure Stage")
    plt.ylim(0, 45)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_attrition_by_tenure_stage.png", dpi=300)
    plt.close()

    # Chart 2: Overtime Workload Strain vs. Attrition
    plt.figure(figsize=(7, 5))
    ot_rates = df.groupby(["Department", "OverTime"])["attrition_binary"].mean().mul(100).reset_index()
    sns.barplot(data=ot_rates, x="Department", y="attrition_binary", hue="OverTime")
    plt.title("Attrition Rate (%) by Department & Overtime Workload", fontsize=13, fontweight="bold")
    plt.ylabel("Attrition Rate (%)")
    plt.xlabel("Department")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_attrition_by_overtime_dept.png", dpi=300)
    plt.close()

    # Chart 3: Monthly Income Distribution by Job Role & Attrition
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x="MonthlyIncome", y="JobRole", hue="Attrition")
    plt.title("Compensation Distribution by Job Role & Attrition Status", fontsize=13, fontweight="bold")
    plt.xlabel("Monthly Income ($)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_income_by_role_attrition.png", dpi=300)
    plt.close()

    # Chart 4: Productivity Rating vs. Job Satisfaction Heatmap
    plt.figure(figsize=(7, 5))
    heat = df.pivot_table(
        index="PerformanceRating",
        columns="JobSatisfaction",
        values="attrition_binary",
        aggfunc="mean"
    ).mul(100)
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Attrition Rate (%): Performance Rating vs. Job Satisfaction", fontsize=12, fontweight="bold")
    plt.xlabel("Job Satisfaction (1=Low, 4=Very High)")
    plt.ylabel("Performance Rating (3=Excellent, 4=Outstanding)")
    plt.tight_layout()
    plt.savefig(figures_dir / "04_productivity_satisfaction_heatmap.png", dpi=300)
    plt.close()

    logger.info(f"Saved 4 workforce analytics figures to {figures_dir}")


def run_workforce_study(df: pd.DataFrame, report_path: Path) -> Dict[str, Any]:
    """Executes statistical hypothesis battery & multivariate driver baseline."""
    cat_features = ["OverTime", "Department", "JobRole", "hiring_tenure_stage", "PerformanceRating"]
    num_features = ["MonthlyIncome", "Age", "YearsAtCompany", "promotion_stagnation_ratio"]

    hypothesis_results: List[Dict[str, Any]] = []
    for col in cat_features:
        if col in df.columns:
            hypothesis_results.append(test_categorical_driver(df, col))
    for col in num_features:
        if col in df.columns:
            hypothesis_results.append(test_numeric_driver(df, col))

    # Multivariate baseline model (5-fold Stratified CV) for risk scoring validation
    drop_cols = ["Attrition", "attrition_binary", "employee_hash", "hiring_tenure_stage"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["attrition_binary"]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
        ]
    )
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    report = {
        "total_employees_analyzed": int(len(df)),
        "overall_attrition_rate_pct": round(float(df["attrition_binary"].mean() * 100), 2),
        "multivariate_5fold_roc_auc": round(float(cv_scores.mean()), 4),
        "hypothesis_tests": hypothesis_results
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Workforce statistical & governance report saved to {report_path}")
    return report
