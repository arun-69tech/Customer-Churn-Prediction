from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "Datasets" / "Telco-Customer-Churn-Cleaned.csv"
IMAGES_DIR = PROJECT_ROOT / "Images"


def save_plot(filename: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


def churn_percentage_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    table = pd.crosstab(data[column], data["ChurnLabel"], normalize="index") * 100
    return table.round(2)


def plot_churn_count(data: pd.DataFrame, column: str, title: str, xlabel: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    sns.countplot(data=data, x=column, hue="ChurnLabel", palette="Set2")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Customer Count")
    plt.xticks(rotation=25)
    save_plot(filename)


def main() -> None:
    data = pd.read_csv(CLEANED_DATA_PATH)
    dataset_shape = data.shape

    sns.set_style("whitegrid")
    saved_visuals = []

    churn_labels = data["Churn"].map({0: "Stayed", 1: "Churned"})

    # STEP 1 : Load Dataset and Basic Verification
    churn_count = churn_labels.value_counts()
    churn_percentage = churn_labels.value_counts(normalize=True) * 100

    print("\nSTEP 1 : Dataset Verification")
    print(f"Dataset shape: {dataset_shape}")
    print("\nFirst five rows:")
    print(data.head())
    print("\nChurn counts:")
    print(churn_count)
    print("\nChurn percentages:")
    print(churn_percentage.round(2))

    data["ChurnLabel"] = churn_labels
    data["SeniorCitizenLabel"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # STEP 2 : Churn Distribution Analysis
    print("\nSTEP 2 : Churn Distribution Analysis")
    print(f"Percentage of customers churned: {churn_percentage.get('Churned', 0):.2f}%")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="ChurnLabel", hue="ChurnLabel", palette="Set2", legend=False)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Customer Count")
    save_plot("churn_distribution_count.png")
    saved_visuals.append("churn_distribution_count.png")

    plt.figure(figsize=(6, 6))
    plt.pie(churn_count, labels=churn_count.index, autopct="%1.1f%%", startangle=90)
    plt.title("Customer Churn Percentage")
    save_plot("churn_distribution_pie.png")
    saved_visuals.append("churn_distribution_pie.png")

    # STEP 3 : Contract Type vs Churn
    contract_percentage = churn_percentage_table(data, "Contract")

    print("\nSTEP 3 : Contract Type vs Churn")
    print(contract_percentage)

    plot_churn_count(
        data,
        "Contract",
        "Contract Type vs Churn",
        "Contract Type",
        "contract_vs_churn.png",
    )
    saved_visuals.append("contract_vs_churn.png")

    # STEP 4 : Tenure vs Churn
    tenure_summary = data.groupby("ChurnLabel")["tenure"].describe()

    print("\nSTEP 4 : Tenure vs Churn")
    print(tenure_summary)

    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="tenure", hue="ChurnLabel", bins=30, palette="Set2")
    plt.title("Tenure Distribution by Churn")
    plt.xlabel("Tenure in Months")
    plt.ylabel("Customer Count")
    save_plot("tenure_distribution.png")
    saved_visuals.append("tenure_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data, x="ChurnLabel", y="tenure", hue="ChurnLabel", palette="Set2", legend=False)
    plt.title("Tenure by Churn Status")
    plt.xlabel("Churn Status")
    plt.ylabel("Tenure in Months")
    save_plot("tenure_vs_churn.png")
    saved_visuals.append("tenure_vs_churn.png")

    # STEP 5 : Monthly Charges vs Churn
    monthly_charges_summary = data.groupby("ChurnLabel")["MonthlyCharges"].describe()

    print("\nSTEP 5 : Monthly Charges vs Churn")
    print(monthly_charges_summary)

    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="MonthlyCharges", hue="ChurnLabel", bins=30, palette="Set2")
    plt.title("Monthly Charges Distribution by Churn")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Customer Count")
    save_plot("monthly_charges_distribution.png")
    saved_visuals.append("monthly_charges_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=data,
        x="ChurnLabel",
        y="MonthlyCharges",
        hue="ChurnLabel",
        palette="Set2",
        legend=False,
    )
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Charges")
    save_plot("monthly_charges_vs_churn.png")
    saved_visuals.append("monthly_charges_vs_churn.png")

    # STEP 6 : Internet Service vs Churn
    internet_percentage = churn_percentage_table(data, "InternetService")

    print("\nSTEP 6 : Internet Service vs Churn")
    print(internet_percentage)

    plot_churn_count(
        data,
        "InternetService",
        "Internet Service vs Churn",
        "Internet Service",
        "internet_service_vs_churn.png",
    )
    saved_visuals.append("internet_service_vs_churn.png")

    # STEP 7 : Payment Method vs Churn
    payment_percentage = churn_percentage_table(data, "PaymentMethod")

    print("\nSTEP 7 : Payment Method vs Churn")
    print(payment_percentage)

    plot_churn_count(
        data,
        "PaymentMethod",
        "Payment Method vs Churn",
        "Payment Method",
        "payment_method_vs_churn.png",
    )
    saved_visuals.append("payment_method_vs_churn.png")

    # STEP 8 : Tech Support vs Churn
    techsupport_percentage = churn_percentage_table(data, "TechSupport")

    print("\nSTEP 8 : Tech Support vs Churn")
    print(techsupport_percentage)

    plot_churn_count(
        data,
        "TechSupport",
        "Tech Support vs Churn",
        "Tech Support",
        "techsupport_vs_churn.png",
    )
    saved_visuals.append("techsupport_vs_churn.png")

    # STEP 9 : Online Security vs Churn
    online_security_percentage = churn_percentage_table(data, "OnlineSecurity")

    print("\nSTEP 9 : Online Security vs Churn")
    print(online_security_percentage)

    plot_churn_count(
        data,
        "OnlineSecurity",
        "Online Security vs Churn",
        "Online Security",
        "online_security_vs_churn.png",
    )
    saved_visuals.append("online_security_vs_churn.png")

    # STEP 10 : Senior Citizen vs Churn
    senior_percentage = churn_percentage_table(data, "SeniorCitizenLabel")

    print("\nSTEP 10 : Senior Citizen vs Churn")
    print(senior_percentage)

    plot_churn_count(
        data,
        "SeniorCitizenLabel",
        "Senior Citizen vs Churn",
        "Senior Citizen",
        "senior_citizen_vs_churn.png",
    )
    saved_visuals.append("senior_citizen_vs_churn.png")

    # STEP 11 : Business Insights
    insights = [
        "The overall churn rate is 26.58%, meaning roughly one in four customers has left the company.",
        "Month-to-month customers have the highest churn rate, showing that weak contract commitment is a major churn driver.",
        "One-year and two-year contracts show substantially lower churn, indicating that long-term contracts improve customer retention.",
        "Customers who churn have much shorter tenure, which makes the early customer lifecycle a critical retention window.",
        "Churned customers pay higher average monthly charges, suggesting price sensitivity among higher-bill customers.",
        "Fiber optic customers show elevated churn compared with DSL and no-internet customers.",
        "Electronic check users have the highest churn rate among payment methods.",
        "Customers without Tech Support are more likely to churn, highlighting support availability as a retention lever.",
        "Customers without Online Security show higher churn, indicating that value-added service adoption can strengthen retention.",
        "Senior citizens churn at a higher rate than non-senior customers and may require targeted retention offers.",
    ]

    print("\nSTEP 11 : Business Insights")
    for index, insight in enumerate(insights, start=1):
        print(f"{index}. {insight}")

    print("\nVisualization Summary")
    print(f"Total visualizations saved: {len(saved_visuals)}")
    print(f"Folder location: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
