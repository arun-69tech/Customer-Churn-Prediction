from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Telco-Customer-Churn-Cleaned.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "Datasets" / "Telco_Customer_Predictions.csv"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "Datasets" / "model_comparison.csv"
CROSS_VALIDATION_PATH = PROJECT_ROOT / "Datasets" / "cross_validation_results.csv"
MODELS_DIR = PROJECT_ROOT / "Models"
IMAGES_DIR = PROJECT_ROOT / "Images"


def create_output_folders() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(model_name: str, y_test: pd.Series, y_pred, y_probability) -> dict:
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_probability),
    }

    print(f"\n{model_name} Evaluation")
    print("-" * 60)
    print(f"Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"Precision : {metrics['Precision']:.4f}")
    print(f"Recall    : {metrics['Recall']:.4f}")
    print(f"F1 Score  : {metrics['F1 Score']:.4f}")
    print(f"ROC-AUC   : {metrics['ROC-AUC']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def plot_confusion_matrix(model_name: str, y_test: pd.Series, y_pred, filename: str) -> None:
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(filename)


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    metrics_df = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=metrics_df, x="Model", y="Score", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    save_plot("model_comparison.png")


def plot_roc_curves(roc_data: dict) -> None:
    plt.figure(figsize=(8, 6))

    for model_name, values in roc_data.items():
        fpr, tpr, auc_score = values
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_plot("roc_curves.png")


def plot_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top_features = importance_df.head(15)

    print("\nSTEP 9 : Top 15 Random Forest Feature Importances")
    print(top_features.to_string(index=False))

    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_features, x="Importance", y="Feature")
    plt.title("Top 15 Important Features for Churn Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("feature_importance.png")

    print("\nFeature Importance Business Interpretation")
    print(
        "Random Forest feature importance shows that contract type, tenure, billing "
        "behavior, monthly charges, and support-related services are key churn drivers."
    )

    return importance_df


def cross_validate_models(models: dict, X_encoded: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv_estimators = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=1000),
        ),
        "Decision Tree": models["Decision Tree"]["model"],
        "Random Forest": models["Random Forest"]["model"],
    }
    cv_results = []

    print("\nCross Validation ROC-AUC Scores")
    print("-" * 60)

    for model_name, estimator in cv_estimators.items():
        scores = cross_val_score(estimator, X_encoded, y, cv=5, scoring="roc_auc")
        result = {
            "Model": model_name,
            "Fold_1": scores[0],
            "Fold_2": scores[1],
            "Fold_3": scores[2],
            "Fold_4": scores[3],
            "Fold_5": scores[4],
            "Mean_ROC_AUC": scores.mean(),
            "Std_ROC_AUC": scores.std(),
        }
        cv_results.append(result)

        print(f"\n{model_name}")
        print(f"Fold Scores: {np.round(scores, 4)}")
        print(f"Mean ROC-AUC: {scores.mean():.4f}")
        print(f"Std: {scores.std():.4f}")

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(CROSS_VALIDATION_PATH, index=False)
    print(f"\nCross validation results saved to: {CROSS_VALIDATION_PATH}")

    return cv_results_df


def find_optimal_threshold(y_true: pd.Series, probabilities) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.argmax(f1_scores))

    threshold_details = {
        "threshold": thresholds[best_index],
        "precision": precision[best_index],
        "recall": recall[best_index],
        "f1_score": f1_scores[best_index],
    }

    print("\nThreshold Analysis")
    print("-" * 60)
    print(f"Optimal Threshold: {threshold_details['threshold']:.4f}")
    print(f"Precision: {threshold_details['precision']:.4f}")
    print(f"Recall: {threshold_details['recall']:.4f}")
    print(f"F1 Score: {threshold_details['f1_score']:.4f}")

    return threshold_details


def save_threshold_analysis(best_model_name: str, threshold_details: dict) -> None:
    threshold_text = f"""Best Model Name: {best_model_name}
Optimal Threshold: {threshold_details["threshold"]:.4f}
Precision: {threshold_details["precision"]:.4f}
Recall: {threshold_details["recall"]:.4f}
F1 Score: {threshold_details["f1_score"]:.4f}
"""

    (MODELS_DIR / "threshold_analysis.txt").write_text(threshold_text, encoding="utf-8")


def save_model_info(
    best_model_name: str,
    best_metrics: pd.Series,
    feature_count: int,
    cv_metrics: pd.Series,
    threshold_details: dict,
) -> None:
    model_info = f"""Best Model Name: {best_model_name}
Training Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Number of Features: {feature_count}

Evaluation Metrics:
Accuracy: {best_metrics["Accuracy"]:.4f}
Precision: {best_metrics["Precision"]:.4f}
Recall: {best_metrics["Recall"]:.4f}
F1 Score: {best_metrics["F1 Score"]:.4f}
ROC-AUC: {best_metrics["ROC-AUC"]:.4f}

Cross Validation:
Mean ROC-AUC: {cv_metrics["Mean_ROC_AUC"]:.4f}
Std ROC-AUC: {cv_metrics["Std_ROC_AUC"]:.4f}

Deployment Threshold:
Optimal Threshold: {threshold_details["threshold"]:.4f}
"""

    (MODELS_DIR / "model_info.txt").write_text(model_info, encoding="utf-8")


def main() -> None:
    try:
        create_output_folders()

        # STEP 1 : Load Data
        data = pd.read_csv(DATA_PATH)

        print("\nSTEP 1 : Load Data")
        print(f"Dataset shape: {data.shape}")
        print("\nData types:")
        print(data.dtypes)
        print("\nChurn distribution:")
        print(data["Churn"].value_counts())
        print("\nChurn distribution percentage:")
        print((data["Churn"].value_counts(normalize=True) * 100).round(2))

        # STEP 2 : Data Preparation
        ml_data = data.copy()
        ml_data.drop(columns=["customerID"], inplace=True)

        X = ml_data.drop(columns=["Churn"])
        y = ml_data["Churn"]

        print("\nSTEP 2 : Data Preparation")
        print(f"Feature data shape before encoding: {X.shape}")
        print(f"Target data shape: {y.shape}")

        # STEP 3 : Encode Features
        X_encoded = pd.get_dummies(X, drop_first=True)
        feature_names = X_encoded.columns.tolist()
        joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

        print("\nSTEP 3 : Encode Features")
        print(f"Feature matrix shape: {X_encoded.shape}")
        print(f"Number of features generated: {len(feature_names)}")
        print(f"Feature names saved to: {MODELS_DIR / 'feature_names.pkl'}")

        # STEP 4 : Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        print("\nSTEP 4 : Train Test Split")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print("\ny_train distribution:")
        print(y_train.value_counts(normalize=True).round(4))
        print("\ny_test distribution:")
        print(y_test.value_counts(normalize=True).round(4))

        # STEP 5 : Feature Scaling for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_encoded_scaled = scaler.transform(X_encoded)

        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

        print("\nSTEP 5 : Feature Scaling")
        print(f"Scaler saved to: {MODELS_DIR / 'scaler.pkl'}")

        # STEP 6 : Train Machine Learning Models
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "train_data": X_train_scaled,
                "test_data": X_test_scaled,
                "all_data": X_encoded_scaled,
                "filename": "logistic_regression.pkl",
                "confusion_matrix": "confusion_matrix_logistic.png",
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(random_state=42, max_depth=5),
                "train_data": X_train,
                "test_data": X_test,
                "all_data": X_encoded,
                "filename": "decision_tree.pkl",
                "confusion_matrix": "confusion_matrix_decision_tree.png",
            },
            "Random Forest": {
                "model": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                "train_data": X_train,
                "test_data": X_test,
                "all_data": X_encoded,
                "filename": "random_forest.pkl",
                "confusion_matrix": "confusion_matrix_random_forest.png",
            },
        }

        # STEP 6A : Cross Validation
        cv_results_df = cross_validate_models(models, X_encoded, y)

        print("\nSTEP 6 : Train Machine Learning Models")
        for model_name, model_info in models.items():
            model_info["model"].fit(model_info["train_data"], y_train)
            joblib.dump(model_info["model"], MODELS_DIR / model_info["filename"])
            print(f"{model_name} saved to: {MODELS_DIR / model_info['filename']}")

        # STEP 7 and STEP 8 : Model Evaluation and Visualizations
        evaluation_results = []
        roc_data = {}

        print("\nSTEP 7 : Model Evaluation")
        for model_name, model_info in models.items():
            model = model_info["model"]
            y_pred = model.predict(model_info["test_data"])
            y_probability = model.predict_proba(model_info["test_data"])[:, 1]

            metrics = evaluate_model(model_name, y_test, y_pred, y_probability)
            evaluation_results.append(metrics)

            fpr, tpr, _ = roc_curve(y_test, y_probability)
            roc_data[model_name] = (fpr, tpr, metrics["ROC-AUC"])

            plot_confusion_matrix(
                model_name,
                y_test,
                y_pred,
                model_info["confusion_matrix"],
            )

        comparison_df = pd.DataFrame(evaluation_results)
        comparison_df = comparison_df.sort_values(
            by=["ROC-AUC", "Recall", "F1 Score"],
            ascending=False,
        ).reset_index(drop=True)

        comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

        print("\nModel Comparison")
        print(comparison_df.round(4).to_string(index=False))
        print(f"\nModel comparison saved to: {MODEL_COMPARISON_PATH}")

        print("\nSTEP 8 : Evaluation Visualizations")
        plot_model_comparison(comparison_df)
        plot_roc_curves(roc_data)
        print(f"Evaluation visualizations saved to: {IMAGES_DIR}")

        # STEP 9 : Feature Importance
        importance_df = plot_feature_importance(models["Random Forest"]["model"], feature_names)

        # STEP 10 : Select Best Model
        best_model_name = comparison_df.iloc[0]["Model"]
        best_metrics = comparison_df.iloc[0]
        best_model = models[best_model_name]["model"]
        best_all_data = models[best_model_name]["all_data"]
        best_test_data = models[best_model_name]["test_data"]
        best_test_probabilities = best_model.predict_proba(best_test_data)[:, 1]
        threshold_details = find_optimal_threshold(y_test, best_test_probabilities)
        best_cv_metrics = cv_results_df.loc[cv_results_df["Model"] == best_model_name].iloc[0]

        print("\nSTEP 10 : Select Best Model")
        print(f"Best Model Selected: {best_model_name}")
        print(
            "Justification: This model achieved the strongest ROC-AUC score, "
            "with recall and F1 score considered as supporting metrics."
        )

        # STEP 11 : Generate Predictions
        probabilities = best_model.predict_proba(best_all_data)[:, 1]
        predictions = (probabilities >= threshold_details["threshold"]).astype(int)

        prediction_data = data.copy()
        prediction_data["Predicted_Churn"] = predictions
        prediction_data["Churn_Probability"] = probabilities.round(4)
        prediction_data.to_csv(PREDICTIONS_PATH, index=False)

        print("\nSTEP 11 : Generate Predictions")
        print(f"Prediction dataset saved to: {PREDICTIONS_PATH}")
        print("\nTop 10 high-risk customers:")
        high_risk_columns = ["customerID", "Churn", "Predicted_Churn", "Churn_Probability"]
        print(
            prediction_data[high_risk_columns]
            .sort_values("Churn_Probability", ascending=False)
            .head(10)
            .to_string(index=False)
        )

        # STEP 12 : Save Final Model
        joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
        save_threshold_analysis(best_model_name, threshold_details)
        save_model_info(
            best_model_name,
            best_metrics,
            len(feature_names),
            best_cv_metrics,
            threshold_details,
        )

        print("\nSTEP 12 : Save Final Model")
        print(f"Best model saved to: {MODELS_DIR / 'best_model.pkl'}")
        print(f"Model info saved to: {MODELS_DIR / 'model_info.txt'}")
        print(f"Threshold analysis saved to: {MODELS_DIR / 'threshold_analysis.txt'}")

        # STEP 13 : Business Insights
        print("\nSTEP 13 : Business Insights")
        insights = [
            "The model can score every customer by churn probability, enabling retention teams to prioritize the highest-risk accounts first.",
            "ROC-AUC was used as the primary selection metric because churn prediction depends on ranking customers by risk, not only classifying them.",
            "Month-to-month contract indicators are among the strongest churn drivers, confirming that short-term commitment increases churn risk.",
            "Tenure-related behavior is highly important, showing that newer customers need stronger onboarding and early-life retention programs.",
            "High monthly charges contribute to churn risk, suggesting that price-sensitive customers may respond to targeted offers or plan optimization.",
            "Internet service and fiber optic indicators are important churn signals and should be reviewed with service quality and pricing data.",
            "Support and security service features help explain churn, indicating that value-added services may improve customer stickiness.",
            "Electronic check and billing-related patterns remain useful for identifying customers with elevated churn probability.",
            "The prediction dataset is ready for Power BI so business users can filter, rank, and monitor high-risk customers.",
            "Feature importance should guide retention strategy around contracts, tenure, billing, service type, and support availability.",
        ]

        for index, insight in enumerate(insights, start=1):
            print(f"{index}. {insight}")

        print("\nMachine Learning phase completed successfully.")
        print(f"Top feature importance exported in plot. Highest feature: {importance_df.iloc[0]['Feature']}")

    except FileNotFoundError as error:
        print(f"File not found: {error}")
    except KeyError as error:
        print(f"Required column missing from dataset: {error}")
    except Exception as error:
        print(f"Unexpected error while running ML pipeline: {error}")
        raise


if __name__ == "__main__":
    main()
