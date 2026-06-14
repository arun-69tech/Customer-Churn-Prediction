from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "Datasets" / "Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "Datasets" / "Telco-Customer-Churn-Cleaned.csv"


def main() -> None:
    data = pd.read_csv(RAW_DATA_PATH)

    print((data["TotalCharges"] == " ").sum())

    data["TotalCharges"] = data["TotalCharges"].replace(" ", np.nan)

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])

    print(data["TotalCharges"].isnull().sum())

    data.dropna(inplace=True)

    data["Churn"] = data["Churn"].map({"No": 0,"Yes": 1})

    print(data["Churn"].value_counts())

    print(data.info())
    
    data.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"File Saved Successfully to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()
