from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Datasets" / "Telco-Customer-Churn.csv"


def main() -> None:
    data = pd.read_csv(DATASET_PATH)

    print(f"{data.info()}\n")

    print(f"Duplicated Values: {data.duplicated().sum()}\n")

    print(f"Null Values: \n{data.isnull().sum()}\n")

    print(f"Unique Values: \n{data.nunique().sort_values()}\n")


if __name__ == "__main__":
    main()
