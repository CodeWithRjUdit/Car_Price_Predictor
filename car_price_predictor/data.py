from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Cleaned_Car.csv"

REQUIRED_COLUMNS = [
    "name",
    "company",
    "year",
    "Price",
    "kms_driven",
    "fuel_type",
]


def load_car_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the cleaned dataset used by the original project."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    data = pd.read_csv(path)

    # The original notebook exported the dataframe with its index.
    unnamed_columns = [
        column
        for column in data.columns
        if str(column).lower().startswith("unnamed")
    ]
    if unnamed_columns:
        data = data.drop(columns=unnamed_columns)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()

    for column in ("year", "Price", "kms_driven"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ("name", "company", "fuel_type"):
        data[column] = data[column].astype("string").str.strip()

    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[
        (data["name"] != "")
        & (data["company"] != "")
        & (data["fuel_type"] != "")
    ].copy()

    if data.empty:
        raise ValueError("The cleaned dataset contains no usable records.")

    data["year"] = data["year"].astype(int)
    data["Price"] = data["Price"].astype(float)
    data["kms_driven"] = data["kms_driven"].astype(int)

    return data.reset_index(drop=True)
