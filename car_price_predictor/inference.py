from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "LinearRegressionModel.pkl"

# Keep this order aligned with the dataframe used to train the saved pipeline.
FEATURE_COLUMNS = ["name", "company", "year", "kms_driven", "fuel_type"]


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load the trusted, pre-trained scikit-learn pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file was not found: {path}")

    # Pickle files can execute code while being loaded. Only load the model
    # artifact committed with this trusted repository.
    with path.open("rb") as model_file:
        model = pickle.load(model_file)

    if not callable(getattr(model, "predict", None)):
        raise ValueError("The saved artifact does not expose a predict() method.")

    return model


def predict_price(
    model: Any,
    name: str,
    company: str,
    year: int,
    kms_driven: int,
    fuel_type: str,
) -> float:
    """Predict the price from the five raw input features expected by the pipeline."""
    input_frame = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=FEATURE_COLUMNS,
    )

    prediction = float(model.predict(input_frame)[0])

    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")

    return prediction
