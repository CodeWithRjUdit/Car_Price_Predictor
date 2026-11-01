from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from car_price_predictor.data import load_car_data
from car_price_predictor.inference import load_model, predict_price


def main() -> None:
    data = load_car_data()
    model = load_model()

    # Use an actual cleaned record so all categorical values are known to
    # the saved OneHotEncoder.
    sample = data.iloc[len(data) // 2]

    prediction = predict_price(
        model=model,
        name=str(sample["name"]),
        company=str(sample["company"]),
        year=int(sample["year"]),
        kms_driven=int(sample["kms_driven"]),
        fuel_type=str(sample["fuel_type"]),
    )

    if not math.isfinite(prediction):
        raise RuntimeError("Smoke test failed: prediction is not finite.")

    print("Artifact smoke test passed.")
    print(f"Sample input: {sample['name']} ({sample['year']})")
    print(f"Model output: ₹{prediction:,.2f}")

    if prediction <= 0:
        print(
            "Warning: the model returned a non-positive estimate for this "
            "sample. Review model quality before using the output."
        )


if __name__ == "__main__":
    main()
