# Model Card: Used Car Price Estimator

## Model overview

- **Task:** Regression
- **Target:** Historical used-car listing price (Indian rupees)
- **Estimator:** scikit-learn `Pipeline`
- **Categorical preprocessing:** `OneHotEncoder` applied to `name`, `company`, and `fuel_type`
- **Regression algorithm:** `LinearRegression`
- **Inference artifact:** `LinearRegressionModel.pkl`

The application passes the five raw input fields to the saved pipeline. Encoding is performed inside the pipeline.

## Data

The repository contains an original CSV (`quikr_car.csv`) and a cleaned CSV (`Cleaned_Car.csv`). The cleaned file contains 815 records, with recorded manufacture years from 1995 to 2019 and prices from ₹30,000 to ₹31,00,000.

## Reported evaluation

The notebook reports an `R²` score of approximately 0.8667 using a 10% random test split with `random_state=655`.

This score is reproduced here as a notebook-reported result only; it has not been independently recomputed as part of the deployment work. The original notebook derived the categorical encoder's category lists before splitting the dataset. A future evaluation should fit preprocessing exclusively on each training fold, then assess the model on an untouched holdout set or cross-validation.

## Intended use

- Educational demonstration of a tabular regression workflow.
- Exploring how the historical dataset's selected features relate to listed used-car prices.
- Demonstrating an end-to-end scikit-learn inference pipeline through Streamlit.

## Out-of-scope use

- Treating the output as an official valuation, guaranteed sale price, loan collateral value, or insurance valuation.
- Making a purchase or sale decision without independent inspection and market research.
- Applying the model to current market listings without retraining and validation on representative recent data.

## Known limitations and risks

- Only 815 cleaned observations are available.
- The dataset covers historical listings and may not represent today's market.
- Important price drivers are absent, including condition, location, ownership count, accident history, service records, transmission, and insurance status.
- The model is linear and can produce implausible or non-positive estimates for unusual feature combinations. The notebook includes a synthetic example that produced a negative prediction.
- The notebook's preprocessing setup should be made fully reproducible before making stronger performance claims.
- The pickle artifact's embedded scikit-learn version metadata records `1.3.0`; the deployment environment should use that version when loading this model.

## Monitoring and maintenance

Before real-world use, collect a recent and representative dataset, rebuild the preprocessing and regression pipeline using a reproducible training script, evaluate with a leakage-safe split and suitable regression metrics (MAE, RMSE, and R²), inspect residuals and subgroup errors, and record package versions alongside the model artifact.

## Security

Only load the trusted model artifact maintained in this repository. Python pickle deserialization can execute arbitrary code.
