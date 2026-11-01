from __future__ import annotations

import streamlit as st

from car_price_predictor.data import load_car_data
from car_price_predictor.inference import load_model, predict_price


st.set_page_config(
    page_title="Used Car Price Estimator",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Used Car Price Estimator")
st.write(
    "Estimate the resale price of a used car from its manufacturer, model, "
    "manufacture year, fuel type, and kilometres driven."
)
st.caption(
    "This is an educational estimate based on historical listing data. "
    "It is not a live market valuation or a substitute for a physical inspection."
)


@st.cache_data(show_spinner=False)
def get_car_data():
    return load_car_data()


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


try:
    car_data = get_car_data()
    model = get_model()
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error(f"The application could not load its data or model: {exc}")
    st.stop()


st.subheader("Enter vehicle details")

col1, col2 = st.columns(2)

companies = sorted(car_data["company"].dropna().unique().tolist())
with col1:
    company = st.selectbox("Manufacturer", options=companies)

    model_options = sorted(
        car_data.loc[car_data["company"] == company, "name"]
        .dropna()
        .unique()
        .tolist()
    )
    car_name = st.selectbox("Car model", options=model_options)

    min_year = int(car_data["year"].min())
    max_year = int(car_data["year"].max())
    year = st.number_input(
        "Year of manufacture",
        min_value=min_year,
        max_value=max_year,
        value=max_year,
        step=1,
        help=f"Available training-data range: {min_year}–{max_year}.",
    )

fuel_options = sorted(
    car_data.loc[
        (car_data["company"] == company) & (car_data["name"] == car_name),
        "fuel_type",
    ].dropna().unique().tolist()
)

with col2:
    fuel_type = st.selectbox("Fuel type", options=fuel_options)

    max_kms = int(car_data["kms_driven"].max())
    default_kms = min(50_000, max_kms)
    kms_driven = st.number_input(
        "Kilometres driven",
        min_value=0,
        max_value=max_kms,
        value=default_kms,
        step=1_000,
        help=f"Maximum mileage in the cleaned dataset: {max_kms:,} km.",
    )

st.caption("The selectable ranges are based on values represented in the training dataset.")

if st.button("Estimate resale price", type="primary", use_container_width=True):
    try:
        estimate = predict_price(
            model=model,
            name=car_name,
            company=company,
            year=int(year),
            kms_driven=int(kms_driven),
            fuel_type=fuel_type,
        )

        observed_min = float(car_data["Price"].min())
        observed_max = float(car_data["Price"].max())

        if estimate <= 0:
            st.warning(
                "The model returned a non-positive estimate for this combination. "
                "This linear model cannot provide a meaningful price for this input. "
                "Try a configuration closer to the examples in the training data."
            )
        else:
            st.success("Estimate generated")
            st.metric("Estimated resale price", f"₹{estimate:,.0f}")

            if estimate < observed_min or estimate > observed_max:
                st.warning(
                    "This estimate is outside the price range observed in the "
                    "training dataset. Treat it with additional caution."
                )

            st.caption(
                "The estimate is produced by the saved OneHotEncoder + "
                "LinearRegression pipeline."
            )

    except (ValueError, TypeError, AttributeError) as exc:
        st.error(f"Unable to generate a prediction: {exc}")


with st.expander("About this estimate"):
    st.markdown(
        """
        - The model uses historical used-car listings stored in this repository.
        - Categorical fields are encoded inside the saved scikit-learn pipeline.
        - The training data does not include several factors that influence resale
          value, such as vehicle condition, ownership history, accident records,
          location, transmission, and service history.
        - Actual selling prices can differ substantially from this estimate.
        """
    )
