
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sri Lanka Ensemble Learning Dashboard",
    page_icon="🏠",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


# =====================================================
# LOAD SAVED MACHINE LEARNING MODELS
# =====================================================

@st.cache_resource
def load_models():

    baseline = joblib.load(
        MODELS_DIR / "baseline.joblib"
    )

    random_forest = joblib.load(
        MODELS_DIR / "random_forest.joblib"
    )

    # XGBoost is loaded using its portable JSON model
    xgb_preprocessor = joblib.load(
        MODELS_DIR / "xgboost_preprocessor.joblib"
    )

    xgb_regressor = XGBRegressor()
    xgb_regressor.load_model(
        str(MODELS_DIR / "xgboost_native.json")
    )

    xgboost = Pipeline([
        ("preprocessor", xgb_preprocessor),
        ("model", xgb_regressor)
    ])

    catboost = joblib.load(
        MODELS_DIR / "catboost.joblib"
    )

    stacking_meta = joblib.load(
        MODELS_DIR / "stacking_meta.joblib"
    )

    return {
        "Baseline": baseline,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "CatBoost": catboost,
        "Stacking Meta": stacking_meta
    }


# =====================================================
# LOAD DATASET AND EVALUATION RESULTS
# =====================================================

@st.cache_data
def load_data():

    housing = pd.read_csv(
        DATA_DIR / "housing_clean.csv"
    )

    validation = pd.read_csv(
        DATA_DIR / "validation_results.csv"
    )

    testing = pd.read_csv(
        DATA_DIR / "test_results.csv"
    )

    with open(
        ROOT / "metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    return housing, validation, testing, metadata


try:
    models = load_models()

    housing, validation, testing, metadata = load_data()

except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_house(input_data):

    rf_prediction = float(
        models["Random Forest"].predict(input_data)[0]
    )

    xgb_prediction = float(
        models["XGBoost"].predict(input_data)[0]
    )

    cat_prediction = float(
        models["CatBoost"].predict(input_data)[0]
    )

    baseline_prediction = float(
        models["Baseline"].predict(input_data)[0]
    )

    # Hybrid: Random Forest + XGBoost
    hybrid_prediction = (
        0.5 * rf_prediction
        + 0.5 * xgb_prediction
    )

    # Equal-weight ensemble
    equal_prediction = (
        rf_prediction
        + xgb_prediction
        + cat_prediction
    ) / 3

    # Stacking meta-learner
    base_predictions = {
        "Random Forest": rf_prediction,
        "XGBoost": xgb_prediction,
        "CatBoost": cat_prediction
    }

    model_order = metadata["base_model_order"]

    stacking_input = pd.DataFrame(
        [{
            name: base_predictions[name]
            for name in model_order
        }]
    )

    stacking_prediction = float(
        models["Stacking Meta"].predict(
            stacking_input
        )[0]
    )

    return {
        "Baseline": baseline_prediction,
        "Random Forest": rf_prediction,
        "XGBoost": xgb_prediction,
        "CatBoost": cat_prediction,
        "Hybrid RF + XGBoost": hybrid_prediction,
        "Equal-Weight Ensemble": equal_prediction,
        "Stacking": stacking_prediction
    }


# =====================================================
# DASHBOARD HEADER
# =====================================================

st.title("🏠 Sri Lankan House Price Prediction")

st.subheader("Ensemble Learning Dashboard")

st.write(
    "Comparing Bagging, Boosting, Stacking, "
    "and Hybrid Machine Learning Models "
    "using Sri Lankan property advertisement data."
)

st.info(
    "This is an educational prediction system based "
    "on historical advertised asking prices. "
    "Predictions do not represent verified current "
    "market values or transaction prices."
)


# =====================================================
# DASHBOARD TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 Model Comparison",
    "🏠 Price Prediction",
    "🧠 Model Insights"
])


# =====================================================
# TAB 1 - DATASET OVERVIEW
# =====================================================

with tab1:

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Property Records",
        f"{len(housing):,}"
    )

    col2.metric(
        "Median House Price",
        f"Rs {housing['price_million'].median():.2f}M"
    )

    col3.metric(
        "Unique Locations",
        housing["location"].nunique()
    )

    st.subheader("House Price Distribution")

    # Limit visualization to the 99th percentile.
    # This does not modify the training dataset.
    price_limit = housing["price_million"].quantile(0.99)

    distribution_data = housing[
        housing["price_million"] <= price_limit
    ]

    fig = px.histogram(
        distribution_data,
        x="price_million",
        nbins=50,
        title="Distribution of Advertised House Prices",
        labels={
            "price_million": "House Price (LKR Millions)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "For readability, this visualization excludes "
        "prices above the 99th percentile. "
        "The original cleaned records remain unchanged."
    )

    st.subheader("Top 10 Property Locations")

    location_counts = (
        housing["location"]
        .value_counts()
        .head(10)
        .rename_axis("Location")
        .reset_index(name="Listings")
    )

    fig = px.bar(
        location_counts,
        x="Location",
        y="Listings",
        title="Locations With the Most House Listings"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        housing.head(20),
        use_container_width=True
    )


# =====================================================
# TAB 2 - MODEL COMPARISON
# =====================================================

with tab2:

    st.header("Ensemble Model Performance")

    comparison = validation.merge(
        testing,
        on="Model"
    )

    comparison = comparison.rename(
        columns={
            "MAE": "Validation_MAE",
            "RMSE": "Validation_RMSE",
            "R2": "Validation_R2"
        }
    )

    comparison = comparison.sort_values(
        "Validation_MAE"
    )

    st.subheader("Validation and Test Results")

    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Mean Absolute Error Comparison")

    fig = px.bar(
        comparison,
        x="Model",
        y=["Validation_MAE", "Test_MAE"],
        barmode="group",
        title="Validation vs Test MAE",
        labels={
            "value": "MAE (LKR Millions)",
            "variable": "Dataset"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Test R² Comparison")

    fig = px.bar(
        comparison,
        x="Model",
        y="Test_R2",
        title="Model Test R² Scores"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    selected_model = metadata["selected_model"]

    st.success(
        f"Validation-selected model: {selected_model}"
    )

    st.caption(
        "Model selection was based on validation MAE. "
        "Test results are reported separately for "
        "final evaluation."
    )


# =====================================================
# TAB 3 - HOUSE PRICE PREDICTION
# =====================================================

with tab3:

    st.header("Interactive House Price Prediction")

    st.write(
        "Enter the characteristics of a house "
        "to compare predictions from the trained models."
    )

    locations = sorted(
        housing["location"].dropna().unique().tolist()
    )

    col1, col2 = st.columns(2)

    with col1:

        location = st.selectbox(
            "Property Location",
            locations
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    with col2:

        house_size = st.number_input(
            "House Size (Square Feet)",
            min_value=300.0,
            max_value=30000.0,
            value=2000.0,
            step=100.0
        )

        land_size = st.number_input(
            "Land Size (Perches)",
            min_value=1.0,
            max_value=500.0,
            value=10.0,
            step=0.5
        )

        selected_model = st.selectbox(
            "Prediction Model",
            [
                "Stacking",
                "Equal-Weight Ensemble",
                "Hybrid RF + XGBoost",
                "XGBoost",
                "CatBoost",
                "Random Forest",
                "Baseline"
            ]
        )

    if st.button(
        "Predict House Price",
        type="primary"
    ):

        input_data = pd.DataFrame([{
            "location": location,
            "bedrooms": float(bedrooms),
            "bathrooms": float(bathrooms),
            "house_size_sqft": float(house_size),
            "land_size_perches": float(land_size)
        }])

        predictions = predict_house(input_data)

        result = predictions[selected_model]

        st.success("Prediction completed!")

        st.metric(
            "Estimated Advertised House Price",
            f"Rs {result:,.2f} Million"
        )

        st.caption(
            f"Approximately Rs {result * 1_000_000:,.0f}. "
            "Historical-data estimate, not a property valuation."
        )

        st.subheader("Predictions From All Models")

        prediction_table = pd.DataFrame([
            {
                "Model": name,
                "Predicted Price (LKR Millions)": value
            }
            for name, value in predictions.items()
        ])

        st.dataframe(
            prediction_table.round(3),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            prediction_table,
            x="Model",
            y="Predicted Price (LKR Millions)",
            title="House Price Predictions by Model"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =====================================================
# TAB 4 - MODEL INSIGHTS
# =====================================================

with tab4:

    st.header("Machine Learning Model Insights")

    st.subheader("Ensemble Techniques")

    st.markdown("""
    **Bagging — Random Forest**

    Multiple Decision Trees are trained using
    randomized data and feature selection.
    Their predictions are averaged.

    **Boosting — XGBoost and CatBoost**

    Models are trained sequentially to improve
    the combined prediction and reduce errors.

    **Stacking — Ridge Meta-Learner**

    Random Forest, XGBoost, and CatBoost predictions
    are combined using a Ridge Regression model
    trained on out-of-fold predictions.

    **Hybrid Ensemble**

    Predictions from multiple models are combined
    using fixed averaging weights.
    """)

    st.subheader("Stacking Meta-Learner Coefficients")

    meta = models["Stacking Meta"]

    coefficients = pd.DataFrame({
        "Model": metadata["base_model_order"],
        "Coefficient": meta.coef_
    })

    fig = px.bar(
        coefficients,
        x="Model",
        y="Coefficient",
        title="Ridge Meta-Learner Coefficients"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "These coefficients are not percentages "
        "or direct measures of feature importance."
    )

    st.subheader("Interpretation")

    st.markdown("""
    - Random Forest provides a bagging-based
      comparison model.
    - XGBoost and CatBoost demonstrate boosting.
    - Stacking was selected using validation MAE.
    - Equal-weight averaging is a competitive
      alternative with simpler combination logic.
    - Differences between validation and test
      results show why evaluation on unseen data
      is important.
    """)

    st.warning(
        "Limitations: The dataset contains historical "
        "asking prices, not final transaction values. "
        "Some repeated advertisements may remain, "
        "and predictions may be inaccurate for "
        "unusual properties or changed market conditions."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Sri Lankan Ensemble Learning Dashboard | "
    "Python • Scikit-learn • XGBoost • CatBoost • Streamlit"
)
