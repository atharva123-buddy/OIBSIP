import os
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "sales_prediction_model.pkl"
DATA_PATH = BASE_DIR / "data" / "advertising.csv"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Load Resources
# --------------------------------------------------

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📈 Sales Prediction Dashboard")

st.markdown(
    """
    Predict product sales based on advertising expenditure
    across **TV, Radio, and Newspaper** channels.
    """
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("📊 Advertising Budget")

tv_spend = st.sidebar.number_input(
    "TV Advertising Spend",
    min_value=0.0,
    value=100.0,
    step=1.0
)

radio_spend = st.sidebar.number_input(
    "Radio Advertising Spend",
    min_value=0.0,
    value=20.0,
    step=1.0
)

newspaper_spend = st.sidebar.number_input(
    "Newspaper Advertising Spend",
    min_value=0.0,
    value=10.0,
    step=1.0
)

predict_button = st.sidebar.button(
    "🚀 Predict Sales",
    use_container_width=True
)


# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.subheader("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(df)
    )

with col2:
    st.metric(
        "Features",
        3
    )

with col3:
    st.metric(
        "Average Sales",
        f"{df['Sales'].mean():.2f}"
    )

with col4:
    st.metric(
        "Maximum Sales",
        f"{df['Sales'].max():.2f}"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.subheader("🎯 Sales Prediction")

input_data = pd.DataFrame({
    "TV": [tv_spend],
    "Radio": [radio_spend],
    "Newspaper": [newspaper_spend]
})


if predict_button:

    prediction = model.predict(input_data)[0]

    st.success(
        f"### Predicted Sales: {prediction:.2f}"
    )

    st.write("Advertising expenditure used for prediction:")

    st.dataframe(
        input_data,
        use_container_width=True
    )

else:

    st.info(
        "Enter the advertising budget in the sidebar "
        "and click **Predict Sales**."
    )


# --------------------------------------------------
# Advertising Channel Information
# --------------------------------------------------

st.divider()

st.subheader("📊 Advertising Channel Analysis")

channel_averages = df[
    ["TV", "Radio", "Newspaper"]
].mean()

st.bar_chart(channel_averages)

st.caption(
    "Average advertising expenditure across the three channels."
)


# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("🔎 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ Project Information")

st.markdown(
    """
    **Objective:** Predict product sales using advertising expenditure.

    **Machine Learning Models:**
    - Linear Regression
    - Random Forest Regression

    **Evaluation Metrics:**
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score

    **Input Features:**
    - TV advertising
    - Radio advertising
    - Newspaper advertising

    **Target:**
    - Sales
    """
)

st.caption(
    "Data Science Internship Project — OASIS INFOBYTE"
)