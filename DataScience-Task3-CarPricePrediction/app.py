import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# File Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "car_price_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Car details v3.csv"
)


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================================================
# Load Dataset
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# =========================================================
# Check Required Files
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Trained model not found.\n\n"
        f"Expected location:\n{MODEL_PATH}"
    )
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(
        "❌ Dataset not found.\n\n"
        f"Expected location:\n{DATA_PATH}"
    )
    st.stop()


model = load_model()
df = load_data()


# =========================================================
# Title
# =========================================================

st.title("🚗 Used Car Price Prediction")

st.markdown(
    """
    ### Machine Learning Based Car Price Estimator

    Enter the details of a used car to estimate its expected
    selling price using a machine learning model trained on
    CarDekho vehicle data.
    """
)

st.divider()


# =========================================================
# Sidebar — Car Details
# =========================================================

st.sidebar.header("🚘 Enter Car Details")


# ---------------------------------------------------------
# Brand
# ---------------------------------------------------------

brands = sorted(
    df["name"]
    .dropna()
    .str.split()
    .str[0]
    .unique()
)

brand = st.sidebar.selectbox(
    "Car Brand",
    brands
)


# ---------------------------------------------------------
# Car Age
# ---------------------------------------------------------

reference_year = int(df["year"].max())

car_age = st.sidebar.slider(
    "Car Age (Years)",
    min_value=0,
    max_value=30,
    value=5,
    step=1
)


# ---------------------------------------------------------
# Kilometers Driven
# ---------------------------------------------------------

km_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=50000,
    step=1000
)


# ---------------------------------------------------------
# Fuel Type
# ---------------------------------------------------------

fuel = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["fuel"].dropna().unique())
)


# ---------------------------------------------------------
# Seller Type
# ---------------------------------------------------------

seller_type = st.sidebar.selectbox(
    "Seller Type",
    sorted(df["seller_type"].dropna().unique())
)


# ---------------------------------------------------------
# Transmission
# ---------------------------------------------------------

transmission = st.sidebar.selectbox(
    "Transmission",
    sorted(df["transmission"].dropna().unique())
)


# ---------------------------------------------------------
# Owner
# ---------------------------------------------------------

owner = st.sidebar.selectbox(
    "Owner",
    sorted(df["owner"].dropna().unique())
)


# =========================================================
# Prediction Button
# =========================================================

predict_button = st.sidebar.button(
    "🔮 Predict Car Price",
    use_container_width=True
)


# =========================================================
# Prediction
# =========================================================

if predict_button:

    input_data = pd.DataFrame({
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "car_age": [car_age],
        "brand": [brand]
    })

    try:

        prediction = model.predict(input_data)[0]

        st.success("✅ Prediction generated successfully!")

        st.subheader("💰 Estimated Selling Price")

        st.metric(
            label="Predicted Car Price",
            value=f"₹ {prediction:,.0f}"
        )

        st.divider()

        st.subheader("📋 Vehicle Information")

        display_data = pd.DataFrame({
            "Parameter": [
                "Car Brand",
                "Car Age",
                "Kilometers Driven",
                "Fuel Type",
                "Seller Type",
                "Transmission",
                "Owner"
            ],
            "Value": [
                brand,
                f"{car_age} years",
                f"{km_driven:,} km",
                fuel,
                seller_type,
                transmission,
                owner
            ]
        })

        st.table(display_data)

    except Exception as e:

        st.error(
            f"❌ Prediction failed:\n\n{str(e)}"
        )


# =========================================================
# Dataset Overview
# =========================================================

st.divider()

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Original Features",
        len(df.columns)
    )

with col3:
    st.metric(
        "Reference Year",
        reference_year
    )


# =========================================================
# About the Project
# =========================================================

st.divider()

st.subheader("ℹ️ About This Application")

st.markdown(
    """
    This application uses a machine learning regression model
    trained on used-car data from CarDekho.

    The prediction considers:

    - 🚘 Car brand
    - 📅 Car age
    - 🛣️ Kilometers driven
    - ⛽ Fuel type
    - 👤 Seller type
    - ⚙️ Transmission
    - 👥 Ownership history

    The trained model was evaluated using **MAE, RMSE, and R²**
    during the machine learning analysis.
    """
)

st.caption(
    "Car Price Prediction | OASIS INFOBYTE Data Science Internship | Task 3"
)