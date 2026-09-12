import os
import re
import joblib
import streamlit as st
import pandas as pd


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "spam_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tfidf_vectorizer.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "SMSSpamCollection"
)


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return data.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------
# Main Application
# --------------------------------------------------

st.title("📧 Email / SMS Spam Detection")

st.markdown(
    """
    ### 🤖 Machine Learning Spam Classifier

    Enter a message below and the trained machine learning
    model will classify it as **Ham (Legitimate)** or
    **Spam**.
    """
)


# --------------------------------------------------
# Check Required Files
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Spam classifier model not found."
    )

    st.stop()


if not os.path.exists(VECTORIZER_PATH):

    st.error(
        "❌ TF-IDF vectorizer not found."
    )

    st.stop()


if not os.path.exists(DATA_PATH):

    st.warning(
        "⚠️ Dataset file was not found. "
        "Prediction may still work."
    )


# --------------------------------------------------
# Load Resources
# --------------------------------------------------

model, vectorizer = load_model()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("📊 Project Information")

st.sidebar.markdown(
    """
    **Task:** Email Spam Detection

    **Technique:** NLP + Machine Learning

    **Feature Extraction:** TF-IDF

    **Models:**
    - Multinomial Naive Bayes
    - Logistic Regression

    **Classes:**
    - Ham
    - Spam
    """
)


# --------------------------------------------------
# Message Input
# --------------------------------------------------

st.subheader("📝 Enter Your Message")

message = st.text_area(
    "Type or paste an email/SMS message:",
    height=180,
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize. Call now!"
    )
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Detect Spam",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message first."
        )

    else:

        cleaned_message = clean_text(message)

        message_features = vectorizer.transform(
            [cleaned_message]
        )

        prediction = model.predict(
            message_features
        )[0]


        # Probability if supported
        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                message_features
            )[0]

            probability = max(probabilities)


        # ------------------------------------------
        # Display Result
        # ------------------------------------------

        if prediction == 1:

            st.error(
                "🚨 SPAM MESSAGE DETECTED"
            )

            st.write(
                "This message has been classified as Spam."
            )

        else:

            st.success(
                "✅ HAM — LEGITIMATE MESSAGE"
            )

            st.write(
                "This message has been classified as Ham."
            )


        # ------------------------------------------
        # Confidence
        # ------------------------------------------

        if probability is not None:

            st.metric(
                "Prediction Confidence",
                f"{probability * 100:.2f}%"
            )


# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.divider()

st.subheader("📊 Dataset Overview")

try:

    data = load_dataset()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Messages",
            len(data)
        )

    with col2:

        spam_count = (
            data["label"] == "spam"
        ).sum()

        st.metric(
            "Spam Messages",
            spam_count
        )

    with col3:

        ham_count = (
            data["label"] == "ham"
        ).sum()

        st.metric(
            "Ham Messages",
            ham_count
        )

except Exception:

    st.info(
        "Dataset statistics are unavailable."
    )


# --------------------------------------------------
# Sample Messages
# --------------------------------------------------

st.subheader("💡 Try These Examples")

example_col1, example_col2 = st.columns(2)

with example_col1:

    st.info(
        """
        **Possible Spam**

        "Congratulations! You have won a free
        prize. Call now to claim your reward!"
        """
    )


with example_col2:

    st.info(
        """
        **Possible Ham**

        "Hey, are we still meeting for dinner
        tonight?"
        """
    )


# --------------------------------------------------
# About
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ About This Project")

st.markdown(
    """
    This application uses Natural Language Processing
    and Machine Learning to classify text messages.

    **Workflow:**

    Text Message → Text Cleaning → TF-IDF →
    Machine Learning Model → Spam/Ham Prediction

    The model was trained using the SMS Spam Collection
    dataset and evaluated using Accuracy, Precision,
    Recall and F1-Score.
    """
)