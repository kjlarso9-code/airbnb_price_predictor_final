import os
import json
import requests
import streamlit as st

# -----------------------------
# Config: Databricks endpoint
# -----------------------------
# These are read from environment variables set in Render (or locally).
# Do NOT hard-code the URL or token in the code.
DATABRICKS_ENDPOINT_URL = os.getenv("DATABRICKS_ENDPOINT_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_ENDPOINT_URL or not DATABRICKS_TOKEN:
    st.error(
        "Endpoint URL or token not set. "
        "Please configure DATABRICKS_ENDPOINT_URL and DATABRICKS_TOKEN "
        "as environment variables in Render."
    )
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# Must match the Databricks model input schema
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "latitude",
    "longitude",
    "room_type",
    "neighbourhood_cleansed",
]


def call_databricks_model(row_dict: dict) -> float:
    """
    Send a single-row prediction request to Databricks model serving.

    The deployed model returns the predicted nightly price in dollars.
    """
    # Arrange into dataframe_split format expected by MLflow pyfunc models
    data_row = [[row_dict[col] for col in FEATURE_COLUMNS]]

    payload = {
        "dataframe_split": {
            "columns": FEATURE_COLUMNS,
            "data": data_row,
        }
    }

    resp = requests.post(
        DATABRICKS_ENDPOINT_URL,
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Request failed with status {resp.status_code}: {resp.text}"
        )

    try:
        resp_json = resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not decode JSON response: {e}")

    # Databricks MLflow serving usually returns {"predictions": [value, ...]}
    try:
        prediction = float(resp_json["predictions"][0])  # already in dollars
    except Exception:
        raise RuntimeError(f"Unexpected response format: {resp_json}")

    return prediction


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏠")

st.title("🏠 Airbnb Price Predictor")
st.write(
    "Enter listing details below and the app will predict the nightly price "
    "using your deployed model on Databricks."
)

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=1, step=1)
    bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, max_value=20.0, value=1.0, step=0.5
    )
    accommodates = st.number_input(
        "Accommodates", min_value=1, max_value=20, value=2, step=1
    )
    minimum_nights = st.number_input(
        "Minimum nights", min_value=1, max_value=365, value=1, step=1
    )
    number_of_reviews = st.number_input(
        "Number of reviews", min_value=0, max_value=1000, value=10, step=1
    )

with col2:
    review_scores_rating = st.number_input(
        "Review score rating (1–5)",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1,
    )
    latitude = st.number_input("Latitude", value=33.45, format="%.6f")
    longitude = st.number_input("Longitude", value=-112.07, format="%.6f")

    room_type = st.selectbox(
        "Room type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room",
            "Hotel room",
        ],
    )

    neighbourhood_cleansed = st.text_input(
        "Neighbourhood (cleansed)", value="San Diego"
    )

if st.button("Predict price"):
    input_row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood_cleansed,
    }

    with st.spinner("Calling model on Databricks..."):
        try:
            predicted_price = call_databricks_model(input_row)
        except Exception as e:
            st.error(f"Error while calling endpoint: {e}")
        else:
            st.success(f"Predicted nightly price: **${predicted_price:,.2f}**")
            st.caption("Best ensemble model served via Databric
