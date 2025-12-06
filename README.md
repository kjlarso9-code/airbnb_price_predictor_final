# 🏡 Airbnb Price Predictor

Machine Learning · Streamlit App · Databricks · End-to-End Deployment

---

## 📌 Project Overview

This project predicts nightly Airbnb prices for **San Diego** listings using machine learning.

It includes:

- Data cleaning & preprocessing in **Databricks**
- Feature engineering & encoding
- Model selection & hyperparameter tuning
- Experiment tracking with **MLflow**
- Deploying the best model via **Databricks Model Serving**
- A **Streamlit** web app (connected to GitHub) that calls the deployed model

The final deliverable is a fully deployed interactive application that allows users to estimate Airbnb prices based on property characteristics.

---

## 🎯 Business Problem

Airbnb hosts must decide what nightly price to charge.

If the price is:

- **Too low** → revenue is lost  
- **Too high** → bookings decrease  

This project aims to help answer:

- *What should a host charge per night for a given listing?*
- *Which features have the strongest impact on price?*
- *How do reviews, room type, and neighborhood affect value?*

Accurate, data-driven pricing helps hosts maximize both occupancy and profit.

---

## 🧹 Data Cleaning & Preparation

All data preparation was done in Databricks.

Main steps:

- Removed rows with missing/invalid values in key fields  
  (e.g., `bedrooms`, `bathrooms`, `review_scores_rating`, `price`)
- Converted `price` from string → numeric  
  (e.g., `$174.00` → `174.0`)
- Filtered out extreme price outliers to focus on realistic nightly rates
- Split data into **train / test** sets
- Built a preprocessing pipeline with:
  - **Numeric features** (passed through with imputation if needed):
    - `bedrooms`
    - `bathrooms`
    - `accommodates`
    - `minimum_nights`
    - `number_of_reviews`
    - `review_scores_rating`
    - `latitude`
    - `longitude`
  - **Categorical features** (one-hot encoded):
    - `room_type`
    - `neighbourhood_cleansed`

These are the same features exposed in the web app UI.

---

## 🤖 Model Development

Multiple regression models were trained and logged to MLflow, including:

- Regression Tree
- Linear / Ridge Regression
- Support Vector Regression (SVR)
- Neural Network (MLPRegressor)
- Random Forest Regression
- XGBoost Regression
- k-Nearest Neighbors Regression (KNN)
- An **ensemble model** combining tree-based models

For each model family, key hyperparameters were tuned (e.g., tree depth, number of estimators, regularization strength, learning rate, number of neighbors, etc.).  

Performance was evaluated on a held-out test set, and metrics such as **MAE**, **RMSE**, and **R²** were tracked in MLflow.

### Final Model

The best performance came from an **ensemble model** based on tree-based learners (Random Forest + XGBoost, with KNN in some configurations). This model achieved:

- Lowest MAE / RMSE on the test set
- Highest R² among all tested models
- Robust performance on tabular data with nonlinear relationships

This best model was registered in MLflow and deployed as a **Databricks Model Serving endpoint**.

---

## 🖥 Deployment (Streamlit App)

The Streamlit application:

- Collects listing details from the user:
  - Bedrooms, bathrooms, accommodates, minimum nights
  - Number of reviews, review score rating
  - Latitude, longitude
  - Room type, neighbourhood
- Builds a single-row payload in `dataframe_split` format
- Sends a POST request to the Databricks **serving endpoint** using:
  - `DATABRICKS_ENDPOINT_URL`
  - `DATABRICKS_TOKEN`
- Receives the model’s prediction (`{"predictions": [value]}`)
- Displays the estimated nightly price in the UI

Secrets (endpoint URL and token) are provided via environment variables, not hard-coded.

---

## 🔗 Live App

Deployed app:  
**[https://airbnbpricepredictor-9hepznyari5xotvfaz6z7e.streamlit.app/](https://airbnb-price-predictor-final.onrender.com/)**


---

## 📁 Repository Structure

```text
airbnb_price_predictor/
├── app.py           # Streamlit application
├── requirements.txt # Python dependencies
├── render.yaml      # Render deployment configuration
├── README.md        # Project documentation
└── notebooks/       # (Optional) exported Databricks notebooks

## 🧠 How the App Works

User enters listing details in the Streamlit form.
The app assembles the inputs into the correct feature order.
A JSON payload in dataframe_split format is sent to the Databricks Model Serving endpoint.
The deployed model (including preprocessing + ensemble) returns a predicted price.
The app displays the estimated nightly price to the user.

## 🙋‍♀️ Author

Kendall Larson
CIS 508 – Term Project
Instructor: Sang-Pil Han
