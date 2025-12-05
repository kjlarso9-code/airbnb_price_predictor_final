# Airbnb Price Predictor (Streamlit + Databricks + Render)

This app uses a Neural Network model (`NN_h(64,)_alpha0.0001_lr0.001`) trained in Databricks
and served via Databricks Model Serving. The Streamlit app calls the serving endpoint
and displays the predicted nightly price for an Airbnb listing.

## Tech stack

- Databricks + MLflow for training, tracking, and serving
- Streamlit front-end
- Deployed on Render as a Python Web Service
- Endpoint authentication via Databricks personal access token (PAT)

## How it works

1. User inputs listing features (bedrooms, bathrooms, etc.).
2. Streamlit sends a POST request to the Databricks serving endpoint in `dataframe_split` format.
3. The endpoint returns a predicted price (`{"predictions": [value]}`).
4. The app shows the predicted nightly price.

## Deployment

1. Push this repo to GitHub.
2. Create a new Web Service on Render, connect this repo.
3. Use `render.yaml` (auto-detected) or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Set environment variables:
   - `DATABRICKS_ENDPOINT_URL`
   - `DATABRICKS_TOKEN`
5. Deploy and open the public URL.
