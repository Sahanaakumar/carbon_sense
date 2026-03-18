from fastapi import FastAPI
import pandas as pd
from prophet import Prophet

# ======================================
# CREATE FASTAPI APP (IMPORTANT)
# ======================================
app = FastAPI()

# ======================================
# LOAD DATASET
# ======================================
df = pd.read_csv("co2_1950_2026_1000_rows.csv")

# ======================================
# DATA CLEANING
# ======================================
df = df.dropna()

# Convert date column
df['ds'] = pd.to_datetime(df['date'])

# Rename for Prophet
df['y'] = df['co2']

# Keep required columns
df = df[['ds', 'y']]

# ======================================
# TRAIN MODEL
# ======================================
model = Prophet()
model.fit(df)

print("✅ Model trained successfully")

# ======================================
# ROUTES
# ======================================

@app.get("/")
def home():
    return {"message": "CarbonSense API is running 🚀"}

# --------------------------------------

@app.get("/forecast")
def get_forecast():
    future = model.make_future_dataframe(periods=1825)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat']].tail(50)

    return result.to_dict(orient="records")

# --------------------------------------

@app.get("/future")
def future_prediction():
    future = model.make_future_dataframe(periods=1825)
    forecast = model.predict(future)

    future_pred = forecast[
        (forecast['ds'].dt.year >= 2026) &
        (forecast['ds'].dt.year <= 2030)
    ]

    return future_pred[['ds', 'yhat']].to_dict(orient="records")

# --------------------------------------

@app.get("/status")
def status():
    return {
        "model": "Prophet",
        "data_points": len(df),
        "status": "Active ✅"
    }