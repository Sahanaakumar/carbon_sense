import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ======================================
# 1. LOAD DATASET
# ======================================

df = pd.read_csv("co2_1950_2026_1000_rows.csv")

print("Original Data:")
print(df.head())

# ======================================
# 2. DATA CLEANING
# ======================================

# Remove missing values
df = df.dropna()

# Convert date column to datetime
df['ds'] = pd.to_datetime(df['date'])

# Rename CO2 column for Prophet
df['y'] = df['co2']

# Keep only required columns
df = df[['ds', 'y']]

print("\nCleaned Data:")
print(df.head())

# ======================================
# 3. TRAIN MODEL
# ======================================

model = Prophet()
model.fit(df)

# ======================================
# 4. CREATE FUTURE DATA
# ======================================

# Predict next 5 years (approx 5*365 days)
future = model.make_future_dataframe(periods=1825)

forecast = model.predict(future)

# ======================================
# 5. GRAPH 1: ACTUAL vs PREDICTED
# ======================================

plt.figure()

# Actual data
plt.plot(df['ds'], df['y'], label="Actual CO2")

# Predicted data
plt.plot(forecast['ds'], forecast['yhat'], label="Predicted CO2")

plt.title("Actual vs Predicted Carbon Emissions")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.legend()
plt.grid()

plt.show()

# ======================================
# 6. GRAPH 2: FUTURE (2026–2030)
# ======================================

future_pred = forecast[
    (forecast['ds'].dt.year >= 2026) &
    (forecast['ds'].dt.year <= 2030)
]

plt.figure()

plt.plot(future_pred['ds'], future_pred['yhat'], marker='o')

plt.title("Future CO2 Prediction (2026–2030)")
plt.xlabel("Year")
plt.ylabel("Predicted CO2 Emissions")
plt.grid()

plt.show()

# ======================================
# 7. PRINT FUTURE VALUES
# ======================================

print("\nFuture Predictions (2026–2030):")
print(future_pred[['ds', 'yhat']])

model.plot_components(forecast)
plt.show()