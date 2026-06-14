import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import os

# Load data
df = pd.read_csv("data/weather_data.csv")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort by time
df = df.sort_values("Timestamp")

# Create time index
df["time_index"] = np.arange(len(df))

# Features
X = df[[
   "time_index",
   "Humidity",
   "Pressure",
   "Min_Temp_C",
   "Max_Temp_C",
   "Avg_Temp_C"
]]

# Target
y = df["Temperature_C"]

# Train model
model = RandomForestRegressor(
   n_estimators=200,
   random_state=42
)

model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/weather_model.pkl")

# Future prediction (24 hours)

future_steps = 24

future_index = np.arange(
   len(df),
   len(df) + future_steps
)

last_row = df.iloc[-1]

future_data = pd.DataFrame({
   "time_index": future_index,
   "Humidity": [last_row["Humidity"]] * future_steps,
   "Pressure": [last_row["Pressure"]] * future_steps,
   "Min_Temp_C": [last_row["Min_Temp_C"]] * future_steps,
   "Max_Temp_C": [last_row["Max_Temp_C"]] * future_steps,
   "Avg_Temp_C": [last_row["Avg_Temp_C"]] * future_steps
})

future_predictions = model.predict(future_data)

future_dates = pd.date_range(
   start=df["Timestamp"].max(),
   periods=future_steps + 1,
   freq="h"
)[1:]

forecast_df = pd.DataFrame({
   "Timestamp": future_dates,
   "Predicted_Temperature": future_predictions
})

print(forecast_df)

# Graph

plt.figure(figsize=(12,6))

plt.plot(
   df["Timestamp"],
   df["Temperature_C"],
   label="Historical"
)

plt.plot(
   forecast_df["Timestamp"],
   forecast_df["Predicted_Temperature"],
   label="Forecast"
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Forecast")
plt.legend()
plt.grid()

os.makedirs("graphs", exist_ok=True)

plt.savefig("graphs/forecast.png")

plt.show()
