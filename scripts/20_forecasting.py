import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    "data/processed/final_model_dataset.csv"
)

print("Dataset loaded!")

# ==========================================
# Choose City
# ==========================================

city_name = "Mumbai, India"

city_df = df[
    df["city"] == city_name
].copy()

print(
    f"\nForecasting for: {city_name}"
)

# ==========================================
# Prepare Data
# Prophet format:
# ds = date
# y = target
# ==========================================

city_df["date"] = pd.to_datetime(
    city_df["year"].astype(str)
)

forecast_df = city_df[[
    "date",
    "sustainability_suitability_score"
]].rename(columns={
    "date": "ds",
    "sustainability_suitability_score": "y"
})

print("\nForecast Data:")
print(forecast_df)

# ==========================================
# Train Prophet
# ==========================================

model = Prophet(
    yearly_seasonality=False,
    daily_seasonality=False,
    weekly_seasonality=False
)

model.fit(
    forecast_df
)

print("\nModel trained!")

# ==========================================
# Future Years
# ==========================================

future = model.make_future_dataframe(
    periods=6,
    freq="Y"
)

forecast = model.predict(
    future
)

# ==========================================
# Show Forecast
# ==========================================

print("\nForecast Output:")
print(
    forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ].tail(10)
)

# ==========================================
# Plot Forecast
# ==========================================

fig = model.plot(
    forecast
)

plt.title(
    f"Sustainability Forecast: "
    f"{city_name}"
)

plt.xlabel("Year")
plt.ylabel(
    "Sustainability Score"
)

plt.show()