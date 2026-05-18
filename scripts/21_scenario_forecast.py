import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/processed/final_model_dataset.csv"
)

print("Dataset loaded!")

# =====================================
# FEATURES USED IN MODEL
# =====================================

feature_columns = [
    "avg_temperature",
    "avg_humidity",
    "precipitation",
    "data_center_count",
    "estimated_mw_capacity",
    "hyperscaler_presence",
    "cooling_severity_index_v2",
    "ai_infrastructure_pressure_score"
]

# =====================================
# TRAIN MODEL
# =====================================

X = df[feature_columns]
y = df["water_stress_risk"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y_encoded)

print("Model trained!")

# =====================================
# SELECT CITY
# =====================================

city_name = "Mumbai, India"

city_df = df[
    df["city"] == city_name
].copy()

latest = city_df.sort_values(
    "year"
).iloc[-1]

print(f"\nScenario Forecast: {city_name}")

# =====================================
# FUTURE SIMULATION
# =====================================

future_predictions = []

temperature = latest["avg_temperature"]
humidity = latest["avg_humidity"]
precipitation = latest["precipitation"]

data_centers = latest["data_center_count"]
mw_capacity = latest["estimated_mw_capacity"]
hyperscaler = latest["hyperscaler_presence"]

for year in range(2025, 2031):

    # yearly growth assumptions
    data_centers *= 1.10
    mw_capacity *= 1.12

    # recalculate engineered features
    cooling_index = (
        temperature
        + (humidity / 10)
        - (precipitation / 1000)
    )

    infra_pressure = (
        (
            data_centers / 100
        )
        +
        (
            mw_capacity / 500
        )
        +
        hyperscaler
    ) / 3

    input_data = pd.DataFrame([{
        "avg_temperature":
            temperature,

        "avg_humidity":
            humidity,

        "precipitation":
            precipitation,

        "data_center_count":
            data_centers,

        "estimated_mw_capacity":
            mw_capacity,

        "hyperscaler_presence":
            hyperscaler,

        "cooling_severity_index_v2":
            cooling_index,

        "ai_infrastructure_pressure_score":
            infra_pressure
    }])

    pred = model.predict(
        input_data
    )[0]

    risk = encoder.inverse_transform(
        [pred]
    )[0]

    future_predictions.append({
        "year": year,
        "data_centers":
            round(data_centers, 1),

        "mw_capacity":
            round(mw_capacity, 1),

        "predicted_risk":
            risk
    })

# =====================================
# RESULTS
# =====================================

forecast_df = pd.DataFrame(
    future_predictions
)

print("\nForecast Results:")
print(forecast_df)

# =====================================
# PLOT
# =====================================

risk_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

forecast_df["risk_score"] = (
    forecast_df[
        "predicted_risk"
    ].map(risk_map)
)

plt.figure(figsize=(10,5))

plt.plot(
    forecast_df["year"],
    forecast_df["risk_score"],
    marker="o"
)

plt.yticks(
    [1,2,3],
    ["Low","Medium","High"]
)

plt.title(
    f"Future AI Water Stress Risk: "
    f"{city_name}"
)

plt.xlabel("Year")
plt.ylabel("Predicted Risk")

plt.grid(True)

plt.savefig(
    "visuals/future_risk_forecast.png",
    dpi=300
)

plt.show()