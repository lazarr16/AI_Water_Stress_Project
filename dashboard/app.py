import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------
# Page Config
# ------------------------------------------------

st.set_page_config(
    page_title="AI Water Stress Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------------------------
# Load Data (Absolute Paths to prevent FileNotFoundError)
# ------------------------------------------------

current_dir = Path(__file__).resolve().parent

# File 1: ML training dataset using long column names
model_data_path = current_dir.parent / "data" / "processed" / "final_model_dataset.csv"
df = pd.read_csv(model_data_path)

# File 2: Selection profiles using short column names
profiles_path = current_dir.parent / "data" / "processed" / "city_profiles.csv"
city_profiles = pd.read_csv(profiles_path)

# ------------------------------------------------
# Train Model
# ------------------------------------------------

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

X = df[feature_columns]
y = df["water_stress_risk"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X, y_encoded)

# ------------------------------------------------
# Header
# ------------------------------------------------

st.title("AI Water Stress Intelligence Dashboard")

st.markdown("""
Simulate AI data-center expansion risk
using climate + infrastructure analytics.
""")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.header("Simulation Settings")

selected_city = st.sidebar.selectbox(
    "Select City Preset",
    city_profiles["city"].unique()
)

# Pulls matching short-column profile row from city_profiles.csv
city_data = city_profiles[
    city_profiles["city"] == selected_city
].iloc[0]

temperature = st.sidebar.slider(
    "Temperature (°C)",
    0,
    50,
    int(city_data["temperature"])
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    0,
    100,
    int(city_data["humidity"])
)

rainfall = st.sidebar.slider(
    "Annual Rainfall (mm)",
    0,
    7000,
    int(city_data["rainfall"])
)

data_centers = st.sidebar.slider(
    "Data Center Count",
    0,
    100,
    int(city_data["data_centers"])
)

mw_capacity = st.sidebar.slider(
    "MW Capacity",
    0,
    500,
    int(city_data["mw_capacity"])
)

hyperscaler = st.sidebar.selectbox(
    "Hyperscaler Presence",
    [0, 1],
    index=int(city_data["hyperscaler"])
)

# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

cooling_index = (
    temperature
    + (humidity / 10)
    - (rainfall / 1000)
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

sustainability_score = max(
    0,
    100 - (
        cooling_index
        + infra_pressure * 40
    )
)

# ------------------------------------------------
# Prediction
# ------------------------------------------------

# Wraps slider inputs cleanly into the exact long feature layout expected by the model
input_data = pd.DataFrame([{
    "avg_temperature": temperature,
    "avg_humidity": humidity,
    "precipitation": rainfall,
    "data_center_count": data_centers,
    "estimated_mw_capacity": mw_capacity,
    "hyperscaler_presence": hyperscaler,
    "cooling_severity_index_v2": cooling_index,
    "ai_infrastructure_pressure_score": infra_pressure
}])

prediction = model.predict(
    input_data
)[0]

predicted_class = (
    encoder.inverse_transform(
        [prediction]
    )[0]
)

# ------------------------------------------------
# KPI Section
# ------------------------------------------------

st.subheader(
    f"📍 Region Analysis: {selected_city}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted Risk",
        predicted_class
    )

with col2:
    st.metric(
        "Cooling Severity",
        round(cooling_index, 1)
    )

with col3:
    st.metric(
        "Infra Pressure",
        round(infra_pressure, 2)
    )

with col4:
    st.metric(
        "Sustainability",
        f"{round(sustainability_score,1)}/100"
    )

# ------------------------------------------------
# Risk Messaging
# ------------------------------------------------

if predicted_class == "High":

    st.error(
        "🚨 HIGH RISK: "
        "Region may experience "
        "water stress pressure "
        "from AI expansion."
    )

elif predicted_class == "Medium":

    st.warning(
        "⚠️ MEDIUM RISK: "
        "Expansion feasible "
        "with mitigation "
        "strategies."
    )

else:

    st.success(
        "✅ LOW RISK: "
        "Suitable candidate "
        "for sustainable "
        "AI infrastructure."
    )

# ------------------------------------------------
# Why Prediction
# ------------------------------------------------

st.subheader(
    "Why this prediction?"
)

insights = []

if humidity > 75:
    insights.append(
        "High humidity may "
        "increase cooling burden."
    )

if temperature > 28:
    insights.append(
        "High temperature "
        "raises cooling demand."
    )

if rainfall < 500:
    insights.append(
        "Low rainfall suggests "
        "higher water scarcity risk."
    )

if hyperscaler == 1:
    insights.append(
        "Hyperscaler presence "
        "increases infrastructure pressure."
    )

for item in insights:
    st.write(f"• {item}")

# ------------------------------------------------
# Recommended Regions
# ------------------------------------------------

st.subheader(
    "Recommended Alternatives"
)

recommendations = [
    "Amsterdam",
    "Dublin",
    "Seattle"
]

for city in recommendations:
    st.write(f" {city}")

# ------------------------------------------------
# Footer
# ------------------------------------------------

st.markdown("---")

st.caption(
    "Built using Machine Learning, "
    "Climate Analytics, and "
    "Sustainability Intelligence"
)