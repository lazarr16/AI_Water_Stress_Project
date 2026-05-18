# Climate Intelligence for Sustainable AI Infrastructure

A machine learning and sustainability analytics system designed to predict the impact of AI data center expansion on regional water stress.

---

## Project Overview

AI data centers require massive cooling infrastructure and consume significant water resources. This project evaluates how rapid AI infrastructure growth may influence regional water stress by integrating:

- Climate analytics (NASA POWER API)
- Water vulnerability indicators (Aqueduct Water Risk Atlas)
- AI infrastructure expansion metrics (Data center footprints)
- Machine learning-based risk prediction (Random Forest & XGBoost)
- Scenario forecasting (2025-2030 predictions)

The system predicts water stress risk levels for global AI infrastructure regions and identifies sustainable expansion zones.

---

## Objectives

- Predict regional water stress risks caused by AI data center growth.
- Identify sustainable regions for AI infrastructure expansion.
- Develop explainable machine learning models using SHAP values.
- Forecast future infrastructure sustainability scenarios (2025-2030).
- Build an interactive sustainability intelligence dashboard.

---

## Dataset Sources

### NASA POWER Climate API
Used for extracting localized:
- Temperature
- Humidity
- Precipitation

### Aqueduct Water Risk Atlas
Used for retrieving localized:
- Baseline Water Stress
- Water Depletion
- Seasonal Variability
- Drought Severity

### AI Infrastructure Indicators
Custom-engineered features representing local demand:
- Data Center Count
- Estimated MW Capacity
- Hyperscaler Presence
- Infrastructure Pressure Score

---

## Machine Learning

### Models Used
- Random Forest Classifier
- XGBoost Classifier

### Prediction Classes
- Low Risk
- Medium Risk
- High Risk

### Model Performance
- Random Forest Accuracy: 91-97%
- Cross-Validation Accuracy: 94%+

### Explainability
- SHAP (SHapley Additive exPlanations): Utilized for global and local feature importance mapping to understand model decisions.

---

## Forecasting

### Scenario-Based Forecasting (2025-2030)
Simulates future compounding AI infrastructure growth corridors under custom scenarios:
- Baseline Growth: +10% compound yearly data center count growth.
- Capacity Surge: +12% compound yearly Megawatt (MW) capacity scaling.

Predicts shifts in regional water stress risk classifications over a 5-year outlook window.

---

## Geospatial Intelligence

An interactive world map built using Folium dynamically visualizes risk layers:
- Low Risk Regions: Highly suitable for sustainable infrastructure.
- Medium Risk Regions: Feasible with strategic local mitigation.
- High Risk Regions: High environmental baseline vulnerability; expansion not advised.

---

## Interactive Dashboard

Built using Streamlit to provide a front-end playground featuring:
- Risk prediction simulator: Real-time calculation using interactive sidebar sliders.
- Sustainability scoring: A custom index combining climate load and grid pressure.
- Scenario testing: Dynamic adjustment of local infrastructure variables.
- Regional comparison: Side-by-side comparison matrix of specific global tech hubs.

---

## Tech Stack

- Data Manipulation: Python, Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost, SHAP
- Visualization & Deployment: Streamlit, Folium, Matplotlib

---

## Project Structure

```text
AI_Water_Stress_Project/
│
├── dashboard/          # Interactive Streamlit interface (app.py)
├── data/               # Project data assets
│   ├── external/       # Third-party baseline environmental records
│   ├── processed/      # Integrated modeling sets (final_model_dataset.csv, city_profiles.csv)
│   └── raw/            # Ingested API outputs and coordinates
├── models/             # Serialized trained model components (.pkl format)
├── notebooks/          # Exploratory analysis and structural prototypes
├── reports/            # Project outcomes and validation documentation
├── scripts/            # Modular pipeline generation (01_ to 19_shap_explainability.py)
├── visuals/            # Extracted SHAP dependencies, forecast graphs, and map visuals
├── README.md           # Main landing overview document
└── requirements.txt    # Project dependencies manifesto