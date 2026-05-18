import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv(
    "data/processed/final_model_dataset.csv"
)

print("Dataset loaded!")

# -------------------------
# Features & Target
# -------------------------

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

# -------------------------
# Encode Target
# -------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# -------------------------
# Train/Test Split
# -------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
)

# -------------------------
# Train Model
# -------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model trained!")

# -------------------------
# SHAP Explainer
# -------------------------

explainer = shap.TreeExplainer(
    model
)

shap_values = (
    explainer.shap_values(
        X_test
    )
)

print("SHAP values created!")

# -------------------------
# SHAP Summary Plot
# -------------------------

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "visuals/shap_feature_importance.png",
    dpi=300
)

plt.show()

print(
    "\nSaved:"
)

print(
    "visuals/shap_feature_importance.png"
)