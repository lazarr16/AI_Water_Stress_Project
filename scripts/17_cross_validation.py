import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder
)

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/processed/final_model_dataset.csv"
)

print("Dataset loaded!")

# -------------------------
# Features & Target
# -------------------------

feature_columns = [

    # Climate
    "avg_temperature",
    "avg_humidity",
    "precipitation",

    # Infrastructure
    "data_center_count",
    "estimated_mw_capacity",
    "hyperscaler_presence",

    # Engineered Features
    "cooling_severity_index_v2",
    "ai_infrastructure_pressure_score"
]

X = df[feature_columns]

y = df["water_stress_risk"]

# -------------------------
# Encode target
# -------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nClasses:")
print(encoder.classes_)

# -------------------------
# Model
# -------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

# -------------------------
# 5-Fold Cross Validation
# -------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

# -------------------------
# Results
# -------------------------

print("\nCross Validation Scores:")
print(scores)

print("\nAverage Accuracy:")
print(round(scores.mean(), 4))

print("\nStandard Deviation:")
print(round(scores.std(), 4))