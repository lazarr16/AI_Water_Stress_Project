import pandas as pd
import numpy as np

# -------------------------
# Load master dataset
# -------------------------

df = pd.read_csv(
    "data/processed/master_dataset_v1.csv"
)

print("Dataset Loaded!")
print(df.shape)

# -------------------------
# Feature 1:
# Cooling Severity Index
# -------------------------

df["cooling_severity_index"] = (
    (
        df["avg_temperature"]
        * df["avg_humidity"]
    )
    /
    (df["precipitation"] + 1)
)

# -------------------------
# Feature 2:
# AI Water Vulnerability Index
# -------------------------

df["ai_water_vulnerability_index"] = (
    df[
        [
            "baseline_water_stress",
            "water_depletion",
            "seasonal_variability",
            "drought_severity"
        ]
    ]
    .mean(axis=1)
)

# -------------------------
# Dynamic Risk Classification
# -------------------------

low_threshold = (
    df["ai_water_vulnerability_index"]
    .quantile(0.33)
)

high_threshold = (
    df["ai_water_vulnerability_index"]
    .quantile(0.66)
)

def classify_risk(score):

    if score <= low_threshold:
        return "Low"

    elif score <= high_threshold:
        return "Medium"

    return "High"


df["water_stress_risk"] = (
    df["ai_water_vulnerability_index"]
    .apply(classify_risk)
)

# -------------------------
# Ranking Score
# -------------------------

df["sustainability_score"] = (
    100
    -
    (
        df["ai_water_vulnerability_index"]
        * 20
    )
)

# Clip values to 0–100
df["sustainability_score"] = (
    df["sustainability_score"]
    .clip(0, 100)
)

# -------------------------
# Save processed dataset
# -------------------------

df.to_csv(
    "data/processed/master_dataset_v2.csv",
    index=False
)

# -------------------------
# Preview
# -------------------------

print("\nNew Features Added!")

print("\nColumns:")
print(df.columns.tolist())

print("\nSample Output:")
print(
    df[
        [
            "city",
            "year",
            "cooling_severity_index",
            "ai_water_vulnerability_index",
            "water_stress_risk",
            "sustainability_score"
        ]
    ].head(15)
)

print("\nRisk Distribution:")
print(
    df["water_stress_risk"]
    .value_counts()
)