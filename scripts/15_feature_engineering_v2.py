import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/processed/model_dataset_v2.csv"
)

print("Dataset loaded!")

# -------------------------
# Feature 1:
# Cooling Severity Index V2
# -------------------------

df["cooling_severity_index_v2"] = (
    df["avg_temperature"]
    +
    (df["avg_humidity"] / 10)
    -
    (df["precipitation"] / 1000)
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
# Feature 3:
# Infrastructure Pressure
# -------------------------

scaler = MinMaxScaler()

infra_cols = [
    "data_center_count",
    "estimated_mw_capacity",
    "hyperscaler_presence"
]

infra_scaled = scaler.fit_transform(
    df[infra_cols]
)

infra_scaled_df = pd.DataFrame(
    infra_scaled,
    columns=infra_cols
)

df["ai_infrastructure_pressure_score"] = (
    infra_scaled_df.mean(axis=1)
)

# -------------------------
# Feature 4:
# Sustainability Suitability
# -------------------------

# Normalize CSI + AWVI
score_scaler = MinMaxScaler()

normalized_scores = (
    score_scaler.fit_transform(
        df[
            [
                "cooling_severity_index_v2",
                "ai_water_vulnerability_index",
                "ai_infrastructure_pressure_score"
            ]
        ]
    )
)

normalized_df = pd.DataFrame(
    normalized_scores,
    columns=[
        "norm_csi",
        "norm_awvi",
        "norm_aips"
    ]
)

# Higher = better
df["sustainability_suitability_score"] = (
    100
    -
    (
        (
            normalized_df["norm_csi"]
            +
            normalized_df["norm_awvi"]
            +
            normalized_df["norm_aips"]
        )
        / 3
        * 100
    )
)

# -------------------------
# Risk Classification
# -------------------------

low_threshold = (
    df[
        "ai_water_vulnerability_index"
    ]
    .quantile(0.33)
)

high_threshold = (
    df[
        "ai_water_vulnerability_index"
    ]
    .quantile(0.66)
)

def classify_risk(score):

    if score <= low_threshold:
        return "Low"

    elif score <= high_threshold:
        return "Medium"

    return "High"

df["water_stress_risk"] = (
    df[
        "ai_water_vulnerability_index"
    ]
    .apply(classify_risk)
)

# -------------------------
# Save output
# -------------------------

df.to_csv(
    "data/processed/"
    "final_model_dataset.csv",
    index=False
)

# -------------------------
# Validation
# -------------------------

print("\nShape:")
print(df.shape)

print("\nNew Features Added:")
print([
    "cooling_severity_index_v2",
    "ai_water_vulnerability_index",
    "ai_infrastructure_pressure_score",
    "sustainability_suitability_score",
    "water_stress_risk"
])

print("\nRisk Distribution:")
print(
    df["water_stress_risk"]
    .value_counts()
)

print("\nPreview:")
print(
    df[
        [
            "city",
            "cooling_severity_index_v2",
            "ai_water_vulnerability_index",
            "ai_infrastructure_pressure_score",
            "sustainability_suitability_score",
            "water_stress_risk"
        ]
    ]
    .head(20)
)