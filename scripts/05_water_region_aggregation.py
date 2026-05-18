import pandas as pd

# Load filtered dataset
df = pd.read_csv(
    "data/processed/water_risk_filtered.csv"
)

# Numeric columns to aggregate
water_metrics = [
    "bws_score",
    "bwd_score",
    "iav_score",
    "sev_score"
]

# Group by country + region
aggregated_df = (
    df.groupby(["name_0", "name_1"])[water_metrics]
    .mean()
    .reset_index()
)

# Rename columns
aggregated_df.rename(columns={
    "name_0": "country",
    "name_1": "state_region",
    "bws_score": "baseline_water_stress",
    "bwd_score": "water_depletion",
    "iav_score": "seasonal_variability",
    "sev_score": "drought_severity"
}, inplace=True)

# Save processed dataset
aggregated_df.to_csv(
    "data/processed/water_risk_aggregated.csv",
    index=False
)

print("\nAggregated Dataset Shape:")
print(aggregated_df.shape)

print("\nFirst 20 Rows:")
print(aggregated_df.head(20))

print("\nIndia Sample:")
print(
    aggregated_df[
        aggregated_df["country"] == "India"
    ].head(15)
)