import pandas as pd

# -------------------------
# Load merged dataset
# -------------------------

df = pd.read_csv(
    "data/processed/master_dataset_scaled_v1.csv"
)

print("Dataset loaded!")

# -------------------------
# Keep only reliable regions
# -------------------------

valid_countries = [
    "India",
    "United States",
    "Singapore",
    "Ireland",
    "Netherlands",
    "Australia"
]

curated_df = df[
    df["country"].isin(
        valid_countries
    )
].copy()

# -------------------------
# Remove rows missing
# water features
# -------------------------

water_columns = [
    "baseline_water_stress",
    "water_depletion",
    "seasonal_variability",
    "drought_severity"
]

curated_df = (
    curated_df
    .dropna(
        subset=water_columns
    )
)

# -------------------------
# Save curated dataset
# -------------------------

curated_df.to_csv(
    "data/processed/"
    "curated_model_dataset_v1.csv",
    index=False
)

# -------------------------
# Validation
# -------------------------

print("\nFinal Shape:")
print(curated_df.shape)

print("\nCountries:")
print(
    curated_df["country"]
    .value_counts()
)

print("\nCities:")
print(
    curated_df["city"]
    .nunique()
)

print("\nMissing Values:")
print(
    curated_df.isnull()
    .sum()
)

print("\nSample Cities:")
print(
    curated_df["city"]
    .unique()[:20]
)