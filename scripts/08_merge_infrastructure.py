import pandas as pd

# -------------------------
# Load datasets
# -------------------------

master_df = pd.read_csv(
    "data/processed/master_dataset_v2.csv"
)

infra_df = pd.read_csv(
    "data/external/data_center_infrastructure.csv"
)

print("Datasets loaded!")

# -------------------------
# Clean city names
# -------------------------

def clean_city_name(city):

    return city.split(",")[0].strip()

master_df["city_clean"] = (
    master_df["city"]
    .apply(clean_city_name)
)

# -------------------------
# Merge infrastructure
# -------------------------

merged_df = master_df.merge(
    infra_df,
    left_on="city_clean",
    right_on="city",
    how="left"
)

# Drop duplicate city column
merged_df.drop(
    columns=["city_y"],
    inplace=True
)

# Rename original city
merged_df.rename(columns={
    "city_x": "city"
}, inplace=True)

# -------------------------
# Save final dataset
# -------------------------

merged_df.to_csv(
    "data/processed/final_master_dataset.csv",
    index=False
)

# -------------------------
# Validation
# -------------------------

print("\nFinal Dataset Shape:")
print(merged_df.shape)

print("\nMissing Values:")
print(
    merged_df.isnull().sum()
)

print("\nPreview:")
print(
    merged_df[
        [
            "city",
            "year",
            "baseline_water_stress",
            "cooling_severity_index",
            "data_center_count",
            "compute_growth_score",
            "estimated_mw_capacity"
        ]
    ].head(15)
)

print("\nCities Present:")
print(
    merged_df["city_clean"]
    .unique()
)