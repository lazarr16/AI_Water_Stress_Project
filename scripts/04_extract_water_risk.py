import pandas as pd

# Load dataset
file_path = "data/external/aqueduct_water_risk.csv"

print("Loading water risk dataset...")

df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded!")

# Replace placeholder missing values
df.replace(-9999, pd.NA, inplace=True)

# Keep only useful columns
selected_columns = [
    "name_0",          # Country
    "name_1",          # State/Province
    "area_km2",

    # Core water stress indicators
    "bws_score",       # Baseline water stress
    "bwd_score",       # Water depletion
    "iav_score",       # Seasonal variability
    "sev_score",       # Drought severity

    # Labels (human-readable)
    "bws_label",
    "bwd_label",
    "iav_label",
    "sev_label"
]

water_df = df[selected_columns].copy()

# Countries we care about
target_countries = [
    "India",
    "United States",
    "Singapore",
    "Ireland",
    "Netherlands"
]

filtered_df = water_df[
    water_df["name_0"].isin(target_countries)
]

# Save cleaned subset
filtered_df.to_csv(
    "data/processed/water_risk_filtered.csv",
    index=False
)

# Show results
print("\nFiltered Dataset Shape:")
print(filtered_df.shape)

print("\nCountries Present:")
print(filtered_df["name_0"].unique())

print("\nSample Rows:")
print(filtered_df.head(20))

print("\nIndian Regions:")
print(
    filtered_df[
        filtered_df["name_0"] == "India"
    ]["name_1"].dropna().unique()
)