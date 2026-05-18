import pandas as pd

# -------------------------
# Load datasets
# -------------------------

climate_df = pd.read_csv(
    "data/raw/nasa_climate_data.csv"
)

water_df = pd.read_csv(
    "data/processed/water_risk_aggregated.csv"
)

# -------------------------
# City → State mapping
# -------------------------

city_state_map = {
    # India
    "Mumbai, India": "Maharashtra",
    "Pune, India": "Maharashtra",
    "Bengaluru, India": "Karnataka",
    "Hyderabad, India": "Telangana",
    "Chennai, India": "Tamil Nadu",

    # USA
    "Phoenix, USA": "Arizona",
    "Dallas, USA": "Texas",

    # Singapore
    "Singapore": "Central",

    # Ireland
    "Dublin, Ireland": "Dublin",

    # Netherlands
    "Amsterdam, Netherlands": "Noord-Holland"
}

# Add state column
climate_df["state_region"] = (
    climate_df["city"]
    .map(city_state_map)
)

# -------------------------
# Country extraction
# -------------------------

def extract_country(city):
    if "," in city:
        return city.split(",")[-1].strip()

    return city

climate_df["country"] = (
    climate_df["city"]
    .apply(extract_country)
)

# Standardize names
country_mapping = {
    "USA": "United States"
}

climate_df["country"] = (
    climate_df["country"]
    .replace(country_mapping)
)

# -------------------------
# Merge datasets
# -------------------------

master_df = climate_df.merge(
    water_df,
    on=["country", "state_region"],
    how="left"
)

# -------------------------
# Save merged dataset
# -------------------------

master_df.to_csv(
    "data/processed/master_climate_water.csv",
    index=False
)

# -------------------------
# Results
# -------------------------

print("\nMerged Dataset Shape:")
print(master_df.shape)

print("\nMissing Values:")
print(master_df.isnull().sum())

print("\nPreview:")
print(master_df.head(15))

# Truth Dataset
master_df.to_csv(
    "data/processed/master_dataset_v1.csv",
    index=False
)

print("\nMaster Dataset V1 saved!")