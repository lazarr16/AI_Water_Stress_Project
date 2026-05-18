import pandas as pd

# -------------------------
# Load datasets
# -------------------------

regions_df = pd.read_csv(
    "data/processed/city_region_mapping.csv"
)

climate_df = pd.read_csv(
    "data/raw/nasa_climate_data_v2.csv"
)

water_df = pd.read_csv(
    "data/processed/water_risk_aggregated.csv"
)

print("Datasets loaded!")

# -------------------------
# Fix country names
# -------------------------

country_mapping = {
    "Éire / Ireland": "Ireland",
    "Nederland": "Netherlands",
    "Deutschland": "Germany",
    "España": "Spain",
    "Italia": "Italy",
    "Danmark": "Denmark",
    "Österreich": "Austria",
    "Norge": "Norway",
    "Polska": "Poland",
    "Portugal": "Portugal",
    "Suomi / Finland": "Finland",
    "Magyarország": "Hungary",
    "Česko": "Czech Republic",
    "Sverige": "Sweden",
    "Türkiye": "Turkey",
    "South Africa": "South Africa",
    "Australia": "Australia",
    "Malaysia": "Malaysia",
    "Indonesia": "Indonesia",
    "Philippines": "Philippines",
    "India": "India",
    "United States": "United States",
    "France": "France",
    "United Kingdom": "United Kingdom",
    "Singapore": "Singapore",
    "Kenya": "Kenya"
}

regions_df["country"] = (
    regions_df["country"]
    .replace(country_mapping)
)

# -------------------------
# Fix state/region names
# -------------------------

state_mapping = {

    # Delhi
    None: "NCT of Delhi",

    # Ireland
    "Leinster": "Dublin",

    # Netherlands
    "Noord-Holland": "Noord-Holland",

    # UK
    "England": "England",

    # Germany
    "Hessen": "Hessen",

    # Spain
    "Comunidad de Madrid":
    "Madrid",

    # France
    "Île-de-France":
    "Île-de-France",

    # Poland
    "województwo mazowieckie":
    "Mazowieckie",

    # Sweden
    "Stockholms län":
    "Stockholm",

    # Australia
    "New South Wales":
    "New South Wales"
}

regions_df["state_region"] = (
    regions_df["state_region"]
    .replace(state_mapping)
)

# -------------------------
# Fix empty regions
# -------------------------

manual_city_region = {
    "Delhi, India": "NCT of Delhi",
    "Singapore": "Central",
    "Berlin, Germany": "Berlin",
    "Vienna, Austria": "Vienna",
    "Oslo, Norway": "Oslo",
    "Tokyo, Japan": "Tokyo",
    "Seoul, South Korea": "Seoul",
    "Bangkok, Thailand": "Bangkok",
    "Beijing, China": "Beijing",
    "Taipei, Taiwan": "Taipei",
    "Kuala Lumpur, Malaysia":
    "Kuala Lumpur"
}

for city, region in (
    manual_city_region.items()
):

    regions_df.loc[
        regions_df["city"] == city,
        "state_region"
    ] = region

# -------------------------
# Merge climate + region
# -------------------------

master_df = climate_df.merge(
    regions_df,
    on="city",
    how="left"
)

# -------------------------
# Merge water data
# -------------------------

master_df = master_df.merge(
    water_df,
    on=["country", "state_region"],
    how="left"
)

# -------------------------
# Save output
# -------------------------

master_df.to_csv(
    "data/processed/"
    "master_dataset_scaled_v1.csv",
    index=False
)

# -------------------------
# Validation
# -------------------------

print("\nShape:")
print(master_df.shape)

print("\nMissing Water Values:")
print(
    master_df[
        [
            "baseline_water_stress",
            "water_depletion",
            "seasonal_variability",
            "drought_severity"
        ]
    ]
    .isnull()
    .sum()
)

print("\nRows Missing Water Data:")

missing = master_df[
    master_df[
        "baseline_water_stress"
    ].isnull()
]

print(
    missing[
        [
            "city",
            "country",
            "state_region"
        ]
    ]
    .drop_duplicates()
)