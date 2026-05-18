import pandas as pd

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/processed/curated_model_dataset_v1.csv"
)

print("Dataset loaded!")

# -------------------------
# Clean city names
# -------------------------

df["city_clean"] = (
    df["city"]
    .str.split(",")
    .str[0]
    .str.strip()
)

# -------------------------
# Infrastructure Rules
# -------------------------

very_high_cities = [
    "Dallas",
    "Phoenix",
    "Singapore",
    "Dublin",
    "Amsterdam",
    "Sydney",
    "Melbourne"
]

high_cities = [
    "Mumbai",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Delhi",
    "Noida",
    "Seattle",
    "Chicago",
    "Austin",
    "New York",
    "Boston",
    "Houston",
    "Los Angeles"
]

medium_cities = [
    "Pune",
    "Ahmedabad",
    "Kochi",
    "Jaipur",
    "Lucknow",
    "Nagpur",
    "Indore",
    "Kolkata",
    "Visakhapatnam"
]

# -------------------------
# Scoring Functions
# -------------------------

def compute_growth(city):

    if city in very_high_cities:
        return "Very High"

    elif city in high_cities:
        return "High"

    return "Medium"


def hyperscaler(city):

    if city in medium_cities:
        return 0

    return 1


def data_center_count(city):

    if city in very_high_cities:
        return 70

    elif city in high_cities:
        return 40

    return 15


def estimated_mw(city):

    if city in very_high_cities:
        return 350

    elif city in high_cities:
        return 200

    return 80

# -------------------------
# Generate Features
# -------------------------

df["compute_growth_score"] = (
    df["city_clean"]
    .apply(compute_growth)
)

df["hyperscaler_presence"] = (
    df["city_clean"]
    .apply(hyperscaler)
)

df["data_center_count"] = (
    df["city_clean"]
    .apply(data_center_count)
)

df["estimated_mw_capacity"] = (
    df["city_clean"]
    .apply(estimated_mw)
)

# -------------------------
# Save output
# -------------------------

df.to_csv(
    "data/processed/"
    "model_dataset_v2.csv",
    index=False
)

# -------------------------
# Validation
# -------------------------

print("\nShape:")
print(df.shape)

print("\nCompute Growth:")
print(
    df["compute_growth_score"]
    .value_counts()
)

print("\nHyperscaler Presence:")
print(
    df["hyperscaler_presence"]
    .value_counts()
)

print("\nPreview:")
print(
    df[
        [
            "city",
            "compute_growth_score",
            "hyperscaler_presence",
            "data_center_count",
            "estimated_mw_capacity"
        ]
    ]
    .drop_duplicates()
    .head(20)
)