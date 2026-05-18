from geopy.geocoders import Nominatim
import pandas as pd
import time

# -------------------------
# Load coordinates
# -------------------------

df = pd.read_csv(
    "data/raw/city_coordinates_v2.csv"
)

print("Coordinates loaded!")

geolocator = Nominatim(
    user_agent="ai_water_stress_project"
)

region_data = []

# -------------------------
# Reverse Geocoding
# -------------------------

for _, row in df.iterrows():

    city = row["city"]
    lat = row["latitude"]
    lon = row["longitude"]

    try:

        location = geolocator.reverse(
            (lat, lon),
            exactly_one=True
        )

        address = (
            location.raw["address"]
        )

        state = (
            address.get("state")
            or address.get("region")
            or address.get("county")
            or address.get(
                "state_district"
            )
        )

        country = (
            address.get("country")
        )

        region_data.append({
            "city": city,
            "state_region": state,
            "country": country
        })

        print(
            f"Done: {city}"
        )

        time.sleep(1)

    except Exception as e:

        print(
            f"Error with {city}: {e}"
        )

# -------------------------
# Save output
# -------------------------

region_df = pd.DataFrame(
    region_data
)

region_df.to_csv(
    "data/processed/"
    "city_region_mapping.csv",
    index=False
)

print("\nCompleted!")

print("\nShape:")
print(region_df.shape)

print("\nPreview:")
print(region_df.head(15))