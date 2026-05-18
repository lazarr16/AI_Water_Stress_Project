from geopy.geocoders import Nominatim
import pandas as pd
import time

# -------------------------
# Load city list
# -------------------------

city_df = pd.read_csv(
    "data/external/city_list_v2.csv"
)

print("City list loaded!")

# Initialize geocoder
geolocator = Nominatim(
    user_agent="ai_water_stress_project"
)

city_data = []

# -------------------------
# Generate coordinates
# -------------------------

for city in city_df["city"]:

    try:
        location = geolocator.geocode(city)

        if location:

            city_data.append({
                "city": city,
                "latitude": location.latitude,
                "longitude": location.longitude
            })

            print(f"Done: {city}")

        else:
            print(f"Not Found: {city}")

        time.sleep(1)

    except Exception as e:
        print(f"Error with {city}: {e}")

# -------------------------
# Save output
# -------------------------

coordinates_df = pd.DataFrame(city_data)

coordinates_df.to_csv(
    "data/raw/city_coordinates_v2.csv",
    index=False
)

print("\nCoordinates Generated!")

print("\nShape:")
print(coordinates_df.shape)

print("\nPreview:")
print(coordinates_df.head())