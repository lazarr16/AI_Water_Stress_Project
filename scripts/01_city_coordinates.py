from geopy.geocoders import Nominatim
import pandas as pd
import time

# City list
cities = [
    "Mumbai, India",
    "Pune, India",
    "Bengaluru, India",
    "Hyderabad, India",
    "Chennai, India",
    "Phoenix, USA",
    "Dallas, USA",
    "Singapore",
    "Dublin, Ireland",
    "Amsterdam, Netherlands"
]

# Initialize geolocator
geolocator = Nominatim(user_agent="ai_water_stress_project")

city_data = []

for city in cities:
    try:
        location = geolocator.geocode(city)

        city_data.append({
            "city": city,
            "latitude": location.latitude,
            "longitude": location.longitude
        })

        print(f"Done: {city}")

        time.sleep(1)

    except Exception as e:
        print(f"Error with {city}: {e}")

# Create dataframe
df = pd.DataFrame(city_data)

# Save CSV
df.to_csv("data/raw/city_coordinates.csv", index=False)

print("\nCoordinates saved successfully!")
print(df)