import pandas as pd
import requests
import time
import json


# Load coordinates dataset
cities_df = pd.read_csv("data/raw/city_coordinates.csv")
# Ensure cities_df is available
all_climate_data = []

# Years to collect
start_year = 2020
end_year = 2024

# Format dates for daily API
start_date = f"{start_year}0101"
end_date = f"{end_year}1231"

for _, row in cities_df.iterrows():
    city = row["city"]
    lat = row["latitude"]
    lon = row["longitude"]

    print(f"Fetching daily climate data for {city}...")

    # Using the 'daily' endpoint which is more reliable
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract parameters
        parameters = data["properties"]["parameter"]

        # Create a temporary dataframe for this city's daily data
        df_daily = pd.DataFrame({
            "temp": parameters["T2M"],
            "rain": parameters["PRECTOTCORR"],
            "hum": parameters["RH2M"]
        })

        # Convert index to datetime and extract year
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily['year'] = df_daily.index.year

        df_daily.to_csv(
            f"data/raw/{city}_daily_climate.csv"
            )

        # Aggregate to annual averages
        annual_stats = df_daily.groupby('year').agg({
            'temp': 'mean',
            'hum': 'mean',
            'rain': 'sum'
        }).reset_index()

        for _, annual_row in annual_stats.iterrows():
            all_climate_data.append({
                "city": city,
                "year": int(annual_row['year']),
                "avg_temperature": round(annual_row['temp'], 2),
                "avg_humidity": round(annual_row['hum'], 2),
                "precipitation": round(annual_row['rain'], 2)
            })

        print(f"Done: {city}")
        time.sleep(1) # Small delay to respect API limits

    except Exception as e:
        print(f"Error with {city}: {e}")

# Create final dataframe
climate_df = pd.DataFrame(all_climate_data)
df_daily.replace(-999, pd.NA, inplace=True)
# Save data
climate_df.to_csv(
    "data/raw/nasa_climate_data.csv",
    index=False
)


if not climate_df.empty:
    print("\nClimate data saved successfully!")
    print(climate_df.head())
else:
    print("\nWarning: No data was collected.")