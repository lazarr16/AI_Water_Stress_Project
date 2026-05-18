import pandas as pd
import requests
import time

# -------------------------
# Load coordinates
# -------------------------

cities_df = pd.read_csv(
    "data/raw/city_coordinates_v2.csv"
)

print("Coordinates loaded!")

all_climate_data = []

# Years
start_year = 2020
end_year = 2024

# Format for daily endpoint
start_date = f"{start_year}0101"
end_date = f"{end_year}1231"

# -------------------------
# NASA Climate Extraction
# -------------------------

for _, row in cities_df.iterrows():

    city = row["city"]
    lat = row["latitude"]
    lon = row["longitude"]

    print(f"Fetching: {city}")

    base_url = (
        "https://power.larc.nasa.gov/"
        "api/temporal/daily/point"
    )

    params = {
        "parameters":
        "T2M,PRECTOTCORR,RH2M",

        "community": "AG",

        "longitude": lon,
        "latitude": lat,

        "start": start_date,
        "end": end_date,

        "format": "JSON"
    }

    try:

        response = requests.get(
            base_url,
            params=params
        )

        response.raise_for_status()

        data = response.json()

        parameters = (
            data["properties"]
            ["parameter"]
        )

        # Daily dataframe
        df_daily = pd.DataFrame({
            "temp":
            parameters["T2M"],

            "rain":
            parameters["PRECTOTCORR"],

            "humidity":
            parameters["RH2M"]
        })

        # Clean placeholder values
        df_daily.replace(
            -999,
            pd.NA,
            inplace=True
        )

        # Convert date index
        df_daily.index = pd.to_datetime(
            df_daily.index
        )

        df_daily["year"] = (
            df_daily.index.year
        )

        # Annual aggregation
        annual_stats = (
            df_daily
            .groupby("year")
            .agg({
                "temp": "mean",
                "humidity": "mean",
                "rain": "sum"
            })
            .reset_index()
        )

        # Save annual rows
        for _, annual_row in (
            annual_stats.iterrows()
        ):

            all_climate_data.append({
                "city": city,
                "year":
                int(annual_row["year"]),

                "avg_temperature":
                round(
                    annual_row["temp"],
                    2
                ),

                "avg_humidity":
                round(
                    annual_row[
                        "humidity"
                    ],
                    2
                ),

                "precipitation":
                round(
                    annual_row["rain"],
                    2
                )
            })

        print(f"Done: {city}")

        time.sleep(1)

    except Exception as e:

        print(
            f"Error with {city}: {e}"
        )

# -------------------------
# Save output
# -------------------------

climate_df = pd.DataFrame(
    all_climate_data
)

climate_df.to_csv(
    "data/raw/"
    "nasa_climate_data_v2.csv",
    index=False
)

print("\nClimate pipeline complete!")

print("\nShape:")
print(climate_df.shape)

print("\nPreview:")
print(climate_df.head())