import pandas as pd
import folium

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/processed/final_model_dataset.csv"
)

coords = pd.read_csv(
    "data/raw/city_coordinates_v2.csv"
)

print("Datasets loaded!")

# =====================================
# KEEP LATEST YEAR ONLY
# =====================================

latest_df = df[
    df["year"] == 2024
].copy()

print(
    f"2024 Cities: {latest_df.shape[0]}"
)

# =====================================
# MERGE COORDINATES
# =====================================

merged = latest_df.merge(
    coords,
    on="city",
    how="left"
)

print("\nMerged Shape:")
print(merged.shape)

# =====================================
# COLOR MAPPING
# =====================================

risk_colors = {
    "Low": "green",
    "Medium": "orange",
    "High": "red"
}

# =====================================
# CREATE MAP
# =====================================

world_map = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles="CartoDB dark_matter"
)

# =====================================
# ADD MARKERS
# =====================================

for _, row in merged.iterrows():

    risk = row[
        "water_stress_risk"
    ]

    color = risk_colors.get(
        risk,
        "blue"
    )

    popup_text = f"""
    <b>{row['city']}</b><br>
    Risk: {risk}<br>
    Sustainability:
    {round(row['sustainability_suitability_score'],1)}<br>
    Data Centers:
    {row['data_center_count']}<br>
    MW Capacity:
    {row['estimated_mw_capacity']}
    """

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=8,
        popup=popup_text,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8
    ).add_to(world_map)

# =====================================
# SAVE MAP
# =====================================

world_map.save(
    "visuals/world_risk_map.html"
)

print(
    "\nMap saved successfully!"
)

print(
    "Open visuals/world_risk_map.html"
)