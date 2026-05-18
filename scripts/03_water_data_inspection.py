import pandas as pd

# File path
file_path = "data/external/aqueduct_water_risk.csv"

print("Loading dataset...")

# Read only a sample first (faster)
df = pd.read_csv(file_path, low_memory=False)
print("\nDataset Loaded Successfully!")

# Basic info
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values (Top 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nData Types:")
print(df.dtypes)