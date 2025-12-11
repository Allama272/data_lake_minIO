import pandas as pd
from minio import Minio
from io import BytesIO

# Connect to MinIO
client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="admin12345",
    secure=False
)

# Load cleaned data from MinIO silver bucket
weather_obj = client.get_object("silver", "synthetic_weather_cleaned.parquet")
weather = pd.read_parquet(BytesIO(weather_obj.read()))

traffic_obj = client.get_object("silver", "synthetic_traffic_cleaned.parquet")
traffic = pd.read_parquet(BytesIO(traffic_obj.read()))

print(f"Weather data shape: {weather.shape}")
print(f"Traffic data shape: {traffic.shape}")

# Merge datasets
merged = pd.merge(
    weather,
    traffic,
    on=['date_time', 'city'],
    how='inner'
)
merged.rename(columns= {'visibility_m_x': 'visibility_m_weather',
           'visibility_m_y': 'visibility_m_traffic'}, inplace=True)

# Converting object to numeric
merged['visibility_m_weather'] = pd.to_numeric(merged['visibility_m_weather'], errors='coerce')

# Clipping extreme outliers and keeping the nan's to fill with median
merged = merged[(merged['visibility_m_weather'] <= 20000) | (merged['visibility_m_weather'].isna())]

# fill with median
merged['visibility_m_weather'] = merged['visibility_m_weather'].fillna(merged['visibility_m_weather'].median())

print(f"\nMerged data shape: {merged.shape}")
print(f"Merge retention: {(merged.shape[0] / min(weather.shape[0], traffic.shape[0])) * 100:.1f}%")

# Save merged dataset to gold bucket
merged_buffer = BytesIO()
merged.to_csv(merged_buffer, index=False)
merged_buffer.seek(0)

client.put_object(
    "gold", 
    "merged_analytics.csv", 
    merged_buffer, 
    len(merged_buffer.getvalue()),
    content_type="text/csv"
)

print(f"\ Merged dataset saved to gold layer under merged_analytics.csv")
