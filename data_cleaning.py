import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO

#connecting to MinIO
client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="admin12345",
    secure=False
)

#functions to clean our CSVs

def clean_weather(df):
    #remove duplicates
    df = df.drop_duplicates()

    

    #convert date_time, drop invalid
    df.loc[:, 'date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    df = df.dropna(subset=['date_time'])

    #remove unrealistic/outlier numeric values 
    if 'temperature_c' in df.columns:
        df = df[df['temperature_c'].between(-10, 40)]
    if 'humidity' in df.columns:
        df = df[df['humidity'].between(0, 100)]
    if 'rain_mm' in df.columns:
        df = df[df['rain_mm'] < 50]  # remove heavy rain outliers
    if 'wind_speed_kmh' in df.columns:
        df = df[df['wind_speed_kmh'] <= 120]
    if 'air_pressure_hpa' in df.columns:
        df = df[df['air_pressure_hpa'].between(970, 1040)]

    #fill numeric NaNs with median
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    #fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df



def clean_traffic(df):
    df = df.drop_duplicates()


    df.loc[:, 'date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    df = df.dropna(subset=['date_time'])

    numeric_cols = ['vehicle_count', 'avg_speed_kmh', 'accident_count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'vehicle_count' in df.columns:
        df = df[df['vehicle_count'].between(0, 12000)]
    if 'avg_speed_kmh' in df.columns:
        df = df[df['avg_speed_kmh'].between(0, 80)]
    if 'accident_count' in df.columns:
        df = df[df['accident_count'].between(0, 10)]

    #visibility conversion
    if 'visibility_m' in df.columns:
        # onvert everything to string first, strip, lowercase
        df['visibility_m'] = df['visibility_m'].astype(str).str.strip().str.lower()

        # Replace non-numeric words with NaN
        df.loc[~df['visibility_m'].str.match(r'^\d+(\.\d+)?$'), 'visibility_m'] = np.nan

        # Convert to numeric
        df['visibility_m'] = pd.to_numeric(df['visibility_m'], errors='coerce')

        # Fill NaNs with median and clamp
        median_vis = df['visibility_m'].median()
        df['visibility_m'] = df['visibility_m'].fillna(median_vis)
        df['visibility_m'] = df['visibility_m'].clip(100, 10000)

    if 'congestion_level' in df.columns:
        # Standardize congestion levels to Low/Medium/High
        mapping = {
            'Low': 'Low',
            'Medium': 'Medium',
            'High': 'High',
            'Very High': 'High',      # Map invalid to valid
            'Extreme': 'High',        # Map invalid to valid
            'Low-Medium': 'Medium',   # Map invalid to valid
            '': None                  # Empty string to None
        }
        df['congestion_level'] = df['congestion_level'].map(mapping)
        # Fill remaining NaNs with mode (most common value)
        if df['congestion_level'].notna().any():
            df['congestion_level'] = df['congestion_level'].fillna(df['congestion_level'].mode()[0])

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    if 'traffic_id' in df.columns:
        max_id = int(df['traffic_id'].max(skipna=True))
        missing_mask = df['traffic_id'].isna()
        df.loc[missing_mask, 'traffic_id'] = np.arange(max_id + 1, max_id + 1 + missing_mask.sum())
        df['traffic_id'] = df['traffic_id'].astype(int)

    return df






# load CSVs from MinIO
weather_obj = client.get_object("bronze", "synthetic_weather_data.csv")
weather_df = pd.read_csv(BytesIO(weather_obj.read()))

traffic_obj = client.get_object("bronze", "synthetic_traffic_data.csv")
traffic_df = pd.read_csv(BytesIO(traffic_obj.read()))


#clean the data
weather_clean = clean_weather(weather_df)
traffic_clean = clean_traffic(traffic_df)

print("clean_weather size: ", weather_clean.shape)
print("clean_traffic size: ", traffic_clean.shape)

#Save cleaned data to Parquet in memory
weather_buffer = BytesIO()
traffic_buffer = BytesIO()

weather_clean.to_parquet(weather_buffer, index=False)
traffic_clean.to_parquet(traffic_buffer, index=False)

weather_buffer.seek(0)
traffic_buffer.seek(0)

# Upload cleaned files to MinIO silver bucket 
client.put_object("silver", "synthetic_weather_cleaned.parquet", weather_buffer, len(weather_buffer.getvalue()))
client.put_object("silver", "synthetic_traffic_cleaned.parquet", traffic_buffer, len(traffic_buffer.getvalue()))

print("Cleaning done! Files uploaded to silver bucket.")
