import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_base_timestamps(n_records=5000):
    """
    Generate base timestamps that will be shared between weather and traffic datasets
    to ensure better alignment for merging
    """
    start_date = datetime(2024, 1, 1)
    timestamps = []
    
    # Generate 70% matching timestamps
    matching_count = int(n_records * 0.7)
    
    # Create base timestamps (every 30 minutes to 2 hours)
    for i in range(matching_count):
        offset_minutes = i * random.randint(30, 120)
        timestamps.append(start_date + timedelta(minutes=offset_minutes))
    
    # Add 30% random timestamps for variety
    for i in range(n_records - matching_count):
        random_minutes = random.randint(0, 525600)  # Random minute in year
        timestamps.append(start_date + timedelta(minutes=random_minutes))
    
    # Shuffle to mix matching and non-matching timestamps
    random.shuffle(timestamps)
    
    return timestamps

def generate_synthetic_weather_dataset(n_records=5000, base_timestamps=None):
    """
    Generate synthetic weather dataset with realistic patterns and messy data
    as specified in the project requirements
    """
    print(f"Generating {n_records} synthetic weather records...")
    
    # Initialize lists for each column
    data = {
        'weather_id': [],
        'date_time': [],
        'city': [],
        'season': [],
        'temperature_c': [],
        'humidity': [],
        'rain_mm': [],
        'wind_speed_kmh': [],
        'visibility_m': [],
        'weather_condition': [],
        'air_pressure_hpa': []
    }
    
    # Use provided timestamps or generate new ones
    if base_timestamps is None:
        start_date = datetime(2024, 1, 1)
        base_timestamps = [start_date + timedelta(hours=random.randint(0, 8760)) 
                          for _ in range(n_records)]
    
    # London seasons temperature ranges (Celsius)
    season_ranges = {
        'Winter': (-5, 10),
        'Spring': (5, 20),
        'Summer': (12, 28),
        'Autumn': (5, 18)
    }
    
    # Weather conditions and their probabilities
    weather_conditions = ['Clear', 'Rain', 'Fog', 'Storm', 'Snow']
    weather_probs = [0.5, 0.25, 0.1, 0.05, 0.1]
    
    # Generate records
    for i in range(n_records):
        # ID with some duplicates (5% chance of duplicate)
        if i > 0 and random.random() < 0.05:
            data['weather_id'].append(data['weather_id'][-1])  # Duplicate last ID
        elif random.random() < 0.02:  # 2% missing IDs
            data['weather_id'].append(None)
        else:
            data['weather_id'].append(5001 + i)
        
        # Date Time with various formats and issues
        current_date = base_timestamps[i] if i < len(base_timestamps) else datetime(2024, 1, 1)
        
        # Introduce date format variations and issues
        format_choice = random.random()
        if format_choice < 0.60:  # 60% correct format
            date_str = current_date.strftime("%Y-%m-%d %H:%M")
        elif format_choice < 0.70:  # 10% alternative format
            date_str = current_date.strftime("%d/%m/%Y %I%p")
        elif format_choice < 0.80:  # 10% ISO format
            date_str = current_date.strftime("%Y-%m-%dT%H:%M:00Z")
        elif format_choice < 0.90:  # 10% invalid format
            date_str = f"2099-13-{random.randint(40,99)} {random.randint(25,99)}:{random.randint(60,99)}"
        else:  # 10% missing
            date_str = None
        
        data['date_time'].append(date_str)
        
        # City (mostly London, some missing)
        if random.random() < 0.98:
            data['city'].append('London')
        else:
            data['city'].append(None)
        
        # Determine season from date (when date is valid)
        if date_str and format_choice < 0.80:  # Only for valid dates
            month = current_date.month
            if month in [12, 1, 2]:
                season = 'Winter'
            elif month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            else:
                season = 'Autumn'
            
            # Introduce some season inconsistencies (5% chance)
            if random.random() < 0.05:
                season = random.choice(['Winter', 'Spring', 'Summer', 'Autumn'])
        else:
            # For invalid/missing dates, random season or missing
            if random.random() < 0.80:
                season = random.choice(['Winter', 'Spring', 'Summer', 'Autumn'])
            else:
                season = None
        
        data['season'].append(season)
        
        # Temperature with outliers and season-specific patterns
        if season and season in season_ranges:
            temp_min, temp_max = season_ranges[season]
            # 95% normal values, 5% outliers
            if random.random() < 0.95:
                temperature = round(np.random.normal((temp_min + temp_max) / 2, 
                                                    (temp_max - temp_min) / 6), 1)
                temperature = max(temp_min, min(temp_max, temperature))
            else:
                # Outliers
                temperature = random.choice([-30, 60, 100, -50])
        else:
            # Random temperature if no season
            temperature = round(np.random.normal(15, 10), 1)
        
        # 3% missing temperatures
        if random.random() < 0.03:
            temperature = None
        
        data['temperature_c'].append(temperature)
        
        # Humidity (20-100%, with outliers)
        if random.random() < 0.95:
            humidity = random.randint(20, 100)
        else:
            humidity = random.choice([-10, 150, 200, -5])  # Outliers
        
        # 2% missing humidity
        if random.random() < 0.02:
            humidity = None
        
        data['humidity'].append(humidity)
        
        # Rainfall (mm) - mostly 0-50, some extremes
        if random.random() < 0.90:
            # Most days have little to no rain
            rain = round(np.random.exponential(5), 1)
            if rain > 50:
                rain = 50
        else:
            # Extreme rainfall
            rain = round(random.uniform(50, 200), 1)
        
        # 3% missing rain values (FIXED)
        if random.random() < 0.03:
            rain = None
        
        data['rain_mm'].append(rain)
        
        # Wind speed (km/h) with outliers
        if random.random() < 0.95:
            wind_speed = round(np.random.gamma(shape=2, scale=10), 1)
            if wind_speed > 80:
                wind_speed = 80
        else:
            wind_speed = random.choice([200, 250, 300, 150])  # Outliers
        
        # 4% missing wind speed
        if random.random() < 0.04:
            wind_speed = None
        
        data['wind_speed_kmh'].append(wind_speed)
        
        # Visibility (meters) with issues (FIXED for exact 2% strings)
        visibility_choice = random.random()
        if visibility_choice < 0.88:  # 88% normal range
            visibility = random.randint(50, 10000)
        elif visibility_choice < 0.93:  # 5% extreme values
            visibility = random.randint(20000, 50000)
        elif visibility_choice < 0.95:  # 2% non-numeric strings
            visibility = random.choice(['Low', 'Unknown', 'N/A', 'Very Poor'])
        else:  # 5% missing
            visibility = None
        
        data['visibility_m'].append(visibility)
        
        # Weather condition with missing values
        if random.random() < 0.95:
            condition = np.random.choice(weather_conditions, p=weather_probs)
        else:
            condition = None
        
        data['weather_condition'].append(condition)
        
        # Air pressure (hPa)
        if random.random() < 0.97:
            pressure = round(np.random.normal(1013, 20), 1)
            pressure = max(950, min(1050, pressure))
        else:
            pressure = None
        
        data['air_pressure_hpa'].append(pressure)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some duplicate rows (5% duplicates)
    duplicate_indices = random.sample(range(100, n_records), int(n_records * 0.05))
    for idx in duplicate_indices:
        if idx < len(df):
            duplicate_row = df.iloc[idx].copy()
            df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)
    
    print(f"Weather dataset generated with {len(df)} records (including duplicates)")
    return df

def generate_synthetic_traffic_dataset(n_records=5000, base_timestamps=None):
    """
    Generate synthetic traffic dataset with realistic patterns and messy data
    """
    print(f"Generating {n_records} synthetic traffic records...")
    
    # Road conditions
    road_conditions = ['Dry', 'Wet', 'Snowy', 'Damaged']
    road_condition_probs = [0.7, 0.2, 0.05, 0.05]
    
    # Congestion levels
    congestion_levels = ['Low', 'Medium', 'High']
    
    # Initialize data dictionary
    data = {
        'traffic_id': [],
        'date_time': [],
        'city': [],
        'area': [],
        'vehicle_count': [],
        'avg_speed_kmh': [],
        'accident_count': [],
        'congestion_level': [],
        'road_condition': [],
        'visibility_m': []
    }
    
    # Use provided timestamps or generate new ones
    if base_timestamps is None:
        start_date = datetime(2024, 1, 1)
        base_timestamps = [start_date + timedelta(minutes=random.randint(0, 525600)) 
                          for _ in range(n_records)]
    
    for i in range(n_records):
        # Traffic ID with duplicates and missing values
        if i > 0 and random.random() < 0.04:
            data['traffic_id'].append(data['traffic_id'][-1])  # Duplicate
        elif random.random() < 0.02:  # Missing IDs
            data['traffic_id'].append(None)
        else:
            data['traffic_id'].append(9001 + i)
        
        # Date Time with variations
        current_date = base_timestamps[i] if i < len(base_timestamps) else datetime(2024, 1, 1)
        
        format_choice = random.random()
        if format_choice < 0.65:  # Correct format
            date_str = current_date.strftime("%Y-%m-%d %H:%M")
        elif format_choice < 0.75:  # Alternative format
            date_str = current_date.strftime("%d/%m/%Y %I%p")
        elif format_choice < 0.85:  # Invalid format
            date_str = f"2099-00-00 {random.randint(90,99)}:{random.randint(60,99)}"
        elif format_choice < 0.90:  # String garbage
            date_str = random.choice(['TBD', 'Unknown', 'N/A', ''])
        else:  # Missing
            date_str = None
        
        data['date_time'].append(date_str)
        
        # City (mostly London, some missing)
        if random.random() < 0.97:
            data['city'].append('London')
        else:
            data['city'].append(None)
        
        # Area - mostly missing (93% missing as per requirements)
        if random.random() < 0.93:
            area = None
        else:
            # Generate generic area names when present
            area = f"Area_{random.randint(1, 15)}"
        
        data['area'].append(area)
        
        # Vehicle count with outliers
        if random.random() < 0.92:
            # Normal traffic (time-dependent)
            hour = current_date.hour
            if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
                vehicles = int(np.random.normal(3000, 800))
            elif 10 <= hour <= 15:  # Daytime
                vehicles = int(np.random.normal(2000, 600))
            elif 19 <= hour <= 22:  # Evening
                vehicles = int(np.random.normal(1500, 500))
            else:  # Night
                vehicles = int(np.random.normal(500, 300))
            
            vehicles = max(0, min(5000, vehicles))
        else:
            # Extreme outliers
            vehicles = random.choice([20000, 25000, 30000, 0, -100])
        
        # 2% missing vehicle counts
        if random.random() < 0.02:
            vehicles = None
        
        data['vehicle_count'].append(vehicles)
        
        # Average speed with negative values and outliers
        if vehicles is not None and vehicles > 0:
            # Speed inversely related to vehicle count
            base_speed = 80 - (vehicles / 5000 * 60)
            speed = round(np.random.normal(base_speed, 15), 1)
            
            # Ensure realistic bounds for normal values
            if speed < 3:
                speed = 3
            elif speed > 120:
                speed = 120
            
            # 3% chance of negative speed (FIXED - happens after normal calculation)
            if random.random() < 0.03:
                speed = -random.uniform(1, 50)
        else:
            # Random speed for missing vehicle counts
            speed = round(np.random.uniform(3, 120), 1)
        
        # 3% missing speeds
        if random.random() < 0.03:
            speed = None
        
        data['avg_speed_kmh'].append(speed)
        
        # Accident count (rare events with some extremes)
        if random.random() < 0.95:
            # Poisson distribution for rare events
            accidents = np.random.poisson(0.1)
        else:
            # Extreme values
            accidents = random.choice([20, 30, 50, 100])
        
        # 5% missing accident counts
        if random.random() < 0.05:
            accidents = None
        
        data['accident_count'].append(accidents)
        
        # Congestion level (related to vehicle count and speed)
        if vehicles is not None and speed is not None and speed >= 0:
            if vehicles > 4000 or speed < 10:
                congestion = 'High'
            elif vehicles > 2500 or speed < 30:
                congestion = 'Medium'
            else:
                congestion = 'Low'
            
            # 10% incorrect congestion categories
            if random.random() < 0.10:
                congestion = random.choice(['Very High', 'Low-Medium', 'Extreme', ''])
        else:
            congestion = random.choice(congestion_levels)
        
        # 4% missing congestion
        if random.random() < 0.04:
            congestion = None
        
        data['congestion_level'].append(congestion)
        
        # Road condition
        if random.random() < 0.96:
            condition = np.random.choice(road_conditions, p=road_condition_probs)
        else:
            condition = None
        
        data['road_condition'].append(condition)
        
        # Visibility (traffic sensor)
        if random.random() < 0.90:
            visibility = random.randint(50, 10000)
        elif random.random() < 0.50:
            visibility = random.randint(15000, 30000)  # Extreme
        else:
            visibility = None
        
        data['visibility_m'].append(visibility)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add duplicate rows (6% duplicates)
    duplicate_indices = random.sample(range(100, n_records), int(n_records * 0.06))
    for idx in duplicate_indices:
        if idx < len(df):
            duplicate_row = df.iloc[idx].copy()
            df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)
    
    print(f"Traffic dataset generated with {len(df)} records (including duplicates)")
    return df

def save_datasets(weather_df, traffic_df):
    """Save datasets to CSV files in the same directory as the script"""
    
    # Save weather dataset
    weather_file = os.path.join(SCRIPT_DIR, 'synthetic_weather_data.csv')
    weather_df.to_csv(weather_file, index=False)
    print(f"\nWeather dataset saved to: {weather_file}")
    print(f"Weather dataset shape: {weather_df.shape}")
    
    # Show sample of messy data
    print("\nWeather dataset sample (with messy data):")
    print(weather_df.head(10))
    
    # Data quality summary for weather (ENHANCED)
    print("\n--- Weather Data Quality Issues ---")
    print(f"Total records: {len(weather_df)}")
    print(f"Duplicate IDs: {weather_df['weather_id'].duplicated().sum()}")
    print(f"Duplicate rows: {weather_df.duplicated().sum()}")
    print(f"Missing weather_id: {weather_df['weather_id'].isnull().sum()}")
    print(f"Missing date_time: {weather_df['date_time'].isnull().sum()}")
    print(f"Missing temperature: {weather_df['temperature_c'].isnull().sum()}")
    print(f"Missing rain_mm: {weather_df['rain_mm'].isnull().sum()}")
    
    # Safe outlier checking
    temp_outliers = 0
    if 'temperature_c' in weather_df.columns:
        temp_outliers = ((weather_df['temperature_c'] < -10) | 
                        (weather_df['temperature_c'] > 35)).sum()
    print(f"Temperature outliers (< -10 or > 35): {temp_outliers}")
    
    humidity_neg = 0
    if 'humidity' in weather_df.columns:
        humidity_neg = (weather_df['humidity'] < 0).sum()
    print(f"Negative humidity values: {humidity_neg}")
    
    # Count non-numeric visibility strings
    visibility_strings = 0
    if 'visibility_m' in weather_df.columns:
        visibility_strings = weather_df['visibility_m'].apply(
            lambda x: isinstance(x, str) and not str(x).replace('.','',1).isdigit()
        ).sum()
    print(f"Non-numeric visibility strings: {visibility_strings}")
    
    # Save traffic dataset
    traffic_file = os.path.join(SCRIPT_DIR, 'synthetic_traffic_data.csv')
    traffic_df.to_csv(traffic_file, index=False)
    print(f"\nTraffic dataset saved to: {traffic_file}")
    print(f"Traffic dataset shape: {traffic_df.shape}")
    
    # Show sample of messy data
    print("\nTraffic dataset sample (with messy data):")
    print(traffic_df.head(10))
    
    # Data quality summary for traffic (ENHANCED)
    print("\n--- Traffic Data Quality Issues ---")
    print(f"Total records: {len(traffic_df)}")
    print(f"Duplicate IDs: {traffic_df['traffic_id'].duplicated().sum()}")
    print(f"Duplicate rows: {traffic_df.duplicated().sum()}")
    print(f"Missing traffic_id: {traffic_df['traffic_id'].isnull().sum()}")
    print(f"Missing area: {traffic_df['area'].isnull().sum()}")
    
    # Safe negative speed checking
    speeds_neg = 0
    if 'avg_speed_kmh' in traffic_df.columns:
        speeds_neg = (traffic_df['avg_speed_kmh'] < 0).sum()
    print(f"Negative speeds: {speeds_neg}")
    
    # Safe vehicle count checking
    vehicles_extreme = 0
    if 'vehicle_count' in traffic_df.columns:
        vehicles_extreme = (traffic_df['vehicle_count'] > 10000).sum()
    print(f"Vehicle count > 10000: {vehicles_extreme}")
    
    print(f"Missing congestion_level: {traffic_df['congestion_level'].isnull().sum()}")
    
    # Check for invalid congestion categories
    invalid_congestion = 0
    if 'congestion_level' in traffic_df.columns:
        valid_categories = ['Low', 'Medium', 'High']
        invalid_congestion = (~traffic_df['congestion_level'].isin(valid_categories) & 
                             traffic_df['congestion_level'].notna()).sum()
    print(f"Invalid congestion categories: {invalid_congestion}")
    
    return weather_file, traffic_file

def generate_data_quality_report(weather_df, traffic_df):
    """Generate a comprehensive data quality report in the same directory as the script"""
    
    report = "# Data Quality Report - Big Data Final Project\n"
    report += "## Weather Impact on Urban Traffic Analysis\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Weather data report
    report += "## Weather Dataset\n"
    report += f"- Total Records: {len(weather_df)}\n"
    report += f"- Duplicate Rows: {weather_df.duplicated().sum()}\n"
    report += f"- Missing Values (Total): {weather_df.isnull().sum().sum()}\n\n"
    
    report += "### Column-wise Issues:\n"
    for col in weather_df.columns:
        missing = weather_df[col].isnull().sum()
        total = len(weather_df)
        report += f"- **{col}**: {missing} missing ({missing/total*100:.1f}%)\n"
    
    # Add specific issues for weather
    report += "\n### Specific Data Quality Issues:\n"
    report += f"- Duplicate weather_id values: {weather_df['weather_id'].duplicated().sum()}\n"
    
    if 'temperature_c' in weather_df.columns:
        temp_outliers = ((weather_df['temperature_c'] < -10) | 
                        (weather_df['temperature_c'] > 35)).sum()
        report += f"- Temperature outliers (< -10°C or > 35°C): {temp_outliers}\n"
    
    if 'humidity' in weather_df.columns:
        humidity_neg = (weather_df['humidity'] < 0).sum()
        humidity_high = (weather_df['humidity'] > 100).sum()
        report += f"- Negative humidity values: {humidity_neg}\n"
        report += f"- Humidity > 100%: {humidity_high}\n"
    
    if 'visibility_m' in weather_df.columns:
        visibility_strings = weather_df['visibility_m'].apply(
            lambda x: isinstance(x, str) and not str(x).replace('.','',1).isdigit()
        ).sum()
        report += f"- Non-numeric visibility strings: {visibility_strings}\n"
    
    # Traffic data report
    report += "\n## Traffic Dataset\n"
    report += f"- Total Records: {len(traffic_df)}\n"
    report += f"- Duplicate Rows: {traffic_df.duplicated().sum()}\n"
    report += f"- Missing Values (Total): {traffic_df.isnull().sum().sum()}\n\n"
    
    report += "### Column-wise Issues:\n"
    for col in traffic_df.columns:
        missing = traffic_df[col].isnull().sum()
        total = len(traffic_df)
        report += f"- **{col}**: {missing} missing ({missing/total*100:.1f}%)\n"
    
    # Add specific issues for traffic
    report += "\n### Specific Data Quality Issues:\n"
    report += f"- Duplicate traffic_id values: {traffic_df['traffic_id'].duplicated().sum()}\n"
    
    if 'avg_speed_kmh' in traffic_df.columns:
        speeds_neg = (traffic_df['avg_speed_kmh'] < 0).sum()
        report += f"- Negative speed values: {speeds_neg}\n"
    
    if 'vehicle_count' in traffic_df.columns:
        vehicles_extreme = (traffic_df['vehicle_count'] > 10000).sum()
        vehicles_neg = (traffic_df['vehicle_count'] < 0).sum()
        report += f"- Vehicle count > 10,000: {vehicles_extreme}\n"
        report += f"- Negative vehicle count: {vehicles_neg}\n"
    
    if 'congestion_level' in traffic_df.columns:
        valid_categories = ['Low', 'Medium', 'High']
        invalid_congestion = (~traffic_df['congestion_level'].isin(valid_categories) & 
                             traffic_df['congestion_level'].notna()).sum()
        report += f"- Invalid congestion categories: {invalid_congestion}\n"
    
    
    # Save report in the script directory
    report_file = os.path.join(SCRIPT_DIR, 'data_quality_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Data quality report saved to: {report_file}")
    return report

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "SYNTHETIC DATA GENERATOR")
    print(" " * 10 + "Big Data Final Project - London Traffic Analysis")
    print("=" * 70)
    print(f"\n📁 Output directory: {SCRIPT_DIR}")
    
    # Generate shared timestamps for better alignment
    print("\n📅 Generating aligned timestamps for both datasets...")
    base_timestamps = generate_base_timestamps(5000)
    
    # Generate datasets with aligned timestamps
    print("\n" + "=" * 70)
    weather_df = generate_synthetic_weather_dataset(5000, base_timestamps)
    
    print("\n" + "=" * 70)
    traffic_df = generate_synthetic_traffic_dataset(5000, base_timestamps)
    
    # Save datasets
    print("\n" + "=" * 70)
    weather_file, traffic_file = save_datasets(weather_df, traffic_df)
    
    # Generate data quality report
    print("\n" + "=" * 70)
    print("📊 Generating comprehensive data quality report...")
    report = generate_data_quality_report(weather_df, traffic_df)
    
    print("\n" + "=" * 70)
    print("✅ DATASETS SUCCESSFULLY GENERATED")
    print("=" * 70)
    print("\n📦 Weather Dataset includes:")
    print("   - ~5,250 records (including 5% duplicates)")
    print("   - Multiple date formats (ISO, UK format, invalid dates)")
    print("   - Missing values in all columns (2-5%)")
    print("   - Temperature outliers (-30°C to 60°C)")
    print("   - Extreme rainfall values (up to 200mm)")
    print("   - Duplicate weather_id values")
    print("   - Inconsistent season labels")
    print("   - Non-numeric visibility strings")
    
    print("\n📦 Traffic Dataset includes:")
    print("   - ~5,300 records (including 6% duplicates)")
    print("   - Missing area/district information")
    print("   - Negative speed values")
    print("   - Extreme vehicle counts (up to 30,000)")
    print("   - Invalid congestion categories")
    print("   - Duplicate traffic_id values")
    print("   - Missing timestamps and invalid date formats")
    
    print("\n🔗 Dataset Alignment:")
    print("   - ~70% of timestamps are aligned between datasets")
    print("   - Ensures easier merging in Phase 4")
    print("   - Both use London as the city")
    
    print("\n" + "=" * 70)
    print("📁 Generated Files:")
    print(f"   1. {os.path.basename(weather_file)}")
    print(f"   2. {os.path.basename(traffic_file)}")
    print("   3. data_quality_report.md")
    print(f"\n📂 All files saved in: {SCRIPT_DIR}")
    print("\n✅ Ready for Phase 1 (Bronze Layer ingestion)")
    print("=" * 70)