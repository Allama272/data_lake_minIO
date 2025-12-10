# Data Quality Report - Big Data Final Project
## Weather Impact on Urban Traffic Analysis

Generated on: 2025-12-10 16:02:37

## Weather Dataset
- Total Records: 5050
- Duplicate Rows: 50
- Missing Values (Total): 1238

### Column-wise Issues:
- **weather_id**: 53 missing (1.0%)
- **date_time**: 107 missing (2.1%)
- **city**: 40 missing (0.8%)
- **season**: 214 missing (4.2%)
- **temperature_c**: 97 missing (1.9%)
- **humidity**: 58 missing (1.1%)
- **rain_mm**: 103 missing (2.0%)
- **wind_speed_kmh**: 157 missing (3.1%)
- **visibility_m**: 151 missing (3.0%)
- **weather_condition**: 161 missing (3.2%)
- **air_pressure_hpa**: 97 missing (1.9%)

### Specific Data Quality Issues:
- Duplicate weather_id values: 269
- Temperature outliers (< -10°C or > 35°C): 52
- Negative humidity values: 21
- Humidity > 100%: 26
- Non-numeric visibility strings: 51

## Traffic Dataset
- Total Records: 5050
- Duplicate Rows: 50
- Missing Values (Total): 5843

### Column-wise Issues:
- **traffic_id**: 34 missing (0.7%)
- **date_time**: 95 missing (1.9%)
- **city**: 83 missing (1.6%)
- **area**: 4724 missing (93.5%)
- **vehicle_count**: 45 missing (0.9%)
- **avg_speed_kmh**: 119 missing (2.4%)
- **accident_count**: 161 missing (3.2%)
- **congestion_level**: 158 missing (3.1%)
- **road_condition**: 165 missing (3.3%)
- **visibility_m**: 259 missing (5.1%)

### Specific Data Quality Issues:
- Duplicate traffic_id values: 185
- Negative speed values: 43
- Vehicle count > 10,000: 71
- Negative vehicle count: 18
- Invalid congestion categories: 250
