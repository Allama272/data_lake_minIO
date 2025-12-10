# Data Quality Report - Big Data Final Project
## Weather Impact on Urban Traffic Analysis

Generated on: 2025-12-07 20:55:04

## Weather Dataset
- Total Records: 5250
- Duplicate Rows: 250
- Missing Values (Total): 2266

### Column-wise Issues:
- **weather_id**: 101 missing (1.9%)
- **date_time**: 557 missing (10.6%)
- **city**: 102 missing (1.9%)
- **season**: 221 missing (4.2%)
- **temperature_c**: 167 missing (3.2%)
- **humidity**: 93 missing (1.8%)
- **rain_mm**: 153 missing (2.9%)
- **wind_speed_kmh**: 198 missing (3.8%)
- **visibility_m**: 257 missing (4.9%)
- **weather_condition**: 263 missing (5.0%)
- **air_pressure_hpa**: 154 missing (2.9%)

### Specific Data Quality Issues:
- Duplicate weather_id values: 603
- Temperature outliers (< -10°C or > 35°C): 254
- Negative humidity values: 128
- Humidity > 100%: 135
- Non-numeric visibility strings: 103

## Traffic Dataset
- Total Records: 5300
- Duplicate Rows: 300
- Missing Values (Total): 6868

### Column-wise Issues:
- **traffic_id**: 128 missing (2.4%)
- **date_time**: 511 missing (9.6%)
- **city**: 140 missing (2.6%)
- **area**: 4923 missing (92.9%)
- **vehicle_count**: 107 missing (2.0%)
- **avg_speed_kmh**: 157 missing (3.0%)
- **accident_count**: 239 missing (4.5%)
- **congestion_level**: 198 missing (3.7%)
- **road_condition**: 212 missing (4.0%)
- **visibility_m**: 253 missing (4.8%)

### Specific Data Quality Issues:
- Duplicate traffic_id values: 622
- Negative speed values: 149
- Vehicle count > 10,000: 241
- Negative vehicle count: 85
- Invalid congestion categories: 485
