"""
Power BI Data Preparation Script
Menggabungkan semua data source dan menyiapkan data untuk import ke Power BI.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("Power BI Data Preparation")
print("=" * 60)

# File paths
BUS_PASSENGER_FILE = "data-jumlah-bus-yang-beroperasi-dan-jumlah-penumpang-layanan-transjakarta-(1765634841830).csv"
HALTE_FILE = "data-halte-transjakarta-(1765634868619).csv"
TRAYEK_FILE = "data-trayek-bus-transjakarta-(1765634860472).csv"
SYNTHETIC_FILE = "synthetic_traffic_jakarta.csv"
OUTPUT_FILE = "powerbi_traffic_data.csv"

# ============================================
# 1. Load & Process Bus Passenger Data
# ============================================
print("\n[1/5] Loading bus & passenger data...")
df_bus = pd.read_csv(BUS_PASSENGER_FILE)
df_bus['date'] = pd.to_datetime(df_bus['periode_data'].astype(str), format='%Y%m', errors='coerce')
df_bus['year'] = df_bus['date'].dt.year
df_bus['quarter'] = df_bus['triwulan']
df_bus['year_quarter'] = df_bus['year'].astype(str) + '-Q' + df_bus['quarter'].astype(str)

# Add calculated columns
df_bus['passengers_per_bus'] = (df_bus['jumlah_penumpang'] / df_bus['jumlah_bus']).round(1)
df_bus['day_name'] = 'N/A'  # Quarterly data
df_bus['is_weekend'] = 'No'
df_bus['is_rush_hour'] = 'No'

print(f"  Bus data loaded: {len(df_bus)} rows")

# ============================================
# 2. Load & Process Halte Data
# ============================================
print("\n[2/5] Loading halte data...")
df_halte = pd.read_csv(HALTE_FILE)
df_halte['data_source'] = 'halte'
print(f"  Halte data loaded: {len(df_halte)} rows")

# ============================================
# 3. Load & Process Trayek Data
# ============================================
print("\n[3/5] Loading trayek data...")
df_trayek = pd.read_csv(TRAYEK_FILE)
df_trayek['data_source'] = 'trayek'
print(f"  Trayek data loaded: {len(df_trayek)} rows")

# ============================================
# 4. Load & Process Synthetic Traffic Data
# ============================================
print("\n[4/5] Loading traffic data...")
df_traffic = pd.read_csv(SYNTHETIC_FILE)
df_traffic['datetime'] = pd.to_datetime(df_traffic['datetime'])
df_traffic['date_only'] = df_traffic['datetime'].dt.date

# Add calculated columns for Power BI
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_names = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

df_traffic['day_name'] = df_traffic['datetime'].dt.dayofweek.map(lambda x: day_names[x])
df_traffic['month_name'] = df_traffic['datetime'].dt.month.map(lambda x: month_names[x-1])
df_traffic['is_weekend'] = df_traffic['datetime'].dt.dayofweek.map(lambda x: 'Yes' if x >= 5 else 'No')
df_traffic['is_rush_hour'] = df_traffic['hour'].map(
    lambda h: 'Yes' if (7 <= h <= 9) or (16 <= h <= 19) else 'No'
)
df_traffic['time_period'] = df_traffic['hour'].map(
    lambda h: 'Morning Rush' if 7 <= h <= 9 else
               'Evening Rush' if 16 <= h <= 19 else
               'Night' if h >= 22 or h <= 5 else
               'Daytime'
)

# LOS description
los_descriptions = {
    'A': 'Free Flow (0-20)',
    'B': 'Reasonably Free (20-40)',
    'C': 'Stable (40-60)',
    'D': 'Approaching Unstable (60-75)',
    'E': 'Unstable (75-90)',
    'F': 'Forced Flow (90-100)'
}
df_traffic['los_description'] = df_traffic['los'].map(los_descriptions)

# Add data source indicator
df_traffic['data_source'] = 'traffic'

print(f"  Traffic data loaded: {len(df_traffic)} rows")

# ============================================
# 5. Combine and Export
# ============================================
print("\n[5/5] Exporting to Power BI format...")

# Select and rename columns for bus data
bus_export = df_bus[[
    'periode_data', 'triwulan', 'jenis_layanan', 'jumlah_bus', 'jumlah_penumpang',
    'year', 'quarter', 'year_quarter', 'passengers_per_bus', 'day_name', 'is_weekend', 'is_rush_hour'
]].copy()
bus_export.columns = [
    'Period', 'Quarter', 'Service_Type', 'Bus_Count', 'Passenger_Count',
    'Year', 'Quarter_Num', 'Year_Quarter', 'Passengers_Per_Bus', 'Day_Name', 'Is_Weekend', 'Is_Rush_Hour'
]

# Select and rename columns for traffic data
traffic_export = df_traffic[[
    'datetime', 'date_only', 'hour', 'day_of_week', 'month', 'year', 'day_name', 'month_name',
    'location', 'latitude', 'longitude', 'traffic_volume', 'los', 'los_description',
    'weather', 'is_holiday', 'is_weekend', 'is_rush_hour', 'time_period'
]].copy()
traffic_export.columns = [
    'DateTime', 'Date', 'Hour', 'Day_of_Week', 'Month', 'Year', 'Day_Name', 'Month_Name',
    'Location', 'Latitude', 'Longitude', 'Traffic_Volume', 'LOS', 'LOS_Description',
    'Weather', 'Is_Holiday', 'Is_Weekend', 'Is_Rush_Hour', 'Time_Period'
]

# Select and rename columns for halte data
halte_export = df_halte[[
    'periode_data', 'wilayah', 'kecamatan', 'kelurahan', 'nama_halte', 'lokasi',
    'koordinat_y', 'koordinat_x'
]].copy()
halte_export.columns = [
    'Period', 'Region', 'District', 'Sub_District', 'Halte_Name', 'Location',
    'Longitude', 'Latitude'
]

# Select and rename columns for trayek data
trayek_export = df_trayek[['periode_data', 'jenis_bus', 'trayek']].copy()
trayek_export.columns = ['Period', 'Bus_Type', 'Route']

# Export each table separately (Power BI works best with multiple tables)
bus_export.to_csv('powerbi_bus_passengers.csv', index=False)
traffic_export.to_csv('powerbi_traffic.csv', index=False)
halte_export.to_csv('powerbi_haltes.csv', index=False)
trayek_export.to_csv('powerbi_routes.csv', index=False)

print("\n" + "=" * 60)
print("Export Complete!")
print("=" * 60)
print("\nFiles created:")
print(f"  1. powerbi_bus_passengers.csv  ({len(bus_export)} rows)")
print(f"  2. powerbi_traffic.csv          ({len(traffic_export)} rows)")
print(f"  3. powerbi_haltes.csv           ({len(halte_export)} rows)")
print(f"  4. powerbi_routes.csv           ({len(trayek_export)} rows)")

print("\n" + "=" * 60)
print("Power BI Import Instructions")
print("=" * 60)
print("\n1. Open Power BI Desktop")
print("2. Click 'Get Data' > 'Text/CSV'")
print("3. Select the CSV files above (one by one)")
print("4. Click 'Load' to import each table")
print("\nTable Relationships:")
print("  - All tables can be connected via 'Year' field")
print("  - Traffic and Haltes can be joined via 'Location' field")

# Statistics
print("\n" + "=" * 60)
print("Data Statistics")
print("=" * 60)

print(f"\n[Bus & Passenger Data]")
print(f"  Total rows: {len(bus_export)}")
print(f"  Year range: {bus_export['Year'].min()} - {bus_export['Year'].max()}")
print(f"  Service types: {bus_export['Service_Type'].nunique()}")
print(f"  Total passengers (latest): {bus_export.groupby('Year')['Passenger_Count'].sum().iloc[-1]:,}")

print(f"\n[Traffic Data]")
print(f"  Total rows: {len(traffic_export):,}")
print(f"  Date range: {traffic_export['Date'].min()} to {traffic_export['Date'].max()}")
print(f"  Locations: {traffic_export['Location'].nunique()}")
print(f"  Average traffic volume: {traffic_export['Traffic_Volume'].mean():.1f}")

print(f"\n[Halte Data]")
print(f"  Total haltes: {len(halte_export)}")
print(f"  Regions: {halte_export['Region'].nunique()}")
print(f"  Districts: {halte_export['District'].nunique()}")

print(f"\n[Route Data]")
print(f"  Total routes: {len(trayek_export)}")
print(f"  Bus types: {trayek_export['Bus_Type'].nunique()}")

print("\n" + "=" * 60)
