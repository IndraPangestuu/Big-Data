"""
Power BI Prediction Script
Script untuk integrasi prediksi traffic di Power BI menggunakan Python.

Cara pakai di Power BI:
1. Buka Power BI Desktop
2. Klik Transform Data > Transform Script
3. Atau buat measure Python yang memanggil script ini
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
MODEL_FILE = "traffic_model.pkl"

# Locations
LOCATIONS = [
    "Sudirman-Thamrin",
    "Harmoni",
    "Semanggi",
    "Kuningan",
    "Tol Dalam Kota",
    "Tol Jagorawi",
    "Tol JORR",
    "Bekasi-Cawang",
    "Cibubur-Cililitan",
    "Tangerang-Kamal",
    "Bogor-Ciawi",
]

# Weather options
WEATHER_OPTIONS = ['clear', 'partly_cloudy', 'cloudy', 'light_rain', 'moderate_rain', 'heavy_rain', 'storm']

# ============================================
# FUNCTIONS
# ============================================

def get_los_class(volume):
    """Convert traffic volume to LOS class."""
    if volume < 20:
        return 'A'
    elif volume < 40:
        return 'B'
    elif volume < 60:
        return 'C'
    elif volume < 75:
        return 'D'
    elif volume < 90:
        return 'E'
    else:
        return 'F'


def get_los_description(los_class):
    """Get LOS description."""
    descriptions = {
        'A': 'Free Flow (0-20) - Lancar',
        'B': 'Reasonably Free (20-40) - Stabil',
        'C': 'Stable (40-60) - Mulai padat',
        'D': 'Approaching Unstable (60-75) - Padat',
        'E': 'Unstable (75-90) - Sangat padat',
        'F': 'Forced Flow (90-100) - Macet total'
    }
    return descriptions.get(los_class, 'Unknown')


def create_features(dt, location, weather, avg_traffic=50):
    """Create features for prediction."""
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)

    features = {
        'hour': dt.hour,
        'day_of_week': dt.dayofweek,
        'month': dt.month,
        'day_of_year': dt.dayofyear,
        'week_of_year': dt.isocalendar()[1],
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        'dow_sin': np.sin(2 * np.pi * dt.dayofweek / 7),
        'dow_cos': np.cos(2 * np.pi * dt.dayofweek / 7),
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12),
        'is_weekend': 1 if dt.dayofweek >= 5 else 0,
        'is_rush_hour': 1 if (7 <= dt.hour <= 9) or (16 <= dt.hour <= 19) else 0,
        'is_rush_hour_morning': 1 if 7 <= dt.hour <= 9 else 0,
        'is_rush_hour_evening': 1 if 16 <= dt.hour <= 19 else 0,
        'is_night': 1 if dt.hour >= 22 or dt.hour <= 5 else 0,
        'location_encoded': LOCATIONS.index(location) if location in LOCATIONS else 0,
        'weather_encoded': WEATHER_OPTIONS.index(weather) if weather in WEATHER_OPTIONS else 0,
        'traffic_lag_1h': avg_traffic,
        'traffic_lag_24h': avg_traffic,
        'traffic_rolling_mean_24h': avg_traffic,
    }

    return features


def predict_traffic(model_file, year, month, day, hour, location, weather, avg_traffic=50):
    """
    Main prediction function for Power BI.

    Parameters:
    -----------
    year : int
        Year (e.g., 2025)
    month : int
        Month (1-12)
    day : int
        Day of month (1-31)
    hour : int
        Hour (0-23)
    location : str
        Location name from LOCATIONS list
    weather : str
        Weather condition from WEATHER_OPTIONS
    avg_traffic : float
        Average traffic value (default 50)

    Returns:
    --------
    DataFrame with prediction results
    """
    # Load model
    model_data = joblib.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']

    # Create datetime
    dt = pd.Timestamp(year, month, day, hour)

    # Create features
    features = create_features(dt, location, weather, avg_traffic)

    # Predict
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    # Get LOS
    los_class = get_los_class(prediction)
    los_desc = get_los_description(los_class)

    # Return as DataFrame
    result = pd.DataFrame([{
        'Year': year,
        'Month': month,
        'Day': day,
        'Hour': hour,
        'DateTime': dt,
        'Location': location,
        'Weather': weather,
        'Predicted_Volume': round(prediction, 2),
        'LOS': los_class,
        'LOS_Description': los_desc
    }])

    return result


def predict_full_day(model_file, year, month, day, location, weather, avg_traffic=50):
    """
    Predict traffic for full 24 hours.

    Returns:
    --------
    DataFrame with hourly predictions
    """
    results = []
    dt = pd.Timestamp(year, month, day)

    for hour in range(24):
        pred = predict_traffic(model_file, year, month, day, hour, location, weather, avg_traffic)
        results.append(pred)

    return pd.concat(results, ignore_index=True)


def predict_location_comparison(model_file, year, month, day, hour, weather, avg_traffic=50):
    """
    Predict traffic for all locations at the same time.

    Returns:
    --------
    DataFrame with predictions for all locations
    """
    results = []

    for location in LOCATIONS:
        pred = predict_traffic(model_file, year, month, day, hour, location, weather, avg_traffic)
        results.append(pred)

    return pd.concat(results, ignore_index=True)


# ============================================
# EXAMPLE USAGE (for testing)
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Power BI Traffic Prediction - Test")
    print("=" * 60)

    # Test single prediction
    print("\n[1] Single Prediction Test:")
    result = predict_traffic(
        model_file=MODEL_FILE,
        year=2025, month=3, day=15, hour=8,
        location="Sudirman-Thamrin",
        weather="clear"
    )
    print(result.to_string(index=False))

    # Test full day prediction
    print("\n[2] Full Day Prediction Test:")
    full_day = predict_full_day(
        model_file=MODEL_FILE,
        year=2025, month=3, day=15,
        location="Sudirman-Thamrin",
        weather="clear"
    )
    print(f"Predicted {len(full_day)} hours for {full_day['DateTime'].iloc[0].date()}")
    print(f"Average volume: {full_day['Predicted_Volume'].mean():.1f}")
    print(f"Peak hour: {full_day.loc[full_day['Predicted_Volume'].idxmax(), 'Hour']:}:00")
    print(f"Peak volume: {full_day['Predicted_Volume'].max():.1f}")

    # Test location comparison
    print("\n[3] Location Comparison Test:")
    comparison = predict_location_comparison(
        model_file=MODEL_FILE,
        year=2025, month=3, day=15, hour=8,
        weather="clear"
    )
    print(comparison[['Location', 'Predicted_Volume', 'LOS']].to_string(index=False))

    # Export test results
    full_day.to_csv('powerbi_prediction_sample.csv', index=False)
    comparison.to_csv('powerbi_location_comparison.csv', index=False)

    print("\n" + "=" * 60)
    print("Test files created:")
    print("  - powerbi_prediction_sample.csv")
    print("  - powerbi_location_comparison.csv")
    print("=" * 60)

    print("\nPower BI Python Integration Guide:")
    print("1. Enable Python in Power BI: Options > Options > Python Scripting")
    print("2. Add this script path in Power BI options")
    print("3. Create measures using dataset.RunPython() function")
