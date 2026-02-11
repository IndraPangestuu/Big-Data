"""
Synthetic Traffic Data Generator for Jakarta
Generates realistic hourly traffic data based on actual traffic patterns in Jakarta.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

# Set random seed for reproducibility
np.random.seed(42)


class JakartaTrafficGenerator:
    """Generate synthetic but realistic traffic data for Jakarta."""

    # Major locations in Jakarta with coordinates
    LOCATIONS = {
        "Sudirman-Thamrin": {"lat": -6.1937, "lon": 106.8230, "base_traffic": 1.2},
        "Monas": {"lat": -6.1754, "lon": 106.8272, "base_traffic": 1.0},
        "Blokm-Bungur": {"lat": -6.2433, "lon": 106.8002, "base_traffic": 1.0},
        "Harmoni": {"lat": -6.1627, "lon": 106.8199, "base_traffic": 1.15},
        "Semanggi": {"lat": -6.2211, "lon": 106.8135, "base_traffic": 1.1},
        "Kuningan": {"lat": -6.2288, "lon": 106.8316, "base_traffic": 1.05},
        "GatotSubroto": {"lat": -6.2326, "lon": 106.8232, "base_traffic": 1.0},
        "Pancoran": {"lat": -6.2417, "lon": 106.8439, "base_traffic": 0.95},
        "Tol Dalam Kota": {"lat": -6.2150, "lon": 106.8500, "base_traffic": 1.3},
        "Tol Jagorawi": {"lat": -6.3000, "lon": 106.8500, "base_traffic": 1.25},
        "Tol JORR": {"lat": -6.2800, "lon": 106.7800, "base_traffic": 1.1},
        "Bekasi-Cawang": {"lat": -6.2500, "lon": 106.9000, "base_traffic": 1.2},
        "Cibubur-Cililitan": {"lat": -6.3000, "lon": 106.8800, "base_traffic": 1.15},
        "Tangerang-Kamal": {"lat": -6.2000, "lon": 106.7000, "base_traffic": 1.1},
        "Bogor-Ciawi": {"lat": -6.6000, "lon": 106.8000, "base_traffic": 1.0},
    }

    # Hourly traffic multiplier (Jakarta pattern)
    # Rush hours: 7-9 AM, 5-7 PM
    HOURLY_PATTERN = {
        0: 0.15,   # 00:00 - Midnight
        1: 0.10,   # 01:00
        2: 0.08,   # 02:00
        3: 0.08,   # 03:00
        4: 0.12,   # 04:00
        5: 0.25,   # 05:00 - Early morning
        6: 0.50,   # 06:00
        7: 0.85,   # 07:00 - Morning rush start
        8: 0.95,   # 08:00 - Peak morning
        9: 0.80,   # 09:00
        10: 0.65,  # 10:00
        11: 0.60,  # 11:00
        12: 0.55,  # 12:00 - Lunch
        13: 0.60,  # 13:00
        14: 0.65,  # 14:00
        15: 0.70,  # 15:00
        16: 0.85,  # 16:00 - Evening rush start
        17: 0.95,  # 17:00 - Peak evening
        18: 0.90,  # 18:00
        19: 0.65,  # 19:00
        20: 0.45,  # 20:00
        21: 0.35,  # 21:00
        22: 0.25,  # 22:00
        23: 0.18,  # 23:00
    }

    # Day of week multiplier (Monday=0, Sunday=6)
    DOW_PATTERN = {
        0: 1.00,  # Monday
        1: 1.00,  # Tuesday
        2: 0.98,  # Wednesday
        3: 0.98,  # Thursday
        4: 0.95,  # Friday - Jumat agak lengang
        5: 0.70,  # Saturday
        6: 0.60,  # Sunday
    }

    # Weather conditions impact on traffic
    WEATHER_IMPACT = {
        "clear": 1.00,
        "partly_cloudy": 1.02,
        "cloudy": 1.03,
        "light_rain": 1.15,
        "moderate_rain": 1.25,
        "heavy_rain": 1.40,
        "storm": 1.60,
    }

    def __init__(self, start_date="2024-01-01", end_date="2025-12-31"):
        """Initialize the traffic generator.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.id_holidays = holidays.CountryHoliday('ID', years=range(2024, 2026))

    def _get_hourly_multiplier(self, hour):
        """Get traffic multiplier for a specific hour."""
        base = self.HOURLY_PATTERN.get(hour, 0.5)
        # Add some randomness
        variation = np.random.normal(0, 0.05)
        return max(0.05, base + variation)

    def _get_dow_multiplier(self, date):
        """Get traffic multiplier for day of week."""
        dow = date.dayofweek
        return self.DOW_PATTERN.get(dow, 1.0)

    def _is_holiday(self, date):
        """Check if date is a holiday."""
        return date in self.id_holidays

    def _get_weather(self, date, hour):
        """Generate weather condition based on seasonal patterns.

        Jakarta has two main seasons:
        - Rainy season: October to April
        - Dry season: May to September
        """
        month = date.month
        hour = hour

        # Rainy season (October - April)
        if month >= 10 or month <= 4:
            # Afternoon rain is common in Jakarta (2-5 PM)
            if 14 <= hour <= 17 and np.random.random() < 0.45:
                weather_probs = ["light_rain", "moderate_rain", "heavy_rain", "storm"]
            elif 12 <= hour <= 18 and np.random.random() < 0.30:
                weather_probs = ["light_rain", "moderate_rain", "cloudy"]
            elif 6 <= hour <= 11 and np.random.random() < 0.20:
                weather_probs = ["light_rain", "cloudy", "partly_cloudy"]
            else:
                weather_probs = ["clear", "partly_cloudy", "cloudy"]
        else:
            # Dry season - mostly clear
            if np.random.random() < 0.15:
                weather_probs = ["partly_cloudy", "cloudy", "light_rain"]
            else:
                weather_probs = ["clear", "partly_cloudy", "clear"]

        weights = [0.4, 0.35, 0.15, 0.1] if len(weather_probs) == 4 else [0.5, 0.35, 0.15]
        return np.random.choice(weather_probs, p=weights[:len(weather_probs)])

    def _calculate_traffic_volume(self, location, date, hour, weather):
        """Calculate traffic volume based on all factors."""
        # Base traffic for location
        base = self.LOCATIONS[location]["base_traffic"]

        # Apply multipliers
        hourly_mult = self._get_hourly_multiplier(hour)
        dow_mult = self._get_dow_multiplier(date)

        # Holiday multiplier
        if self._is_holiday(date):
            holiday_mult = 0.45
        else:
            holiday_mult = 1.0

        # Weather multiplier
        weather_mult = self.WEATHER_IMPACT.get(weather, 1.0)

        # School holiday effect (June-July, December)
        if date.month in [6, 7, 12]:
            school_mult = 0.85
        else:
            school_mult = 1.0

        # Special events (random)
        event_mult = 1.0
        if np.random.random() < 0.02:  # 2% chance of event
            event_mult = 1.2

        # Calculate final traffic volume (0-100 scale)
        base_volume = 100 * base
        traffic = base_volume * hourly_mult * dow_mult * holiday_mult * weather_mult * school_mult * event_mult

        # Add random noise
        noise = np.random.normal(0, 5)
        traffic = max(0, min(100, traffic + noise))

        # Calculate Level of Service (LOS)
        # LOS A: 0-20 (Free flow)
        # LOS B: 20-40 (Reasonably free flow)
        # LOS C: 40-60 (Stable flow)
        # LOS D: 60-75 (Approaching unstable)
        # LOS E: 75-90 (Unstable flow)
        # LOS F: 90-100 (Forced flow / Congestion)
        if traffic < 20:
            los = "A"
        elif traffic < 40:
            los = "B"
        elif traffic < 60:
            los = "C"
        elif traffic < 75:
            los = "D"
        elif traffic < 90:
            los = "E"
        else:
            los = "F"

        return traffic, los

    def generate(self, locations=None, sample_rate="all"):
        """Generate synthetic traffic data.

        Args:
            locations: List of locations to generate data for (default: all)
            sample_rate: "all", "hourly", "daily"

        Returns:
            DataFrame with synthetic traffic data
        """
        if locations is None:
            locations = list(self.LOCATIONS.keys())

        data = []
        current_date = self.start_date

        while current_date <= self.end_date:
            # Generate hourly data for each location
            for hour in range(24):
                weather = self._get_weather(current_date, hour)
                is_holiday = self._is_holiday(current_date)

                for location in locations:
                    traffic, los = self._calculate_traffic_volume(
                        location, current_date, hour, weather
                    )

                    row = {
                        "datetime": current_date.replace(hour=hour),
                        "date": current_date.strftime("%Y-%m-%d"),
                        "hour": hour,
                        "day_of_week": current_date.dayofweek,
                        "month": current_date.month,
                        "year": current_date.year,
                        "location": location,
                        "latitude": self.LOCATIONS[location]["lat"],
                        "longitude": self.LOCATIONS[location]["lon"],
                        "traffic_volume": round(traffic, 2),
                        "los": los,
                        "weather": weather,
                        "is_holiday": 1 if is_holiday else 0,
                        "is_weekend": 1 if current_date.dayofweek >= 5 else 0,
                    }
                    data.append(row)

            current_date += timedelta(days=1)

        df = pd.DataFrame(data)
        return df.sort_values(["datetime", "location"]).reset_index(drop=True)


def generate_weather_data(start_date="2024-01-01", end_date="2025-12-31"):
    """Generate separate weather data for Jakarta."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq="H")

    data = []
    for dt in dates:
        # Rainy season pattern
        month = dt.month
        hour = dt.hour

        if month >= 10 or month <= 4:
            if 14 <= hour <= 17 and np.random.random() < 0.45:
                weather = np.random.choice(["light_rain", "moderate_rain", "heavy_rain", "storm"],
                                          p=[0.4, 0.35, 0.15, 0.1])
            elif 12 <= hour <= 18 and np.random.random() < 0.30:
                weather = np.random.choice(["light_rain", "moderate_rain", "cloudy"],
                                          p=[0.4, 0.4, 0.2])
            else:
                weather = np.random.choice(["clear", "partly_cloudy", "cloudy"],
                                          p=[0.5, 0.35, 0.15])
        else:
            weather = np.random.choice(["clear", "partly_cloudy", "cloudy", "light_rain"],
                                      p=[0.6, 0.25, 0.10, 0.05])

        # Temperature (Jakarta: 24-32°C)
        if hour >= 10 and hour <= 15:
            temp = np.random.normal(31, 2)
        elif hour >= 6 and hour <= 9:
            temp = np.random.normal(28, 2)
        elif hour >= 16 and hour <= 20:
            temp = np.random.normal(29, 2)
        else:
            temp = np.random.normal(26, 2)
        temp = max(22, min(35, temp))

        # Humidity (Jakarta: 65-95%)
        if weather in ["moderate_rain", "heavy_rain", "storm"]:
            humidity = np.random.normal(90, 5)
        elif weather == "light_rain":
            humidity = np.random.normal(85, 5)
        else:
            humidity = np.random.normal(75, 10)
        humidity = max(60, min(100, humidity))

        data.append({
            "datetime": dt,
            "date": dt.strftime("%Y-%m-%d"),
            "hour": hour,
            "weather": weather,
            "temperature_c": round(temp, 1),
            "humidity_percent": round(humidity, 1),
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate traffic data
    print("Generating synthetic traffic data...")
    generator = JakartaTrafficGenerator("2024-01-01", "2025-12-31")

    # Generate for main locations
    main_locations = [
        "Sudirman-Thamrin",
        "Harmoni",
        "Semanggi",
        "Kuningan",
        "Tol Dalam Kota",
        "Tol Jagorawi",
    ]

    df_traffic = generator.generate(locations=main_locations)

    # Save to CSV
    output_path = "synthetic_traffic_jakarta.csv"
    df_traffic.to_csv(output_path, index=False)
    print(f"Traffic data saved to {output_path}")
    print(f"Shape: {df_traffic.shape}")
    print(f"\nColumns: {list(df_traffic.columns)}")
    print(f"\nSample data:")
    print(df_traffic.head(10))

    # Generate weather data
    print("\nGenerating weather data...")
    df_weather = generate_weather_data("2024-01-01", "2025-12-31")
    weather_output = "weather_jakarta.csv"
    df_weather.to_csv(weather_output, index=False)
    print(f"Weather data saved to {weather_output}")
    print(f"Shape: {df_weather.shape}")
    print(f"\nSample weather data:")
    print(df_weather.head(10))

    # Statistics
    print("\n=== TRAFFIC STATISTICS ===")
    print(f"Average traffic volume by hour:")
    print(df_traffic.groupby("hour")["traffic_volume"].mean().round(2))
    print(f"\nAverage traffic volume by day of week:")
    print(df_traffic.groupby("day_of_week")["traffic_volume"].mean().round(2))
    print(f"\nTraffic by LOS:")
    print(df_traffic["los"].value_counts())
    print(f"\nWeather distribution:")
    print(df_traffic["weather"].value_counts())
