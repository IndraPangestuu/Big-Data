"""
Traffic Prediction Model for Jakarta
A comprehensive machine learning pipeline for predicting traffic congestion.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class JakartaTrafficPredictor:
    """Traffic prediction model for Jakarta."""

    def __init__(self, data_path=None):
        """Initialize the traffic predictor.

        Args:
            data_path: Path to the traffic data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None

    def load_data(self, data_path=None):
        """Load and preprocess traffic data.

        Args:
            data_path: Path to the traffic data CSV file

        Returns:
            DataFrame with loaded data
        """
        if data_path:
            self.data_path = data_path

        if not self.data_path:
            # Generate synthetic data if no path provided
            print("No data path provided. Generating synthetic data...")
            from synthetic_traffic_data import JakartaTrafficGenerator
            generator = JakartaTrafficGenerator("2024-01-01", "2025-12-31")
            locations = ["Sudirman-Thamrin", "Harmoni", "Semanggi", "Kuningan", "Tol Dalam Kota"]
            self.df = generator.generate(locations=locations)
        else:
            self.df = pd.read_csv(self.data_path)
            if 'datetime' in self.df.columns:
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])

        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def engineer_features(self, df=None):
        """Create features for machine learning.

        Features created:
        - Temporal: hour, day_of_week, month, is_weekend, is_holiday, is_rush_hour
        - Location: location_encoded
        - Weather: weather_encoded
        - Lag features: traffic_lag_1h, traffic_lag_24h
        - Rolling features: traffic_rolling_mean_24h
        """
        if df is None:
            df = self.df.copy()

        # Ensure datetime is in datetime format
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns and 'hour' in df.columns:
            df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

        # Temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)

        # Cyclical features for periodic patterns
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_rush_hour_evening'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_rush_hour_morning'] | df['is_rush_hour_evening']).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

        # Encode location if exists
        if 'location' in df.columns:
            le_location = LabelEncoder()
            df['location_encoded'] = le_location.fit_transform(df['location'])
        else:
            df['location_encoded'] = 0

        # Encode weather if exists
        if 'weather' in df.columns:
            weather_order = ['clear', 'partly_cloudy', 'cloudy', 'light_rain', 'moderate_rain', 'heavy_rain', 'storm']
            le_weather = LabelEncoder()
            le_weather.classes_ = np.array(weather_order)
            df['weather_encoded'] = df['weather'].map({w: i for i, w in enumerate(weather_order)})
            # Fill missing with 0
            df['weather_encoded'] = df['weather_encoded'].fillna(0).astype(int)
        else:
            df['weather_encoded'] = 0

        # Lag features (requires sorting)
        df = df.sort_values(['location', 'datetime'])
        df['traffic_lag_1h'] = df.groupby('location')['traffic_volume'].shift(1)
        df['traffic_lag_24h'] = df.groupby('location')['traffic_volume'].shift(24)

        # Rolling mean
        df['traffic_rolling_mean_24h'] = df.groupby('location')['traffic_volume'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )

        # Fill NaN from lag features
        df['traffic_lag_1h'] = df['traffic_lag_1h'].fillna(df['traffic_volume'].mean())
        df['traffic_lag_24h'] = df['traffic_lag_24h'].fillna(df['traffic_volume'].mean())

        self.df = df
        return df

    def prepare_data(self, target_col='traffic_volume', test_size=0.2):
        """Prepare data for training.

        Args:
            target_col: Name of target column
            test_size: Proportion of data for testing
        """
        # Feature columns
        feature_cols = [
            'hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_rush_hour', 'is_rush_hour_morning', 'is_rush_hour_evening',
            'is_night', 'location_encoded', 'weather_encoded',
            'traffic_lag_1h', 'traffic_lag_24h', 'traffic_rolling_mean_24h'
        ]

        # Filter available columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]

        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()

        # Split data (time-series split)
        split_idx = int(len(X) * (1 - test_size))
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {len(feature_cols)}")

        return feature_cols

    def train_model(self, model_type='random_forest', tune_hyperparameters=False):
        """Train the prediction model.

        Args:
            model_type: 'random_forest', 'gradient_boosting', 'linear', 'ridge'
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        if model_type == 'random_forest':
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                }
                base_model = RandomForestRegressor(random_state=42)
                self.model = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, verbose=1)
            else:
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )

        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )

        elif model_type == 'linear':
            self.model = LinearRegression()

        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)

        print(f"Training {model_type} model...")
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Training completed!")

        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        return self.model

    def evaluate(self):
        """Evaluate model performance."""
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)

        results = {
            'train': {
                'mae': mean_absolute_error(self.y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'r2': r2_score(self.y_train, y_pred_train)
            },
            'test': {
                'mae': mean_absolute_error(self.y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'r2': r2_score(self.y_test, y_pred_test)
            }
        }

        print("\n=== MODEL EVALUATION ===")
        print(f"Training Set:")
        print(f"  MAE:  {results['train']['mae']:.2f}")
        print(f"  RMSE: {results['train']['rmse']:.2f}")
        print(f"  R²:   {results['train']['r2']:.4f}")
        print(f"\nTest Set:")
        print(f"  MAE:  {results['test']['mae']:.2f}")
        print(f"  RMSE: {results['test']['rmse']:.2f}")
        print(f"  R²:   {results['test']['r2']:.4f}")

        return results, y_pred_test

    def predict(self, datetime_str, location, weather='clear'):
        """Make prediction for specific datetime and location.

        Args:
            datetime_str: Datetime string (YYYY-MM-DD HH:MM:SS)
            location: Location name
            weather: Weather condition

        Returns:
            Predicted traffic volume
        """
        dt = pd.to_datetime(datetime_str)

        # Create feature dict
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
            'location_encoded': 0,  # Default
            'weather_encoded': 0,   # Default
            'traffic_lag_1h': self.df['traffic_volume'].mean(),
            'traffic_lag_24h': self.df['traffic_volume'].mean(),
            'traffic_rolling_mean_24h': self.df['traffic_volume'].mean(),
        }

        # Get location encoding if available
        if 'location' in self.df.columns:
            locations = self.df['location'].unique()
            for i, loc in enumerate(locations):
                if loc == location:
                    features['location_encoded'] = i
                    break

        # Get weather encoding
        weather_order = ['clear', 'partly_cloudy', 'cloudy', 'light_rain', 'moderate_rain', 'heavy_rain', 'storm']
        for i, w in enumerate(weather_order):
            if w == weather:
                features['weather_encoded'] = i
                break

        # Create DataFrame and scale
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]

        # Get LOS
        if prediction < 20:
            los = "A (Free Flow)"
        elif prediction < 40:
            los = "B (Reasonably Free Flow)"
        elif prediction < 60:
            los = "C (Stable Flow)"
        elif prediction < 75:
            los = "D (Approaching Unstable)"
        elif prediction < 90:
            los = "E (Unstable Flow)"
        else:
            los = "F (Congestion)"

        return {
            'datetime': datetime_str,
            'location': location,
            'weather': weather,
            'predicted_volume': round(prediction, 2),
            'los': los
        }

    def plot_predictions(self, y_pred, n_samples=500):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(14, 6))

        y_test_sample = self.y_test.iloc[:n_samples].values
        y_pred_sample = y_pred[:n_samples]

        plt.subplot(1, 2, 1)
        plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--', lw=2)
        plt.xlabel('Actual Traffic Volume')
        plt.ylabel('Predicted Traffic Volume')
        plt.title('Actual vs Predicted Traffic')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(y_test_sample, label='Actual', alpha=0.7)
        plt.plot(y_pred_sample, label='Predicted', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Traffic Volume')
        plt.title('Traffic Prediction Timeline')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('traffic_predictions.png', dpi=150)
        print("\nPlot saved to traffic_predictions.png")

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance."""
        if self.feature_importance is None:
            print("Feature importance not available for this model.")
            return

        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance.head(top_n)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        print("Feature importance plot saved to feature_importance.png")

    def plot_hourly_pattern(self):
        """Plot average hourly traffic pattern."""
        if self.df is None:
            print("No data available.")
            return

        hourly = self.df.groupby('hour')['traffic_volume'].mean()

        plt.figure(figsize=(12, 5))
        plt.plot(hourly.index, hourly.values, marker='o', linewidth=2)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Traffic Volume')
        plt.title('Average Hourly Traffic Pattern (Jakarta)')
        plt.xticks(range(24))
        plt.grid(True)

        # Highlight rush hours
        plt.axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush')
        plt.axvspan(16, 19, alpha=0.2, color='orange', label='Evening Rush')
        plt.legend()

        plt.tight_layout()
        plt.savefig('hourly_pattern.png', dpi=150)
        print("Hourly pattern plot saved to hourly_pattern.png")

    def save_model(self, path='traffic_model.pkl'):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.X_train.columns.tolist() if self.X_train is not None else [],
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path='traffic_model.pkl'):
        """Load trained model."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        print(f"Model loaded from {path}")


def run_full_pipeline():
    """Run the complete traffic prediction pipeline."""

    print("=" * 60)
    print("JAKARTA TRAFFIC PREDICTION MODEL")
    print("=" * 60)

    # Initialize predictor
    predictor = JakartaTrafficPredictor()

    # Step 1: Load data
    print("\n[Step 1/6] Loading data...")
    predictor.load_data()

    # Step 2: Engineer features
    print("\n[Step 2/6] Engineering features...")
    predictor.engineer_features()

    # Step 3: Prepare data
    print("\n[Step 3/6] Preparing data for training...")
    feature_cols = predictor.prepare_data()

    # Step 4: Train model
    print("\n[Step 4/6] Training model...")
    predictor.train_model(model_type='random_forest', tune_hyperparameters=False)

    # Step 5: Evaluate
    print("\n[Step 5/6] Evaluating model...")
    results, y_pred = predictor.evaluate()

    # Step 6: Visualize
    print("\n[Step 6/6] Creating visualizations...")
    predictor.plot_predictions(y_pred)
    predictor.plot_feature_importance()
    predictor.plot_hourly_pattern()

    # Save model
    predictor.save_model('traffic_model.pkl')

    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    scenarios = [
        ("2025-03-15 08:00:00", "Sudirman-Thamrin", "clear"),  # Morning rush weekday
        ("2025-03-15 18:00:00", "Sudirman-Thamrin", "moderate_rain"),  # Evening rush with rain
        ("2025-03-16 08:00:00", "Sudirman-Thamrin", "clear"),  # Weekend morning
        ("2025-03-17 08:00:00", "Harmoni", "heavy_rain"),  # Heavy rain morning
        ("2025-03-17 14:00:00", "Tol Dalam Kota", "clear"),  # Afternoon
    ]

    for dt, loc, weather in scenarios:
        result = predictor.predict(dt, loc, weather)
        print(f"\n{dt} | {loc} | {weather}")
        print(f"  Predicted Volume: {result['predicted_volume']}")
        print(f"  LOS: {result['los']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED!")
    print("=" * 60)

    return predictor


if __name__ == "__main__":
    # Run the full pipeline
    predictor = run_full_pipeline()
