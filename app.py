"""
Jakarta Traffic Prediction Dashboard
A Streamlit app for predicting and visualizing traffic congestion in Jakarta.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Page config
st.set_page_config(
    page_title="Jakarta Traffic Predictor",
    page_icon="://",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .los-a { background-color: #2ecc71; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold; }
    .los-b { background-color: #3498db; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold; }
    .los-c { background-color: #f39c12; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold; }
    .los-d { background-color: #e67e22; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold; }
    .los-e { background-color: #e74c3c; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold; }
    .los-f { background-color: #8b0000; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = None

# Constants
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

WEATHER_OPTIONS = ['clear', 'partly_cloudy', 'cloudy', 'light_rain', 'moderate_rain', 'heavy_rain', 'storm']

LOS_DESCRIPTIONS = {
    'A': 'Free Flow - Traffic flows freely with no restrictions',
    'B': 'Reasonably Free Flow - Slight restrictions',
    'C': 'Stable Flow - Acceptable speeds with some restrictions',
    'D': 'Approaching Unstable - Reduced speeds, noticeable congestion',
    'E': 'Unstable Flow - Severe congestion, very low speeds',
    'F': 'Forced Flow - Complete gridlock'
}

LOS_COLORS = {
    'A': '#2ecc71',
    'B': '#3498db',
    'C': '#f39c12',
    'D': '#e67e22',
    'E': '#e74c3c',
    'F': '#8b0000'
}


@st.cache_resource
def load_model():
    """Load the trained traffic prediction model."""
    try:
        model_data = joblib.load('traffic_model.pkl')
        return model_data
    except Exception as e:
        return None


@st.cache_data
def load_traffic_data():
    """Load traffic data for visualization."""
    try:
        df = pd.read_csv('synthetic_traffic_jakarta.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        return None


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


def create_features(dt, location, weather):
    """Create features for prediction."""
    # Convert to pandas datetime for dayofweek attribute
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
        'traffic_lag_1h': 50.0,
        'traffic_lag_24h': 50.0,
        'traffic_rolling_mean_24h': 50.0,
    }

    # Get average traffic values from data if available
    if st.session_state.traffic_data is not None:
        avg_traffic = st.session_state.traffic_data['traffic_volume'].mean()
        features['traffic_lag_1h'] = avg_traffic
        features['traffic_lag_24h'] = avg_traffic
        features['traffic_rolling_mean_24h'] = avg_traffic

    return features


def predict_traffic(model_data, features):
    """Make traffic prediction."""
    model = model_data['model']
    scaler = model_data['scaler']

    feature_cols = model_data['feature_cols']
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    return prediction


def main():
    """Main app function."""

    # Load model and data
    model_data = load_model()
    if model_data:
        st.session_state.model_loaded = True

    traffic_df = load_traffic_data()
    if traffic_df is not None:
        st.session_state.traffic_data = traffic_df

    # Header
    st.markdown('<div class="main-header">Jakarta Traffic Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict traffic congestion in Jakarta based on time, location, and weather</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## Prediction Parameters")

    # Date and time input
    default_datetime = datetime.now() + timedelta(days=1)
    selected_datetime = st.sidebar.datetime_input(
        "Select Date & Time",
        value=default_datetime,
        max_value=datetime.now() + timedelta(days=365)
    )

    # Location selection
    selected_location = st.sidebar.selectbox(
        "Select Location",
        LOCATIONS,
        index=0
    )

    # Weather selection
    selected_weather = st.sidebar.selectbox(
        "Weather Condition",
        WEATHER_OPTIONS,
        index=0,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    # Prediction button
    predict_btn = st.sidebar.button("Predict Traffic", type="primary", use_container_width=True)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Prediction Result")

        if predict_btn and model_data:
            with st.spinner("Predicting traffic..."):
                features = create_features(selected_datetime, selected_location, selected_weather)
                prediction = predict_traffic(model_data, features)
                los_class = get_los_class(prediction)

                # Display prediction
                st.markdown(f'<div class="los-{los_class.lower()}" style="font-size: 2rem; margin: 1rem 0;">LOS {los_class}</div>', unsafe_allow_html=True)
                st.metric("Predicted Traffic Volume", f"{prediction:.1f}")

                # LOS Description
                st.info(f"**{LOS_DESCRIPTIONS[los_class]}**")

                # Additional info
                st.markdown("---")
                st.markdown("#### Prediction Details")
                details_col1, details_col2 = st.columns(2)
                with details_col1:
                    st.write("**Date:**", selected_datetime.strftime("%Y-%m-%d"))
                    st.write("**Time:**", selected_datetime.strftime("%H:%M"))
                with details_col2:
                    st.write("**Day:**", selected_datetime.strftime("%A"))
                    st.write("**Weather:**", selected_weather.replace('_', ' ').title())

                # Rush hour indicator
                is_rush = (7 <= selected_datetime.hour <= 9) or (16 <= selected_datetime.hour <= 19)
                if is_rush:
                    st.warning("Rush Hour Alert: This is typically a high-traffic period!")
                elif selected_datetime.hour >= 22 or selected_datetime.hour <= 5:
                    st.info("Night Time: Traffic is usually lighter at night.")

        elif not model_data:
            st.error("Model not found. Please train the model first using traffic_prediction_model.py")
        else:
            st.info("Click 'Predict Traffic' to see the prediction")

    with col2:
        st.markdown("### Hourly Traffic Pattern")

        if traffic_df is not None:
            # Calculate hourly pattern
            hourly_data = traffic_df.groupby('hour')['traffic_volume'].mean().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['traffic_volume'],
                mode='lines+markers',
                name='Average Traffic',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))

            # Highlight rush hours
            fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.2, layer="below", line_width=0)
            fig.add_vrect(x0=16, x1=19, fillcolor="orange", opacity=0.2, layer="below", line_width=0)

            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Traffic Volume",
                hovermode='x unified',
                height=300,
                margin=dict(l=0, r=0, t=20, b=40)
            )

            st.plotly_chart(fig, width='stretch')

            # Current time indicator
            current_hour = selected_datetime.hour
            current_avg = hourly_data[hourly_data['hour'] == current_hour]['traffic_volume'].values
            if len(current_avg) > 0:
                st.metric(f"Average at {current_hour}:00", f"{current_avg[0]:.1f}")

    # Full day prediction
    st.markdown("---")
    st.markdown("### Full Day Prediction")

    if model_data:
        # Generate predictions for the entire selected day
        day_start = selected_datetime.replace(hour=0, minute=0, second=0)
        hours = list(range(24))
        predictions = []

        for hour in hours:
            dt = day_start.replace(hour=hour)
            features = create_features(dt, selected_location, selected_weather)
            pred = predict_traffic(model_data, features)
            predictions.append(pred)

        # Create prediction chart
        pred_df = pd.DataFrame({
            'Hour': hours,
            'Predicted_Volume': predictions,
            'LOS': [get_los_class(p) for p in predictions]
        })

        fig = go.Figure()

        # Add colored bars based on LOS
        for los in ['F', 'E', 'D', 'C', 'B', 'A']:
            data = pred_df[pred_df['LOS'] == los]
            if len(data) > 0:
                fig.add_trace(go.Bar(
                    x=data['Hour'],
                    y=data['Predicted_Volume'],
                    name=f'LOS {los}',
                    marker_color=LOS_COLORS[los],
                    hovertemplate=f"Hour: %{{x}}:00<br>Traffic: %{{y:.1f}}<br>LOS {los}<extra></extra>"
                ))

        fig.update_layout(
            barmode='stack',
            xaxis_title="Hour of Day",
            yaxis_title="Predicted Traffic Volume",
            height=400,
            hovermode='x unified',
            legend_title="Level of Service",
            margin=dict(l=0, r=0, t=20, b=40)
        )

        st.plotly_chart(fig, width='stretch')

    # Location comparison
    st.markdown("---")
    st.markdown("### Location Comparison")

    if model_data and traffic_df is not None:
        selected_hour = st.slider("Select Hour for Comparison", 0, 23, 8)

        comparison_data = []
        for loc in LOCATIONS[:6]:  # Show top 6 locations
            dt = selected_datetime.replace(hour=selected_hour)
            features = create_features(dt, loc, selected_weather)
            pred = predict_traffic(model_data, features)
            los = get_los_class(pred)
            comparison_data.append({
                'Location': loc,
                'Predicted_Volume': pred,
                'LOS': los
            })

        comp_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comp_df,
            x='Location',
            y='Predicted_Volume',
            color='LOS',
            color_discrete_map=LOS_COLORS,
            title=f'Traffic Prediction at {selected_hour}:00',
            height=350
        )
        fig.update_layout(xaxis_tickangle=-45, margin=dict(l=0, r=0, t=40, b=60))
        st.plotly_chart(fig, width='stretch')

    # Statistics section
    st.markdown("---")
    st.markdown("### Traffic Statistics")

    if traffic_df is not None:
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Total Records", f"{traffic_df.shape[0]:,}")
        with stat_col2:
            st.metric("Avg Traffic Volume", f"{traffic_df['traffic_volume'].mean():.1f}")
        with stat_col3:
            st.metric("Peak Hour", f"{traffic_df.groupby('hour')['traffic_volume'].mean().idxmax()}:00")
        with stat_col4:
            peak_los = traffic_df['los'].value_counts().idxmax()
            st.metric("Most Common LOS", f"LOS {peak_los}")

    # Model info
    with st.expander("About the Model"):
        st.markdown("""
        **Model Information:**
        - **Algorithm:** Random Forest Regressor
        - **Features:** Temporal (hour, day, month), Location, Weather, Lag features
        - **Training Data:** Synthetic traffic data based on Jakarta patterns
        - **Output:** Traffic Volume (0-100 scale) and Level of Service (LOS A-F)

        **Level of Service (LOS) Scale:**
        - **LOS A:** Free Flow (0-20) - No congestion
        - **LOS B:** Reasonably Free Flow (20-40) - Light traffic
        - **LOS C:** Stable Flow (40-60) - Moderate traffic
        - **LOS D:** Approaching Unstable (60-75) - Heavy traffic
        - **LOS E:** Unstable Flow (75-90) - Severe congestion
        - **LOS F:** Forced Flow (90-100) - Gridlock

        **Note:** This is a demonstration model using synthetic data. For production use,
        train the model with real traffic data from Jakarta's transportation authorities.
        """)

    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666;">Jakarta Traffic Predictor | Built with Streamlit</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
