"""
Jakarta Traffic Visualization Dashboard
Tableau-style interactive visualization dashboard for Transjakarta and traffic data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Jakarta Traffic Viz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #17becf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
    .insight-title {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .insight-text {
        font-size: 0.9rem;
        color: #555;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Constants
BUS_PASSENGER_FILE = "data-jumlah-bus-yang-beroperasi-dan-jumlah-penumpang-layanan-transjakarta-(1765634841830).csv"
HALTE_FILE = "data-halte-transjakarta-(1765634868619).csv"
SYNTHETIC_FILE = "synthetic_traffic_jakarta.csv"
HALTE_FILE_SHORT = "data-halte-transjakarta-(1765634868619).csv"
TRAYEK_FILE = "data-trayek-bus-transjakarta-(1765634860472).csv"
REKAP_FILE = "data-rekap-lalu-lintas-di-dki-jakarta-(1765634832516).csv"
MODEL_FILE = "traffic_model.pkl"

# Locations for prediction
PREDICTION_LOCATIONS = [
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

# LOS descriptions
LOS_DESCRIPTIONS = {
    'A': 'Free Flow (0-20) - Lalu lintas lancar tanpa hambatan',
    'B': 'Reasonably Free Flow (20-40) - Arus lalu lintas stabil dengan sedikit hambatan',
    'C': 'Stable Flow (40-60) - Arus stabil namun mulai terasa padat',
    'D': 'Approaching Unstable (60-75) - Padat, kecepatan menurun',
    'E': 'Unstable Flow (75-90) - Sangat padat, macet',
    'F': 'Forced Flow (90-100) - Macet total / gridlock'
}


@st.cache_data
def load_bus_passenger_data():
    """Load bus and passenger data."""
    try:
        df = pd.read_csv(BUS_PASSENGER_FILE)
        df['date'] = pd.to_datetime(df['periode_data'].astype(str), format='%Y%m', errors='coerce')
        df['year'] = df['date'].dt.year
        df['quarter'] = df['triwulan']
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        return df
    except:
        return None


@st.cache_data
def load_halte_data():
    """Load halte (bus stop) data."""
    try:
        df = pd.read_csv(HALTE_FILE_SHORT)
        return df
    except:
        return None


@st.cache_data
def load_synthetic_traffic():
    """Load synthetic traffic data."""
    try:
        df = pd.read_csv(SYNTHETIC_FILE)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        return df
    except:
        return None


@st.cache_data
def load_trayek_data():
    """Load bus route data."""
    try:
        df = pd.read_csv(TRAYEK_FILE)
        return df
    except:
        return None


def create_los_color_scale():
    """Create color scale for LOS levels."""
    return {
        'A': '#2ecc71',
        'B': '#3498db',
        'C': '#f39c12',
        'D': '#e67e22',
        'E': '#e74c3c',
        'F': '#8b0000'
    }


def render_insight(title, text, icon="💡"):
    """Render an insight box below a chart."""
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">{icon} {title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_prediction_model():
    """Load the trained traffic prediction model."""
    try:
        model_data = joblib.load(MODEL_FILE)
        return model_data
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


def create_prediction_features(dt, location, weather, avg_traffic=50):
    """Create features for prediction."""
    # Convert to pandas Timestamp
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
        'location_encoded': PREDICTION_LOCATIONS.index(location) if location in PREDICTION_LOCATIONS else 0,
        'weather_encoded': WEATHER_OPTIONS.index(weather) if weather in WEATHER_OPTIONS else 0,
        'traffic_lag_1h': avg_traffic,
        'traffic_lag_24h': avg_traffic,
        'traffic_rolling_mean_24h': avg_traffic,
    }

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


def render_los_badge(los_class):
    """Render LOS badge with color."""
    los_colors = {
        'A': '#2ecc71',
        'B': '#3498db',
        'C': '#f39c12',
        'D': '#e67e22',
        'E': '#e74c3c',
        'F': '#8b0000'
    }
    color = los_colors.get(los_class, '#888')

    st.markdown(f"""
    <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 8px;
                text-align: center; margin: 1rem 0;">
        <div style="font-size: 2rem; font-weight: bold;">LOS {los_class}</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">{LOS_DESCRIPTIONS[los_class]}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard function."""

    # Load all data
    df_bus = load_bus_passenger_data()
    df_halte = load_halte_data()
    df_traffic = load_synthetic_traffic()
    df_trayek = load_trayek_data()

    # Header
    st.markdown('<div class="main-header">Jakarta Traffic Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive visualization of Transjakarta and traffic data</div>', unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.markdown("## Filters")

    # Year filter
    if df_bus is not None:
        years_available = sorted(df_bus['year'].dropna().unique())
        selected_years = st.sidebar.multiselect(
            "Select Years",
            years_available,
            default=years_available
        )
    else:
        selected_years = []

    # Service type filter
    if df_bus is not None:
        service_types = sorted(df_bus['jenis_layanan'].unique())
        selected_services = st.sidebar.multiselect(
            "Select Service Type",
            service_types,
            default=service_types
        )
    else:
        selected_services = []

    # Location filter for traffic
    if df_traffic is not None:
        locations = sorted(df_traffic['location'].unique())
        selected_locations = st.sidebar.multiselect(
            "Select Locations",
            locations,
            default=locations[:5]
        )
    else:
        selected_locations = []

    # Apply filters
    df_bus_filtered = df_bus[
        (df_bus['year'].isin(selected_years)) &
        (df_bus['jenis_layanan'].isin(selected_services))
    ] if df_bus is not None else None

    df_traffic_filtered = df_traffic[
        df_traffic['location'].isin(selected_locations)
    ] if df_traffic is not None else None

    # Key Metrics Row
    st.markdown("---")
    st.markdown("## Key Performance Indicators")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    if df_bus_filtered is not None and len(df_bus_filtered) > 0:
        latest_data = df_bus_filtered[df_bus_filtered['periode_data'] == df_bus_filtered['periode_data'].max()]

        total_passengers = latest_data['jumlah_penumpang'].sum()
        total_buses = latest_data['jumlah_bus'].sum()
        passengers_per_bus = total_passengers / total_buses if total_buses > 0 else 0

        metric_col1.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Total Passengers</div>
        </div>
        """.format(total_passengers), unsafe_allow_html=True)

        metric_col2.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Operating Buses</div>
        </div>
        """.format(total_buses), unsafe_allow_html=True)

        metric_col3.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:,.1f}</div>
            <div class="metric-label">Passengers per Bus</div>
        </div>
        """.format(passengers_per_bus), unsafe_allow_html=True)
    else:
        metric_col1.info("No data available")
        metric_col2.info("No data available")
        metric_col3.info("No data available")

    if df_halte is not None:
        metric_col4.markdown("""
        <div class="metric-box">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Total Haltes</div>
        </div>
        """.format(len(df_halte)), unsafe_allow_html=True)
    else:
        metric_col4.info("No data available")

    # KPI Insights
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">📊 KPI Overview</div>
        <div class="insight-text">
        <b>Total Passengers:</b> Jumlah total penumpang Transjakarta pada periode terakhir yang dipilih.<br>
        <b>Operating Buses:</b> Jumlah bus yang beroperasi aktif melayani penumpang.<br>
        <b>Passengers per Bus:</b> Rata-rata efisiensi penggunaan bus - semakin tinggi semakin efisien.<br>
        <b>Total Haltes:</b> Jumlah total halte/bus stop Transjakarta di seluruh Jakarta.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for different views
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Passenger Analytics", "Traffic Patterns", "Halte & Routes", "Correlations", "Prediction"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        st.markdown("### Data Overview")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Quarterly Passenger Trends")
            if df_bus_filtered is not None and len(df_bus_filtered) > 0:
                quarterly_agg = df_bus_filtered.groupby(['year_quarter', 'date'])['jumlah_penumpang'].sum().reset_index()
                quarterly_agg = quarterly_agg.sort_values('date')

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=quarterly_agg['year_quarter'],
                    y=quarterly_agg['jumlah_penumpang'],
                    mode='lines+markers',
                    name='Total Passengers',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=10)
                ))

                fig.update_layout(
                    xaxis_title="Quarter",
                    yaxis_title="Total Passengers",
                    hovermode='x unified',
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                # Insight
                if len(quarterly_agg) > 1:
                    growth = (quarterly_agg['jumlah_penumpang'].iloc[-1] - quarterly_agg['jumlah_penumpang'].iloc[0]) / quarterly_agg['jumlah_penumpang'].iloc[0] * 100
                    peak = quarterly_agg.loc[quarterly_agg['jumlah_penumpang'].idxmax()]
                    render_insight(
                        "Tren Penumpang Per Kuartal",
                        f"Visualisasi ini menunjukkan <b>fluktuasi jumlah penumpang Transjakarta per kuartal</b>. "
                        f"Pertumbuhan sebesar <b>{growth:+.1f}%</b> dari awal sampai akhir periode. "
                        f"Kuartal dengan penumpang terbanyak adalah <b>{peak['year_quarter']}</b> dengan <b>{peak['jumlah_penumpang']:,}</b> penumpang. "
                        "Pola musiman biasanya terlihat dengan peningkatan pada awal tahun (Q1) dan menjelang akhir tahun (Q4)."
                    )

            else:
                st.warning("No bus data available")

        with col2:
            st.markdown("#### Passengers by Service Type")
            if df_bus_filtered is not None and len(df_bus_filtered) > 0:
                latest = df_bus_filtered[df_bus_filtered['periode_data'] == df_bus_filtered['periode_data'].max()]
                service_data = latest.groupby('jenis_layanan')['jumlah_penumpang'].sum().sort_values(ascending=True)

                fig = go.Figure(go.Bar(
                    x=service_data.values,
                    y=service_data.index,
                    orientation='h',
                    marker=dict(color='steelblue')
                ))

                fig.update_layout(
                    xaxis_title="Total Passengers",
                    yaxis_title="Service Type",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                # Insight
                top_service = service_data.index[-1]
                top_pct = service_data.values[-1] / service_data.sum() * 100
                render_insight(
                    "Distribusi Penumpang per Layanan",
                    f"Layanan <b>{top_service}</b> menjadi kontributor terbesar dengan <b>{top_pct:.1f}%</b> dari total penumpang. "
                    "Layanan BRT (Bus Rapid Transit) dan Angkutan Pengumpan Bus Kecil biasanya mendominasi karena melayani rute utama dan kawasan permukiman. "
                    "Layanan pariwisata dan penugasan memiliki jumlah penumpang lebih sedikit karena bersifat khusus."
                )

            else:
                st.warning("No bus data available")

        # Hourly Traffic Pattern
        st.markdown("#### Average Hourly Traffic Pattern")
        if df_traffic_filtered is not None and len(df_traffic_filtered) > 0:
            hourly_avg = df_traffic_filtered.groupby('hour')['traffic_volume'].mean().reset_index()

            fig = go.Figure()

            # Add hourly line
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg['traffic_volume'],
                mode='lines+markers',
                name='Average Traffic',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))

            # Highlight rush hours
            fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.15,
                         annotation_text="Morning Rush", annotation_position="top")
            fig.add_vrect(x0=16, x1=19, fillcolor="orange", opacity=0.15,
                         annotation_text="Evening Rush", annotation_position="top")

            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Traffic Volume",
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                height=350,
                hovermode='x unified',
                margin=dict(l=0, r=0, t=20, b=40)
            )
            st.plotly_chart(fig, width='stretch')

            # Insight
            morning_peak = hourly_avg[hourly_avg['hour'].between(7, 9)]['traffic_volume'].mean()
            evening_peak = hourly_avg[hourly_avg['hour'].between(16, 19)]['traffic_volume'].mean()
            night_avg = hourly_avg[hourly_avg['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])]['traffic_volume'].mean()

            render_insight(
                "Pola Lalu Lintas Per Jam",
                f"Pola kemacetan Jakarta menunjukkan <b>dua puncak rush hour</b> yang jelas: "
                f"<b>Pagi (07:00-09:00)</b> dengan volume rata-rata <b>{morning_peak:.1f}</b> saat orang berangkat kerja/sekolah, "
                f"dan <b>Sore (16:00-19:00)</b> dengan volume <b>{evening_peak:.1f}</b> saat pulang kerja. "
                f"Volume lalu lintas malam hari hanya <b>{night_avg:.1f}</b> atau sekitar <b>{night_avg/evening_peak*100:.0f}%</b> dari jam sibuk sore."
            )

        else:
            st.warning("No traffic data available")

    # TAB 2: PASSENGER ANALYTICS
    with tab2:
        st.markdown("### Passenger Analytics")

        # Service type pie chart
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Service Type Distribution (Latest Quarter)")
            if df_bus_filtered is not None and len(df_bus_filtered) > 0:
                latest = df_bus_filtered[df_bus_filtered['periode_data'] == df_bus_filtered['periode_data'].max()]
                service_pie = latest.groupby('jenis_layanan')['jumlah_penumpang'].sum().reset_index()

                fig = go.Figure(go.Pie(
                    labels=service_pie['jenis_layanan'],
                    values=service_pie['jumlah_penumpang'],
                    hole=0.4,
                    textinfo='percent+label',
                    textposition='outside'
                ))

                fig.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=20, b=20)
                )
                st.plotly_chart(fig, width='stretch')

                render_insight(
                    "Market Share Layanan",
                    "Pie chart ini menunjukkan <b>proporsi penumpang per jenis layanan</b>. "
                    "Layanan dengan porsi terbesar mengindikasikan rute yang paling padat dan penting. "
                    "Layanan pengumpan (feeder) berperan vital menghubungkan kawasan permukiman dengan koridor utama BRT."
                )

            else:
                st.warning("No data available")

        with col2:
            st.markdown("#### Service Type Over Time")
            if df_bus_filtered is not None and len(df_bus_filtered) > 0:
                service_trend = df_bus_filtered.groupby(['year_quarter', 'jenis_layanan'])['jumlah_penumpang'].sum().reset_index()

                fig = go.Figure()

                for service in service_trend['jenis_layanan'].unique():
                    data = service_trend[service_trend['jenis_layanan'] == service]
                    fig.add_trace(go.Scatter(
                        x=data['year_quarter'],
                        y=data['jumlah_penumpang'],
                        mode='lines+markers',
                        name=service,
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    xaxis_title="Quarter",
                    yaxis_title="Passengers",
                    hovermode='x unified',
                    height=450,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                render_insight(
                    "Evolusi Layanan dari Waktu ke Waktu",
                    "Grafik ini menunjukkan <b>tren penumpang setiap layanan sepanjang periode</b>. "
                    "Perhatikan layanan yang mengalami pertumbuhan signifikan - ini menunjukkan keberhasilan ekspansi rute "
                    "atau meningkatnya preferensi masyarakat terhadap layanan tersebut."
                )

            else:
                st.warning("No data available")

        # Bus Efficiency
        st.markdown("#### Bus Efficiency Analysis")
        if df_bus_filtered is not None and len(df_bus_filtered) > 0:
            efficiency = df_bus_filtered.groupby(['year_quarter', 'jenis_layanan']).agg({
                'jumlah_penumpang': 'sum',
                'jumlah_bus': 'first'
            }).reset_index()
            efficiency['passengers_per_bus'] = efficiency['jumlah_penumpang'] / efficiency['jumlah_bus']

            fig = go.Figure()

            for service in efficiency['jenis_layanan'].unique():
                data = efficiency[efficiency['jenis_layanan'] == service]
                fig.add_trace(go.Scatter(
                    x=data['year_quarter'],
                    y=data['passengers_per_bus'],
                    mode='lines+markers',
                    name=service,
                    line=dict(width=2)
                ))

            fig.update_layout(
                xaxis_title="Quarter",
                yaxis_title="Passengers per Bus",
                hovermode='x unified',
                height=400,
                margin=dict(l=0, r=0, t=20, b=40)
            )
            st.plotly_chart(fig, width='stretch')

            render_insight(
                "Efisiensi Operasional Bus",
                "Metrik <b>Passengers per Bus</b> mengukur seberapa efisien penggunaan armada bus. "
                "Nilai lebih tinggi menunjukkan setiap bus mengangkut lebih banyak penumpang - indikasi baik secara operasional dan finansial. "
                "Layanan BRT biasanya memiliki efisiensi tertinggi karena melayani korpad padat dengan frekuensi tinggi. "
                "Penurunan efisiensi bisa menandakan perlu penyesuaian frekuensi atau rute."
            )

    # TAB 3: TRAFFIC PATTERNS
    with tab3:
        st.markdown("### Traffic Patterns Analysis")

        # LOS Distribution
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Level of Service (LOS) Distribution")
            if df_traffic_filtered is not None and len(df_traffic_filtered) > 0:
                los_dist = df_traffic_filtered['los'].value_counts().sort_index()
                los_colors = create_los_color_scale()

                fig = go.Figure(go.Bar(
                    x=los_dist.index,
                    y=los_dist.values,
                    marker=dict(color=[los_colors.get(los, '#888') for los in los_dist.index])
                ))

                # Add percentage labels
                total = los_dist.sum()
                annotations = [
                    dict(
                        x=los, y=count,
                        text=f"{count/total*100:.1f}%",
                        showarrow=False,
                        yshift=10
                    ) for los, count in los_dist.items()
                ]

                fig.update_layout(
                    xaxis_title="LOS Level",
                    yaxis_title="Count",
                    annotations=annotations,
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                los_good = los_dist.get('A', 0) + los_dist.get('B', 0)
                los_bad = los_dist.get('E', 0) + los_dist.get('F', 0)

                render_insight(
                    "Distribusi Level of Service (LOS)",
                    f"<b>LOS A</b> = Arus bebas, <b>LOS F</b> = Macet total. "
                    f"Dari data, <b>{los_good/total*100:.1f}%</b> kondisi lalu lintas berada pada LOS A-B (lancar), "
                    f"sedangkan <b>{los_bad/total*100:.1f}%</b> pada LOS E-F (sangat padat/macet). "
                    "Pola ini tipikal untuk kota besar - lalu lintas umumnya lancar di luar jam sibuk, "
                    "namun beberapa lokasi dan waktu mengalami kemacetan signifikan."
                )

            else:
                st.warning("No data available")

        with col2:
            st.markdown("#### Traffic by Day of Week")
            if df_traffic_filtered is not None and len(df_traffic_filtered) > 0:
                dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_avg = df_traffic_filtered.groupby('day_of_week')['traffic_volume'].mean().reset_index()
                dow_avg['day_name'] = dow_avg['day_of_week'].map(lambda x: dow_names[x])

                fig = go.Figure(go.Bar(
                    x=dow_avg['day_name'],
                    y=dow_avg['traffic_volume'],
                    marker=dict(color='steelblue')
                ))

                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title="Average Traffic Volume",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                weekday_avg = dow_avg[dow_avg['day_of_week'] < 5]['traffic_volume'].mean()
                weekend_avg = dow_avg[dow_avg['day_of_week'] >= 5]['traffic_volume'].mean()

                render_insight(
                    "Pola Lalu Lintas per Hari",
                    f"<b>Rata-rata weekday (Senin-Jumat): {weekday_avg:.1f}</b> vs "
                    f"<b>weekend (Sabtu-Minggu): {weekend_avg:.1f}</b>. "
                    f"Lalu lintas weekend lebih rendah sekitar <b>{(1-weekend_avg/weekday_avg)*100:.0f}%</b> "
                    "karena aktivitas perkantoran dan sekolah berkurang. Jumat biasanya sedikit lebih lengang "
                    "karena sebagian orang mulai libur akhir pekan."
                )

            else:
                st.warning("No data available")

        # Heatmap: Hour vs Day of Week
        st.markdown("#### Traffic Heatmap: Hour vs Day of Week")
        if df_traffic_filtered is not None and len(df_traffic_filtered) > 0:
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            heatmap_data = df_traffic_filtered.groupby(['day_of_week', 'hour'])['traffic_volume'].mean().unstack()

            fig = go.Figure(go.Heatmap(
                z=heatmap_data.values,
                x=list(range(24)),
                y=[dow_names[i] for i in heatmap_data.index],
                colorscale='RdYlGn_r',
                colorbar=dict(title="Traffic Volume")
            ))

            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
                height=400,
                margin=dict(l=0, r=0, t=20, b=40)
            )
            st.plotly_chart(fig, width='stretch')

            render_insight(
                "Heatmap Pola Kemacetan",
                "Heatmap ini memberikan <b>gambaran visual lengkap pola kemacetan</b>. "
                "Warna merah menunjukkan volume lalu lintas tinggi (macet), hijau menunjukkan lancar. "
                "Perhatikan pola diagonal - rush hour pagi (07:00-09:00) dan sore (16:00-19:00) "
                "jelas terlihat sebagai area merah di hari kerja (Mon-Fri). Weekend cenderung hijau sepanjang hari."
            )

        # Location Comparison
        st.markdown("#### Location Comparison")
        if df_traffic_filtered is not None and len(df_traffic_filtered) > 0:
            location_avg = df_traffic_filtered.groupby('location')['traffic_volume'].mean().sort_values(ascending=True)

            fig = go.Figure(go.Bar(
                x=location_avg.values,
                y=location_avg.index,
                orientation='h',
                marker=dict(color='coral')
            ))

            fig.update_layout(
                xaxis_title="Average Traffic Volume",
                yaxis_title="Location",
                height=max(300, len(location_avg) * 30),
                margin=dict(l=0, r=0, t=20, b=40)
            )
            st.plotly_chart(fig, width='stretch')

            highest = location_avg.index[-1]
            lowest = location_avg.index[0]
            diff_pct = (location_avg.values[-1] - location_avg.values[0]) / location_avg.values[0] * 100

            render_insight(
                "Perbandingan Volume Lalu Lintas per Lokasi",
                f"Lokasi dengan volume tertinggi adalah <b>{highest}</b> sedangkan terendah <b>{lowest}</b>. "
                f"Perbedaannya mencapai <b>{diff_pct:.0f}%</b>. "
                "Lokasi seperti Tol Dalam Kota dan Sudirman-Thamrin biasanya memiliki volume tertinggi "
                "karena merupakan akses utama ke pusat bisnis. Area pinggiran seperti Tangerang dan Bogor "
                "cenderung lebih rendah kecuali pada jam commuting."
            )

    # TAB 4: HALTE & ROUTES
    with tab4:
        st.markdown("### Halte (Bus Stops) & Routes Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Haltes by Region (Wilayah)")
            if df_halte is not None:
                wilayah_counts = df_halte['wilayah'].value_counts().reset_index()
                wilayah_counts.columns = ['Wilayah', 'Count']

                fig = go.Figure(go.Bar(
                    x=wilayah_counts['Wilayah'],
                    y=wilayah_counts['Count'],
                    marker=dict(color='steelblue')
                ))

                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Number of Haltes",
                    xaxis_tickangle=-45,
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=60)
                )
                st.plotly_chart(fig, width='stretch')

                top_region = wilayah_counts.iloc[0]
                total_haltes = len(df_halte)

                render_insight(
                    "Sebaran Halte per Wilayah",
                    f"<b>{top_region['Wilayah']}</b> memiliki jumlah halte terbanyak dengan <b>{top_region['Count']}</b> halte "
                    f"dari total <b>{total_haltes}</b> halte. "
                    "Jakarta Selatan dan Timur biasanya memiliki halte terbanyak karena luas wilayah dan banyaknya kawasan pemukimen. "
                    "Sebaran halte yang merata penting untuk memastikan aksesibilitas transportasi publik yang adil."
                )

            else:
                st.warning("No halte data available")

        with col2:
            st.markdown("#### Top Districts (Kecamatan) by Halte Count")
            if df_halte is not None:
                kecamatan_counts = df_halte['kecamatan'].value_counts().head(10).reset_index()
                kecamatan_counts.columns = ['Kecamatan', 'Count']

                fig = go.Figure(go.Bar(
                    x=kecamatan_counts['Count'],
                    y=kecamatan_counts['Kecamatan'],
                    orientation='h',
                    marker=dict(color='coral')
                ))

                fig.update_layout(
                    xaxis_title="Number of Haltes",
                    yaxis_title="District",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                top_kec = kecamatan_counts.iloc[0]

                render_insight(
                    "Kecamatan dengan Halte Terbanyak",
                    f"Kecamatan <b>{top_kec['Kecamatan']}</b> memiliki <b>{top_kec['Count']}</b> halte terbanyak. "
                    "Kecamatan dengan halte terbanyak biasanya merupakan kawasan dengan kepadatan penduduk tinggi "
                    "atau menjadi hub transportasi penting. Halte yang banyak menunjukkan keterjangkauan transportasi publik yang baik."
                )

            else:
                st.warning("No halte data available")

        # Halte Map
        st.markdown("#### Halte Location Map")
        if df_halte is not None:
            # Create scatter plot
            wilayah_colors = {
                'Kota Adm. Jakarta Pusat': 'red',
                'Kota Adm. Jakarta Utara': 'blue',
                'Kota Adm. Jakarta Barat': 'green',
                'Kota Adm. Jakarta Selatan': 'orange',
                'Kota Adm. Jakarta Timur': 'purple',
            }

            fig = go.Figure()

            for wilayah, color in wilayah_colors.items():
                data = df_halte[df_halte['wilayah'] == wilayah]
                if len(data) > 0:
                    fig.add_trace(go.Scatter(
                        x=data['koordinat_y'].astype(float),
                        y=data['koordinat_x'].astype(float),
                        mode='markers',
                        name=wilayah,
                        marker=dict(size=6, color=color, opacity=0.7),
                        text=data['nama_halte'],
                        hovertemplate='<b>%{text}</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>'
                    ))

            fig.update_layout(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=500,
                hovermode='closest',
                legend=dict(x=0.02, y=0.98),
                margin=dict(l=0, r=0, t=20, b=40)
            )
            st.plotly_chart(fig, width='stretch')

            render_insight(
                "Peta Sebaran Halte Transjakarta",
                "Peta ini menunjukkan <b>lokasi geografis semua halte</b> dengan pewarnaan per wilayah administratif. "
                "Halte terkonsentrasi di area pusat kota dan koridor utama. Hover pada titik untuk melihat nama halte. "
                "Sebaran yang merata mengindikasikan pelayanan transportasi publik yang menjangkau seluruh wilayah Jakarta."
            )

        else:
            st.warning("No halte data available")

        # Route Analysis
        if df_trayek is not None:
            st.markdown("#### Bus Routes Analysis")

            col1, col2 = st.columns([1, 1])

            with col1:
                route_counts = df_trayek['jenis_bus'].value_counts().reset_index()
                route_counts.columns = ['Bus Type', 'Count']

                fig = go.Figure(go.Pie(
                    labels=route_counts['Bus Type'],
                    values=route_counts['Count'],
                    hole=0.3
                ))
                fig.update_layout(
                    title="Routes by Bus Type",
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=20)
                )
                st.plotly_chart(fig, width='stretch')

                render_insight(
                    "Komposisi Armada Berdasarkan Tipe Bus",
                    "Pie chart menunjukkan <b>proporsi tipe bus dalam armada</b>. "
                    "BRT (Bus Rapid Transit) adalah backbone sistem dengan koridor dedicated. "
                    "Bus kecil berperan sebagai feeder menghubungkan kawasan pemukimen ke koridor utama."
                )

            with col2:
                st.info(f"Total Routes: {len(df_trayek)}")
                st.write("Sample Routes:")
                st.dataframe(df_trayek.head(10), use_container_width=True)

    # TAB 5: CORRELATIONS
    with tab5:
        st.markdown("### Correlation & Advanced Analytics")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Weather Impact on Traffic")
            if df_traffic_filtered is not None:
                weather_impact = df_traffic_filtered.groupby('weather')['traffic_volume'].agg(['mean', 'count']).reset_index()
                weather_order = ['clear', 'partly_cloudy', 'cloudy', 'light_rain', 'moderate_rain', 'heavy_rain', 'storm']
                weather_impact = weather_impact.set_index('weather').reindex(weather_order).dropna().reset_index()

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=weather_impact['weather'],
                    y=weather_impact['mean'],
                    name='Avg Traffic',
                    marker=dict(color='steelblue'),
                    text=weather_impact['mean'].round(1),
                    textposition='outside'
                ))

                fig.update_layout(
                    xaxis_title="Weather Condition",
                    yaxis_title="Average Traffic Volume",
                    xaxis_tickangle=-45,
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=60)
                )
                st.plotly_chart(fig, width='stretch')

                clear_traffic = weather_impact[weather_impact['weather'] == 'clear']['mean'].values[0]
                rain_traffic = weather_impact[weather_impact['weather'].isin(['moderate_rain', 'heavy_rain', 'storm'])]['mean'].mean()
                increase_pct = (rain_traffic - clear_traffic) / clear_traffic * 100

                render_insight(
                    "Dampak Cuaca terhadap Lalu Lintas",
                    f"Lalu lintas pada kondisi hujan sedang/lebat meningkat sekitar <b>{increase_pct:.0f}%</b> "
                    f"dibanding cuaca cerah ({clear_traffic:.1f} vs {rain_traffic:.1f}). "
                    "Hujan menyebabkan: (1) kecepatan kendaraan turun, (2) pengendara lebih hati-hati, "
                    "(3) jadi rawan kecelakaan, (4) pengguna motor pindah ke mobil. "
                    "Badai (storm) bisa menimbulkan kemacetan parah hingga 60% lebih tinggi."
                )

            else:
                st.warning("No data available")

        with col2:
            st.markdown("#### Rush Hour vs Non-Rush Hour")
            if df_traffic_filtered is not None:
                df_traffic_copy = df_traffic_filtered.copy()
                df_traffic_copy['is_rush'] = ((df_traffic_copy['hour'] >= 7) &
                                              (df_traffic_copy['hour'] <= 9)) | \
                                             ((df_traffic_copy['hour'] >= 16) &
                                              (df_traffic_copy['hour'] <= 19))

                rush_comparison = df_traffic_copy.groupby('is_rush')['traffic_volume'].mean().reset_index()
                rush_comparison['Period'] = rush_comparison['is_rush'].map({True: 'Rush Hour', False: 'Normal Hours'})

                fig = go.Figure(go.Bar(
                    x=rush_comparison['Period'],
                    y=rush_comparison['traffic_volume'],
                    marker=dict(color=['coral', 'steelblue'])
                ))

                fig.update_layout(
                    xaxis_title="Period",
                    yaxis_title="Average Traffic Volume",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )
                st.plotly_chart(fig, width='stretch')

                rush_vol = rush_comparison[rush_comparison['Period'] == 'Rush Hour']['traffic_volume'].values[0]
                normal_vol = rush_comparison[rush_comparison['Period'] == 'Normal Hours']['traffic_volume'].values[0]
                ratio = rush_vol / normal_vol

                render_insight(
                    "Perbandingan Jam Sibuk vs Normal",
                    f"Volume lalu lintas jam sibuk lebih tinggi <b>{ratio:.2f}x</b> dibanding jam normal "
                    f"({rush_vol:.1f} vs {normal_vol:.1f}). "
                    "Jam sibuk pagi (07:00-09:00) dan sore (16:00-19:00) mewakili peak commuting "
                    "karyawan pelajar dan mahasiswa. Di luar jam ini, lalu lintas cenderung lancer "
                    "meski tetap ada aktivitas ekonomi dan sosial."
                )

            else:
                st.warning("No data available")

        # Summary Data Table
        st.markdown("#### Summary Statistics")

        if df_bus_filtered is not None:
            summary_stats = df_bus_filtered.groupby(['year_quarter', 'jenis_layanan']).agg({
                'jumlah_penumpang': 'sum',
                'jumlah_bus': 'first'
            }).reset_index()
            summary_stats['passengers_per_bus'] = (summary_stats['jumlah_penumpang'] /
                                                    summary_stats['jumlah_bus']).round(1)
            summary_stats.columns = ['Year-Quarter', 'Service Type', 'Total Passengers', 'Buses', 'Pax per Bus']

            st.dataframe(summary_stats, use_container_width=True, hide_index=True)

            render_insight(
                "Ringkasan Statistik Operasional",
                "Tabel ini menyajikan <b>data lengkap per kuartal per layanan</b>. Gunakan untuk analisis mendalam: "
                "identifikasi tren jangka panjang, bandingkan performa layanan, dan temukan insight operasional. "
                "Kolom 'Pax per Bus' adalah KPI efisiensi - semakin tinggi semakin baik."
            )

    # TAB 6: PREDICTION
    with tab6:
        st.markdown("### Prediksi Kemacetan")

        # Load model
        model_data = load_prediction_model()

        if model_data is None:
            st.error("""
            Model prediksi tidak ditemukan. Pastikan file `traffic_model.pkl` ada.

            Untuk membuat model, jalankan:
            ```python
            python traffic_prediction_model.py
            ```
            """)
        else:
            st.info("🤖 Model prediksi menggunakan Random Forest yang dilatih dengan data pola kemacetan Jakarta")

            # Get average traffic from data for lag features
            avg_traffic = 50
            if df_traffic_filtered is not None:
                avg_traffic = df_traffic_filtered['traffic_volume'].mean()

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                pred_date = st.date_input(
                    "Tanggal Prediksi",
                    value=datetime.now().date() + timedelta(days=1),
                    min_value=datetime.now().date(),
                    max_value=datetime.now().date() + timedelta(days=365)
                )

            with col2:
                pred_hour = st.slider(
                    "Jam",
                    min_value=0,
                    max_value=23,
                    value=8
                )

            with col3:
                pred_location = st.selectbox(
                    "Lokasi",
                    PREDICTION_LOCATIONS,
                    index=0
                )

            col4, col5 = st.columns([1, 1])

            with col4:
                pred_weather = st.selectbox(
                    "Cuaca",
                    WEATHER_OPTIONS,
                    index=0,
                    format_func=lambda x: x.replace('_', ' ').title()
                )

            with col5:
                st.write("")  # spacing
                st.write("")  # spacing
                predict_btn = st.button("🔮 Prediksi Traffic", type="primary", use_container_width=True)

            # Prediction result
            if predict_btn or 'prediction_made' not in st.session_state:
                st.markdown("---")

                # Create datetime for prediction
                pred_dt = pd.Timestamp.combine(pred_date, datetime.min.time()).replace(hour=pred_hour)

                # Create features and predict
                features = create_prediction_features(pred_dt, pred_location, pred_weather, avg_traffic)
                prediction = predict_traffic(model_data, features)
                los_class = get_los_class(prediction)

                # Display results
                result_col1, result_col2, result_col3 = st.columns([1, 1, 1])

                with result_col1:
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px;">
                        <div style="font-size: 0.9rem; color: #666;">Volume Lalu Lintas</div>
                        <div style="font-size: 2.5rem; font-weight: bold; color: #1f77b4;">{:.1f}</div>
                        <div style="font-size: 0.8rem; color: #999;">Skala 0-100</div>
                    </div>
                    """.format(prediction), unsafe_allow_html=True)

                with result_col2:
                    render_los_badge(los_class)

                with result_col3:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px;">
                        <div style="font-size: 0.9rem; color: #666;">Waktu</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{pred_dt.strftime("%H:%M")}</div>
                        <div style="font-size: 0.9rem; color: #999;">{pred_dt.strftime("%A, %d %B %Y")}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional info
                info_col1, info_col2, info_col3 = st.columns(3)

                with info_col1:
                    is_rush = (7 <= pred_hour <= 9) or (16 <= pred_hour <= 19)
                    if is_rush:
                        st.warning("⚠️ Jam Sibuk - Diperkirakan padat")
                    else:
                        st.success("✓ Di luar jam sibuk")

                with info_col2:
                    is_weekend = pred_dt.dayofweek >= 5
                    if is_weekend:
                        st.info("📅 Akhir Pekan - Lalu lintas biasanya lebih lancar")
                    else:
                        st.info("📅 Hari Kerja")

                with info_col3:
                    weather_impact = {"clear": 0, "partly_cloudy": 2, "cloudy": 3, "light_rain": 15, "moderate_rain": 25, "heavy_rain": 40, "storm": 60}
                    impact = weather_impact.get(pred_weather, 0)
                    if impact > 20:
                        st.error(f"🌧️ Cuaca buruk - Traffic bisa naik ~{impact}%")
                    elif impact > 10:
                        st.warning(f"🌦️ Hujan ringan - Traffic bisa naik ~{impact}%")
                    else:
                        st.success("☀️ Cuaca mendukung")

                st.markdown("---")

                # Full day prediction
                st.markdown("#### Prediksi Full Day")

                day_start = pred_dt.replace(hour=0)
                hours = list(range(24))
                predictions_day = []

                for hour in hours:
                    dt = day_start.replace(hour=hour)
                    feat = create_prediction_features(dt, pred_location, pred_weather, avg_traffic)
                    pred = predict_traffic(model_data, feat)
                    predictions_day.append(pred)

                pred_df = pd.DataFrame({
                    'Hour': hours,
                    'Predicted_Volume': predictions_day,
                    'LOS': [get_los_class(p) for p in predictions_day]
                })

                # Create prediction chart
                fig = go.Figure()

                los_colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12', 'D': '#e67e22', 'E': '#e74c3c', 'F': '#8b0000'}

                for los in ['F', 'E', 'D', 'C', 'B', 'A']:
                    data = pred_df[pred_df['LOS'] == los]
                    if len(data) > 0:
                        fig.add_trace(go.Bar(
                            x=data['Hour'],
                            y=data['Predicted_Volume'],
                            name=f'LOS {los}',
                            marker_color=los_colors[los],
                            hovertemplate=f"Hour: %{{x}}:00<br>Traffic: %{{y:.1f}}<br>LOS {los}<extra></extra>"
                        ))

                # Highlight selected hour
                fig.add_vline(x=pred_hour, line_dash="dash", line_color="black",
                             annotation_text=f"Selected: {pred_hour}:00")

                fig.update_layout(
                    barmode='stack',
                    xaxis_title="Jam",
                    yaxis_title="Volume Prediksi",
                    height=400,
                    hovermode='x unified',
                    legend_title="Level of Service",
                    margin=dict(l=0, r=0, t=20, b=40)
                )

                st.plotly_chart(fig, width='stretch')

                render_insight(
                    "Prediksi Lalu Lintas 24 Jam",
                    f"Grafik di atas menunjukkan <b>prediksi volume lalu lintas untuk 24 jam</b> di lokasi {pred_location}. "
                    f"Jam dengan warna merah/orange (LOS E-F) menunjukkan kemacetan parah. "
                    f"Perhatikan puncak pagi (07:00-09:00) dan sore (16:00-19:00). "
                    f"Cuaca {pred_weather.replace('_', ' ')} juga mempengaruhi prediksi."
                )

                # Location comparison
                st.markdown("#### Perbandingan Prediksi per Lokasi")

                location_predictions = []
                for loc in PREDICTION_LOCATIONS[:8]:  # Top 8 locations
                    feat = create_prediction_features(pred_dt, loc, pred_weather, avg_traffic)
                    pred = predict_traffic(model_data, feat)
                    location_predictions.append({
                        'Location': loc,
                        'Predicted_Volume': pred,
                        'LOS': get_los_class(pred)
                    })

                loc_pred_df = pd.DataFrame(location_predictions).sort_values('Predicted_Volume', ascending=True)

                fig = go.Figure(go.Bar(
                    x=loc_pred_df['Predicted_Volume'],
                    y=loc_pred_df['Location'],
                    orientation='h',
                    marker=dict(
                        color=[los_colors.get(los, '#888') for los in loc_pred_df['LOS']]
                    ),
                    text=loc_pred_df['LOS'],
                    textposition='outside'
                ))

                fig.update_layout(
                    xaxis_title="Volume Prediksi",
                    yaxis_title="Lokasi",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=40)
                )

                st.plotly_chart(fig, width='stretch')

                render_insight(
                    "Perbandingan Antar Lokasi",
                    "Grafik ini membandingkan prediksi kemacetan di berbagai lokasi pada waktu yang sama. "
                    "Lokasi dengan warna lebih merah menunjukkan prediksi kemacetan lebih parah. "
                    "Gunakan untuk memilih rute alternatif jika lokasi tujuan Anda memiliki LOS merah/orange."
                )

    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666;">Jakarta Traffic Analytics Dashboard | Data Source: Satu Data Jakarta</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
