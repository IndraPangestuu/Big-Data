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
</style>
""", unsafe_allow_html=True)

# Constants
BUS_PASSENGER_FILE = "data-jumlah-bus-yang-beroperasi-dan-jumlah-penumpang-layanan-transjakarta-(1765634841830).csv"
HALTE_FILE = "data-halte-transjakarta-(1765634868619).csv"
SYNTHETIC_FILE = "synthetic_traffic_jakarta.csv"
HALTE_FILE_SHORT = "data-halte-transjakarta-(1765634868619).csv"
TRAYEK_FILE = "data-trayek-bus-transjakarta-(1765634860472).csv"
REKAP_FILE = "data-rekap-lalu-lintas-di-dki-jakarta-(1765634832516).csv"


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

    # Tabs for different views
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Passenger Analytics", "Traffic Patterns", "Halte & Routes", "Correlations"
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
            else:
                st.warning("No data available")

        with col2:
            st.markdown("#### Rush Hour vs Non-Rush Hour")
            if df_traffic_filtered is not None:
                df_traffic_filtered['is_rush'] = ((df_traffic_filtered['hour'] >= 7) &
                                                  (df_traffic_filtered['hour'] <= 9)) | \
                                                 ((df_traffic_filtered['hour'] >= 16) &
                                                  (df_traffic_filtered['hour'] <= 19))

                rush_comparison = df_traffic_filtered.groupby('is_rush')['traffic_volume'].mean().reset_index()
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

    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666;">Jakarta Traffic Analytics Dashboard | Data Source: Satu Data Jakarta</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
