"""
Integrate Real Transjakarta Data with Synthetic Traffic Data
This script combines real quarterly Transjakarta data with synthetic hourly data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# File paths
BUS_PASSENGER_FILE = "data-jumlah-bus-yang-beroperasi-dan-jumlah-penumpang-layanan-transjakarta-(1765634841830).csv"
HALTE_FILE = "data-halte-transjakarta-(1765634868619).csv"
SYNTHETIC_FILE = "synthetic_traffic_jakarta.csv"
OUTPUT_FILE = "merged_traffic_transjakarta.csv"


class TransjakartaDataIntegrator:
    """Integrate real Transjakarta data with synthetic traffic data."""

    def __init__(self):
        self.df_bus = None
        self.df_halte = None
        self.df_synthetic = None
        self.df_merged = None

    def load_data(self):
        """Load all data sources."""
        print("Loading data...")

        # Load bus and passenger data
        if os.path.exists(BUS_PASSENGER_FILE):
            self.df_bus = pd.read_csv(BUS_PASSENGER_FILE)
            print(f"  Bus/Passenger data: {self.df_bus.shape[0]} rows")
        else:
            print(f"  Warning: {BUS_PASSENGER_FILE} not found")

        # Load halte data
        if os.path.exists(HALTE_FILE):
            self.df_halte = pd.read_csv(HALTE_FILE)
            print(f"  Halte data: {self.df_halte.shape[0]} rows")
        else:
            print(f"  Warning: {HALTE_FILE} not found")

        # Load synthetic data
        if os.path.exists(SYNTHETIC_FILE):
            self.df_synthetic = pd.read_csv(SYNTHETIC_FILE)
            self.df_synthetic['datetime'] = pd.to_datetime(self.df_synthetic['datetime'])
            print(f"  Synthetic traffic data: {self.df_synthetic.shape[0]} rows")
        else:
            print(f"  Warning: {SYNTHETIC_FILE} not found")

    def process_bus_data(self):
        """Process bus and passenger data to get quarterly statistics."""
        if self.df_bus is None:
            return None

        print("\nProcessing bus and passenger data...")

        # Clean and transform data
        df = self.df_bus.copy()

        # Convert periode_data to datetime
        df['date'] = pd.to_datetime(df['periode_data'].astype(str), format='%Y%m', errors='coerce')

        # Extract year and quarter
        df['year'] = df['date'].dt.year
        df['quarter'] = df['triwulan']
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

        # Aggregate by year_quarter and service type
        agg = df.groupby(['year_quarter', 'year', 'quarter', 'jenis_layanan']).agg({
            'jumlah_bus': 'first',
            'jumlah_penumpang': 'first'
        }).reset_index()

        # Calculate totals per quarter
        quarterly_totals = df.groupby(['year_quarter', 'year', 'quarter']).agg({
            'jumlah_bus': 'sum',
            'jumlah_penumpang': 'sum'
        }).reset_index()
        quarterly_totals['jenis_layanan'] = 'TOTAL'

        # Combine
        self.df_bus_processed = pd.concat([agg, quarterly_totals], ignore_index=True)

        # Calculate passengers per bus ratio
        self.df_bus_processed['passengers_per_bus'] = (
            self.df_bus_processed['jumlah_penumpang'] / self.df_bus_processed['jumlah_bus']
        ).round(1)

        print(f"  Processed {len(self.df_bus_processed)} records")

        return self.df_bus_processed

    def process_halte_data(self):
        """Process halte (bus stop) data."""
        if self.df_halte is None:
            return None

        print("\nProcessing halte data...")

        df = self.df_halte.copy()

        # Count haltes by wilayah (city/region)
        halte_by_wilayah = df.groupby('wilayah').agg({
            'nama_halte': 'count',
            'koordinat_x': 'mean',
            'koordinat_y': 'mean'
        }).rename(columns={'nama_halte': 'jumlah_halte'}).reset_index()

        # Count by kecamatan (district)
        halte_by_kecamatan = df.groupby(['wilayah', 'kecamatan']).agg({
            'nama_halte': 'count'
        }).rename(columns={'nama_halte': 'jumlah_halte'}).reset_index()

        self.df_halte_summary = halte_by_wilayah.sort_values('jumlah_halte', ascending=False)
        self.df_halte_by_kecamatan = halte_by_kecamatan

        print(f"  Total haltes: {len(df)}")
        print(f"  Regions: {len(halte_by_wilayah)}")

        return self.df_halte_summary

    def create_enhanced_synthetic_data(self):
        """Create enhanced synthetic data using real Transjakarta patterns."""
        if self.df_synthetic is None or self.df_bus_processed is None:
            return None

        print("\nCreating enhanced synthetic data...")

        # Get quarterly scaling factors from real data
        quarterly_scale = self.df_bus_processed[
            self.df_bus_processed['jenis_layanan'] == 'TOTAL'
        ][['year_quarter', 'passengers_per_bus']].set_index('year_quarter')

        # Map synthetic data to quarters
        df_enhanced = self.df_synthetic.copy()
        df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
        df_enhanced['year_quarter'] = (
            df_enhanced['date'].dt.year.astype(str) + '-Q' +
            ((df_enhanced['date'].dt.month - 1) // 3 + 1).astype(str)
        )

        # Calculate base traffic multipliers from real data
        avg_ppb = quarterly_scale['passengers_per_bus'].mean()
        df_enhanced['quarterly_multiplier'] = df_enhanced['year_quarter'].map(
            lambda x: quarterly_scale.loc[x, 'passengers_per_bus'] / avg_ppb if x in quarterly_scale.index else 1.0
        ).fillna(1.0)

        # Apply quarterly multiplier to traffic volume
        df_enhanced['traffic_volume_original'] = df_enhanced['traffic_volume']
        df_enhanced['traffic_volume'] = (
            df_enhanced['traffic_volume'] * df_enhanced['quarterly_multiplier']
        ).clip(0, 100)

        # Recalculate LOS
        def get_los(vol):
            if vol < 20: return 'A'
            elif vol < 40: return 'B'
            elif vol < 60: return 'C'
            elif vol < 75: return 'D'
            elif vol < 90: return 'E'
            else: return 'F'

        df_enhanced['los'] = df_enhanced['traffic_volume'].apply(get_los)

        self.df_merged = df_enhanced

        print(f"  Enhanced data: {len(df_enhanced)} rows")

        return df_enhanced

    def save_merged_data(self):
        """Save the merged dataset."""
        if self.df_merged is not None:
            self.df_merged.to_csv(OUTPUT_FILE, index=False)
            print(f"\nMerged data saved to {OUTPUT_FILE}")
            print(f"  Total rows: {len(self.df_merged)}")

    def generate_summary_report(self):
        """Generate summary statistics and visualizations."""
        print("\n" + "=" * 60)
        print("TRANSJAKARTA DATA SUMMARY REPORT")
        print("=" * 60)

        # Bus and passenger statistics
        if self.df_bus is not None:
            print("\n--- Bus & Passenger Statistics ---")
            print(f"Total data points: {len(self.df_bus)}")

            latest = self.df_bus[self.df_bus['periode_data'] == self.df_bus['periode_data'].max()]
            total_passengers = latest['jumlah_penumpang'].sum()
            total_buses = latest['jumlah_bus'].sum()

            print(f"Latest period passengers: {total_passengers:,}")
            print(f"Latest period buses: {total_buses:,}")
            print(f"Passengers per bus: {total_passengers/total_buses:.1f}")

            # Service type breakdown
            print("\nPassengers by Service Type (Latest Period):")
            service_summary = latest.groupby('jenis_layanan')['jumlah_penumpang'].sum().sort_values(ascending=False)
            for service, count in service_summary.items():
                print(f"  {service}: {count:,}")

        # Halte statistics
        if self.df_halte is not None:
            print("\n--- Halte (Bus Stop) Statistics ---")
            print(f"Total haltes: {len(self.df_halte)}")

            if hasattr(self, 'df_halte_summary'):
                print("\nHaltes by Region:")
                for _, row in self.df_halte_summary.head().iterrows():
                    print(f"  {row['wilayah']}: {row['jumlah_halte']} haltes")

        # Synthetic data statistics
        if self.df_synthetic is not None:
            print("\n--- Traffic Data Statistics ---")
            print(f"Total hourly records: {len(self.df_synthetic):,}")

            print("\nAverage traffic by hour:")
            hourly_avg = self.df_synthetic.groupby('hour')['traffic_volume'].mean()
            rush_morning = hourly_avg[7:9].mean()
            rush_evening = hourly_avg[16:19].mean()
            print(f"  Morning rush (7-9): {rush_morning:.1f}")
            print(f"  Evening rush (16-19): {rush_evening:.1f}")

            print("\nLOS Distribution:")
            los_dist = self.df_synthetic['los'].value_counts().sort_index()
            for los, count in los_dist.items():
                pct = count / len(self.df_synthetic) * 100
                print(f"  LOS {los}: {count:,} ({pct:.1f}%)")

        print("\n" + "=" * 60)

    def create_visualizations(self):
        """Create visualization plots."""
        print("\nGenerating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Quarterly Passenger Trends
        if self.df_bus is not None:
            quarterly = self.df_bus.copy()
            quarterly['date'] = pd.to_datetime(quarterly['periode_data'].astype(str), format='%Y%m', errors='coerce')
            quarterly_trend = quarterly.groupby('date')['jumlah_penumpang'].sum().sort_index()

            axes[0, 0].plot(quarterly_trend.index, quarterly_trend.values, marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_title('Quarterly Passenger Trends', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Total Passengers')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Passengers by Service Type
        if self.df_bus is not None:
            latest = self.df_bus[self.df_bus['periode_data'] == self.df_bus['periode_data'].max()]
            service_data = latest.groupby('jenis_layanan')['jumlah_penumpang'].sum().sort_values()

            axes[0, 1].barh(service_data.index, service_data.values, color='steelblue')
            axes[0, 1].set_title('Passengers by Service Type (Latest Period)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Total Passengers')

        # 3. Hourly Traffic Pattern
        if self.df_synthetic is not None:
            hourly_avg = self.df_synthetic.groupby('hour')['traffic_volume'].mean()

            axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='coral')
            axes[1, 0].axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush')
            axes[1, 0].axvspan(16, 19, alpha=0.2, color='orange', label='Evening Rush')
            axes[1, 0].set_title('Average Hourly Traffic Pattern', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Traffic Volume')
            axes[1, 0].set_xticks(range(24))
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # 4. LOS Distribution
        if self.df_synthetic is not None:
            los_counts = self.df_synthetic['los'].value_counts().sort_index()
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#8b0000']
            los_counts.plot(kind='bar', ax=axes[1, 1], color=colors)
            axes[1, 1].set_title('Level of Service (LOS) Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('LOS')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('transjakarta_analysis.png', dpi=150, bbox_inches='tight')
        print("  Visualization saved to transjakarta_analysis.png")

        # Halte map visualization
        if self.df_halte is not None:
            fig, ax = plt.subplots(figsize=(12, 10))

            wilayah_colors = {
                'Kota Adm. Jakarta Pusat': 'red',
                'Kota Adm. Jakarta Utara': 'blue',
                'Kota Adm. Jakarta Barat': 'green',
                'Kota Adm. Jakarta Selatan': 'orange',
                'Kota Adm. Jakarta Timur': 'purple',
            }

            for wilayah, color in wilayah_colors.items():
                data = self.df_halte[self.df_halte['wilayah'] == wilayah]
                if len(data) > 0:
                    ax.scatter(data['koordinat_y'].astype(float), data['koordinat_x'].astype(float),
                              c=color, label=wilayah, alpha=0.6, s=20)

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Transjakarta Halte Locations', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('halte_map.png', dpi=150, bbox_inches='tight')
            print("  Halte map saved to halte_map.png")


def main():
    """Run the integration pipeline."""
    print("=" * 60)
    print("TRANSJAKARTA DATA INTEGRATION")
    print("=" * 60)

    integrator = TransjakartaDataIntegrator()

    # Step 1: Load data
    integrator.load_data()

    # Step 2: Process bus data
    integrator.process_bus_data()

    # Step 3: Process halte data
    integrator.process_halte_data()

    # Step 4: Create enhanced synthetic data
    integrator.create_enhanced_synthetic_data()

    # Step 5: Save merged data
    integrator.save_merged_data()

    # Step 6: Generate report
    integrator.generate_summary_report()

    # Step 7: Create visualizations
    integrator.create_visualizations()

    print("\nIntegration complete!")


if __name__ == "__main__":
    main()
