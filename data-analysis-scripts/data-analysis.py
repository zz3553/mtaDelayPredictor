"""
MTA Train Delay Exploratory Data Analysis (EDA)
===============================================

This notebook will help you understand your collected data patterns and relationships
between weather conditions and train delays.

Prerequisites:
- Install required packages: pip install pandas numpy matplotlib seaborn plotly psycopg2-binary
- Set up your database credentials
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Database connection
import psycopg2
from sqlalchemy import create_engine
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 MTA Delay Analysis - EDA Notebook")
print("=" * 50)


# =============================================================================
# 1. DATABASE CONNECTION & DATA LOADING
# =============================================================================

def connect_to_database():
    """
    Connect to your PostgreSQL database
    Update these credentials to match your setup
    """
    connection_string = f"postgresql://neondb_owner:npg_VOXZBcRohC81@ep-spring-truth-ae312q45.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    engine = create_engine(connection_string)
    return engine


def load_data(engine, limit=None):
    """Load train delay and weather data"""

    # Load train delays
    delay_query ="""
        SELECT 
        trip_id, stop_id, station_name, 
        timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' as timestamp,
        delay_min, status, direction, stop_sequence, rush_hour, 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' as created_at
    FROM train_delays 
    ORDER BY timestamp DESC
    """
    if limit:
        delay_query += f" LIMIT {limit}"

    print("📡 Loading train delay data...")
    delays_df = pd.read_sql(delay_query, engine)

    # Load weather data
    weather_query = """
    SELECT 
        timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' as timestamp, location_name, temp_fahrenheit, feels_like_fahrenheit,
        weather_main, weather_description, pressure, humidity, visibility,
        wind_speed, wind_direction, rain_1h, rain_3h, snow_1h, snow_3h,
        cloudiness, is_precipitation, is_snow, is_extreme_temp, 
        is_high_humidity, is_high_wind, weather_severity_score
    FROM weather_data 
    ORDER BY timestamp DESC
    """
    if limit:
        weather_query += f" LIMIT {limit}"

    print("🌤️  Loading weather data...")
    weather_df = pd.read_sql(weather_query, engine)

    # Convert timestamp columns and handle timezone issues
    delays_df['timestamp'] = pd.to_datetime(delays_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

    # Fix timezone issue: train_delays timestamps are in UTC, need to convert to EST
    print("🕐 Converting train delays timestamps from UTC to EST...")

    # If timestamps are naive (no timezone info), assume they're UTC
    if delays_df['timestamp'].dt.tz is None:
        delays_df['timestamp'] = delays_df['timestamp'].dt.tz_localize('UTC')

    # Convert to Eastern Time
    delays_df['timestamp'] = delays_df['timestamp'].dt.tz_convert('US/Eastern')

    # Make weather timestamps timezone-aware as EST if they aren't already
    if weather_df['timestamp'].dt.tz is None:
        weather_df['timestamp'] = weather_df['timestamp'].dt.tz_localize('US/Eastern')

    # print(f"✅ Timezone conversion complete")
    # print(f"   Train delays now in: {delays_df['timestamp'].iloc[0].tz}")
    # print(f"   Weather data in: {weather_df['timestamp'].iloc[0].tz}")

    print(f"✅ Loaded {len(delays_df):,} delay records and {len(weather_df):,} weather records")
    return delays_df, weather_df


# Load the data
try:
    engine = connect_to_database()
    delays_df, weather_df = load_data(engine, limit=10000)  # Start with 10k records for speed
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("💡 Create sample data for demonstration...")

    # Create sample data if database connection fails
    np.random.seed(42)
    n_samples = 1000

    delays_df = pd.DataFrame({
        'trip_id': [f'trip_{i}' for i in range(n_samples)],
        'station_name': np.random.choice(['Times Square', 'Union Square', 'Grand Central'], n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30min'),
        'delay_min': np.random.normal(2, 5, n_samples),
        'status': np.random.choice(['on_time', 'delayed', 'early'], n_samples, p=[0.6, 0.3, 0.1]),
        'direction': np.random.choice(['N', 'S'], n_samples),
        'rush_hour': np.random.choice(['morning', 'evening', None], n_samples, p=[0.3, 0.3, 0.4])
    })

    weather_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='3h'),
        'temp_fahrenheit': np.random.normal(50, 20, 200),
        'weather_main': np.random.choice(['Clear', 'Rain', 'Snow', 'Clouds'], 200, p=[0.4, 0.2, 0.1, 0.3]),
        'humidity': np.random.uniform(30, 90, 200),
        'wind_speed': np.random.exponential(5, 200),
        'is_precipitation': np.random.choice([True, False], 200, p=[0.3, 0.7]),
        'weather_severity_score': np.random.randint(1, 11, 200)
    })

# =============================================================================
# 2. BASIC DATA EXPLORATION
# =============================================================================

print("\n📋 BASIC DATA OVERVIEW")
print("=" * 30)


def basic_data_overview(delays_df, weather_df):
    """Get basic information about the datasets"""

    print("🚇 TRAIN DELAYS DATA:")
    print(f"  • Shape: {delays_df.shape}")
    print(f"  • Date range: {delays_df['timestamp'].min()} to {delays_df['timestamp'].max()}")
    print(f"  • Unique stations: {delays_df['station_name'].nunique()}")
    print(f"  • Unique trips: {delays_df['trip_id'].nunique()}")

    print(f"\n🌤️  WEATHER DATA:")
    print(f"  • Shape: {weather_df.shape}")
    print(f"  • Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    print(f"  • Unique locations: {weather_df['location_name'].nunique()}")

    # Check for missing values
    print(f"\n❓ MISSING VALUES:")
    print("Delays:")
    missing_delays = delays_df.isnull().sum()
    for col, count in missing_delays[missing_delays > 0].items():
        print(f"  • {col}: {count} ({count / len(delays_df) * 100:.1f}%)")

    print("Weather:")
    missing_weather = weather_df.isnull().sum()
    for col, count in missing_weather[missing_weather > 0].items():
        print(f"  • {col}: {count} ({count / len(weather_df) * 100:.1f}%)")


basic_data_overview(delays_df, weather_df)

# =============================================================================
# 3. DELAY ANALYSIS
# =============================================================================

print("\n🚇 DELAY PATTERNS ANALYSIS")
print("=" * 35)


def analyze_delays(delays_df):
    """Analyze delay patterns"""

    # Basic delay statistics
    print("📊 DELAY STATISTICS:")
    print(f"  • Average delay: {delays_df['delay_min'].mean():.2f} minutes")
    print(f"  • Median delay: {delays_df['delay_min'].median():.2f} minutes")
    print(f"  • Max delay: {delays_df['delay_min'].max():.2f} minutes")
    print(f"  • Min delay: {delays_df['delay_min'].min():.2f} minutes")

    # Status distribution
    print(f"\n📈 STATUS DISTRIBUTION:")
    status_counts = delays_df['status'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(delays_df) * 100
        print(f"  • {status}: {count:,} ({pct:.1f}%)")

    # Rush hour analysis
    if 'rush_hour' in delays_df.columns:
        print(f"\n⏰ RUSH HOUR ANALYSIS:")
        rush_analysis = delays_df.groupby('rush_hour')['delay_min'].agg(['count', 'mean', 'std']).round(2)
        print(rush_analysis)


analyze_delays(delays_df)


# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_delay_distribution(delays_df):
    """Plot delay distribution"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🚇 Train Delay Distribution Analysis', fontsize=16, fontweight='bold')

    # Histogram of delays
    ax1.hist(delays_df['delay_min'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Delays')
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(delays_df['delay_min'].mean(), color='red', linestyle='--',
                label=f'Mean: {delays_df["delay_min"].mean():.1f}min')
    ax1.legend()

    # Box plot by status
    delays_df.boxplot(column='delay_min', by='status', ax=ax2)
    ax2.set_title('Delays by Status')
    ax2.set_xlabel('Status')
    ax2.set_ylabel('Delay (minutes)')

    # Status distribution pie chart
    status_counts = delays_df['status'].value_counts()
    ax3.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Status Distribution')

    # Delays by direction
    if 'direction' in delays_df.columns:
        direction_delays = delays_df.groupby('direction')['delay_min'].mean()
        ax4.bar(direction_delays.index, direction_delays.values, color=['lightcoral', 'lightblue'])
        ax4.set_title('Average Delay by Direction')
        ax4.set_xlabel('Direction')
        ax4.set_ylabel('Average Delay (minutes)')

    plt.tight_layout()
    return fig


def plot_time_series_analysis(delays_df):
    """Plot time series analysis"""
    # Create hourly aggregations
    delays_df['hour'] = delays_df['timestamp'].dt.hour
    delays_df['day_of_week'] = delays_df['timestamp'].dt.day_name()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('⏰ Time-Based Delay Analysis', fontsize=16, fontweight='bold')

    # Delays by hour of day
    hourly_delays = delays_df.groupby('hour')['delay_min'].mean()
    ax1.plot(hourly_delays.index, hourly_delays.values, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Average Delay by Hour of Day')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Average Delay (minutes)')
    ax1.grid(True, alpha=0.3)

    # Delays by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_delays = delays_df.groupby('day_of_week')['delay_min'].mean().reindex(day_order)
    ax2.bar(range(len(daily_delays)), daily_delays.values, color='lightgreen')
    ax2.set_title('Average Delay by Day of Week')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Delay (minutes)')
    ax2.set_xticks(range(len(daily_delays)))
    ax2.set_xticklabels(daily_delays.index, rotation=45)

    # Rush hour comparison
    if 'rush_hour' in delays_df.columns:
        rush_delays = delays_df.groupby('rush_hour')['delay_min'].mean()
        ax3.bar(rush_delays.index, rush_delays.values, color='orange', alpha=0.7)
        ax3.set_title('Average Delay by Rush Hour Period')
        ax3.set_xlabel('Rush Hour Period')
        ax3.set_ylabel('Average Delay (minutes)')

    # Daily delay trend over time
    daily_trend = delays_df.groupby(delays_df['timestamp'].dt.date)['delay_min'].mean()
    ax4.plot(daily_trend.index, daily_trend.values, alpha=0.7)
    ax4.set_title('Daily Average Delay Trend')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Average Delay (minutes)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def plot_weather_analysis(weather_df):
    """Plot weather data analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🌤️ Weather Data Analysis', fontsize=16, fontweight='bold')

    # Temperature distribution
    ax1.hist(weather_df['temp_fahrenheit'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax1.set_title('Temperature Distribution')
    ax1.set_xlabel('Temperature (°F)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(weather_df['temp_fahrenheit'].mean(), color='red', linestyle='--',
                label=f'Mean: {weather_df["temp_fahrenheit"].mean():.1f}°F')
    ax1.legend()

    # Weather conditions
    if 'weather_main' in weather_df.columns:
        weather_counts = weather_df['weather_main'].value_counts()
        ax2.pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
        ax2.set_title('Weather Conditions Distribution')

    # Weather severity score
    if 'weather_severity_score' in weather_df.columns:
        ax3.hist(weather_df['weather_severity_score'], bins=10, alpha=0.7, color='red', edgecolor='black')
        ax3.set_title('Weather Severity Score Distribution')
        ax3.set_xlabel('Severity Score (1-10)')
        ax3.set_ylabel('Frequency')

    # Temperature vs Humidity scatter
    if 'humidity' in weather_df.columns:
        ax4.scatter(weather_df['temp_fahrenheit'], weather_df['humidity'], alpha=0.6, color='blue')
        ax4.set_title('Temperature vs Humidity')
        ax4.set_xlabel('Temperature (°F)')
        ax4.set_ylabel('Humidity (%)')

    plt.tight_layout()
    return fig


# =============================================================================
# 5. WEATHER-DELAY CORRELATION ANALYSIS
# =============================================================================

def merge_weather_delays(delays_df, weather_df, time_window='1H'):
    """
    Merge weather and delay data by timestamp
    time_window: pandas frequency string for time matching tolerance
    """
    print(f"\n🔗 MERGING WEATHER AND DELAY DATA")
    # print(f"Time window: {time_window}")

    # Round timestamps to nearest hour for better matching
    delays_df = delays_df.copy()
    weather_df = weather_df.copy()

    delays_df['timestamp_rounded'] = delays_df['timestamp'].dt.round(time_window)
    weather_df['timestamp_rounded'] = weather_df['timestamp'].dt.round(time_window)

    # Merge on rounded timestamp
    merged_df = pd.merge(
        delays_df,
        weather_df,
        on='timestamp_rounded',
        how='inner',
        suffixes=('_delay', '_weather')
    )

    print(f"✅ Merged dataset shape: {merged_df.shape}")
    print(f"📊 Coverage: {len(merged_df) / len(delays_df) * 100:.1f}% of delays have weather data")

    return merged_df


def analyze_weather_delay_correlation(merged_df):
    """Analyze correlation between weather and delays"""
    print(f"\n🌩️ WEATHER-DELAY CORRELATION ANALYSIS")
    print("=" * 45)

    # Correlation with numerical weather features
    weather_features = ['temp_fahrenheit', 'humidity', 'wind_speed', 'weather_severity_score']
    available_features = [f for f in weather_features if f in merged_df.columns]

    if available_features:
        correlations = merged_df[available_features + ['delay_min']].corr()['delay_min'].sort_values(key=abs,
                                                                                                     ascending=False)

        print("📊 CORRELATION WITH DELAY:")
        for feature, corr in correlations.items():
            if feature != 'delay_min':
                strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
                print(f"  • {feature}: {corr:.3f} ({strength})")

    # Delay by weather condition
    if 'weather_main' in merged_df.columns:
        print(f"\n🌤️ DELAY BY WEATHER CONDITION:")
        weather_delays = merged_df.groupby('weather_main')['delay_min'].agg(['count', 'mean', 'std']).round(2)
        print(weather_delays)

    # Delay by precipitation
    if 'is_precipitation' in merged_df.columns:
        print(f"\n🌧️ DELAY BY PRECIPITATION:")
        precip_delays = merged_df.groupby('is_precipitation')['delay_min'].agg(['count', 'mean', 'std']).round(2)
        print(precip_delays)


def plot_weather_delay_relationships(merged_df):
    """Plot weather-delay relationships"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature vs Delay', 'Weather Severity vs Delay',
                        'Delay by Weather Condition', 'Delay by Precipitation'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # Temperature vs Delay
    if 'temp_fahrenheit' in merged_df.columns:
        fig.add_trace(
            go.Scatter(x=merged_df['temp_fahrenheit'], y=merged_df['delay_min'],
                       mode='markers', opacity=0.6, name='Temp vs Delay'),
            row=1, col=1
        )

    # Weather Severity vs Delay
    if 'weather_severity_score' in merged_df.columns:
        fig.add_trace(
            go.Scatter(x=merged_df['weather_severity_score'], y=merged_df['delay_min'],
                       mode='markers', opacity=0.6, name='Severity vs Delay'),
            row=1, col=2
        )

    # Delay by Weather Condition
    if 'weather_main' in merged_df.columns:
        weather_delays = merged_df.groupby('weather_main')['delay_min'].mean()
        fig.add_trace(
            go.Bar(x=weather_delays.index, y=weather_delays.values, name='Weather Delays'),
            row=2, col=1
        )

    # Delay by Precipitation
    if 'is_precipitation' in merged_df.columns:
        precip_delays = merged_df.groupby('is_precipitation')['delay_min'].mean()
        fig.add_trace(
            go.Bar(x=precip_delays.index, y=precip_delays.values, name='Precipitation Delays'),
            row=2, col=2
        )

    fig.update_layout(height=800, title_text="🌩️ Weather-Delay Relationships")
    return fig


# =============================================================================
# 6. RUN THE ANALYSIS
# =============================================================================

print("\n🚀 RUNNING EXPLORATORY DATA ANALYSIS")
print("=" * 40)

# Generate all visualizations
try:
    # Basic delay analysis plots
    delay_dist_fig = plot_delay_distribution(delays_df)
    plt.show()

    # Time series analysis
    time_series_fig = plot_time_series_analysis(delays_df)
    plt.show()

    # Weather analysis
    weather_fig = plot_weather_analysis(weather_df)
    plt.show()

    # Weather-delay correlation (if both datasets exist)
    if len(delays_df) > 0 and len(weather_df) > 0:
        merged_df = merge_weather_delays(delays_df, weather_df)

        if len(merged_df) > 0:
            analyze_weather_delay_correlation(merged_df)

            # Interactive plot
            weather_delay_fig = plot_weather_delay_relationships(merged_df)
            weather_delay_fig.show()
        else:
            print("⚠️ No overlapping timestamps between weather and delay data")

    print("\n✅ EDA COMPLETE!")
    print("\n📝 KEY INSIGHTS TO DOCUMENT:")
    print("1. What are the peak delay hours?")
    print("2. Which weather conditions correlate most with delays?")
    print("3. Are there seasonal patterns?")
    print("4. Which stations/directions are most affected?")
    print("5. What's the typical delay distribution?")

except Exception as e:
    print(f"❌ Error during analysis: {e}")
    print("💡 Try running sections individually to debug")


# =============================================================================
# 7. EXPORT RESULTS
# =============================================================================

def export_summary_stats(delays_df, merged_df=None):
    """Export key statistics to CSV for further analysis"""

    # Delay summary stats
    delay_summary = {
        'metric': ['count', 'mean_delay', 'median_delay', 'std_delay', 'max_delay', 'min_delay'],
        'value': [
            len(delays_df),
            delays_df['delay_min'].mean(),
            delays_df['delay_min'].median(),
            delays_df['delay_min'].std(),
            delays_df['delay_min'].max(),
            delays_df['delay_min'].min()
        ]
    }

    delay_summary_df = pd.DataFrame(delay_summary)
    delay_summary_df.to_csv('delay_summary_stats.csv', index=False)

    # Status distribution
    status_dist = delays_df['status'].value_counts().to_frame('count')
    status_dist.to_csv('delay_status_distribution.csv')

    print("📁 Exported summary files:")
    print("  • delay_summary_stats.csv")
    print("  • delay_status_distribution.csv")

    if merged_df is not None and len(merged_df) > 0:
        # Weather correlation summary
        weather_features = ['temp_fahrenheit', 'humidity', 'wind_speed', 'weather_severity_score']
        available_features = [f for f in weather_features if f in merged_df.columns]

        if available_features:
            correlations = merged_df[available_features + ['delay_min']].corr()['delay_min']
            correlations.to_csv('weather_delay_correlations.csv', header=['correlation'])
            print("  • weather_delay_correlations.csv")


# Export summary statistics
try:
    if 'merged_df' in locals():
        export_summary_stats(delays_df, weather_df, merged_df)
    else:
        export_summary_stats(delays_df, weather_df)
except Exception as e:
    print(f"⚠️ Could not export summary: {e}")

print("\n🎉 EDA NOTEBOOK COMPLETE!")