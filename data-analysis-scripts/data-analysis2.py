import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')

# Database connection - update with your credentials
DATABASE_URL = "postgresql://neondb_owner:npg_VOXZBcRohC81@ep-spring-truth-ae312q45.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
engine = create_engine(DATABASE_URL)


class WeatherDelayExplorer:
    def __init__(self, engine):
        self.engine = engine
        self.train_delays = None
        self.weather_data = None
        self.merged_data = None

    def load_data(self):
        """Load and basic preprocessing of data"""
        print("Loading train delays data...")
        self.train_delays = pd.read_sql("""
            SELECT * FROM train_delays 
            WHERE delay_min IS NOT NULL
            ORDER BY timestamp
        """, self.engine)

        print("Loading weather data...")
        self.weather_data = pd.read_sql("""
            SELECT * FROM weather_data 
            ORDER BY timestamp
        """, self.engine)

        # Convert timestamps and handle timezone conversion
        # Train delays are in UTC, convert to EST to match weather data
        self.train_delays['timestamp'] = pd.to_datetime(self.train_delays['timestamp'])
        self.train_delays['created_at'] = pd.to_datetime(self.train_delays['created_at'])

        # Convert UTC to EST (UTC-5, or UTC-4 during DST)
        self.train_delays['timestamp'] = self.train_delays['timestamp'].dt.tz_localize('UTC').dt.tz_convert(
            'US/Eastern')
        self.train_delays['created_at'] = self.train_delays['created_at'].dt.tz_localize('UTC').dt.tz_convert(
            'US/Eastern')

        # Weather data is already in EST
        self.weather_data['timestamp'] = pd.to_datetime(self.weather_data['timestamp'])

        # Remove timezone info for merging (both now in EST)
        self.train_delays['timestamp'] = self.train_delays['timestamp'].dt.tz_localize(None)
        self.train_delays['created_at'] = self.train_delays['created_at'].dt.tz_localize(None)

        print(f"Loaded {len(self.train_delays)} delay records and {len(self.weather_data)} weather records")
        print(f"Converted train_delays timestamps from UTC to EST to match weather data")

    def analyze_target_distribution(self):
        """Analyze delay_min distribution and characteristics"""
        print("\n" + "=" * 50)
        print("TARGET VARIABLE ANALYSIS: delay_min")
        print("=" * 50)

        delays = self.train_delays['delay_min']

        # Basic statistics
        print(f"Total delay records: {len(delays)}")
        print(f"Mean delay: {delays.mean():.2f} minutes")
        print(f"Median delay: {delays.median():.2f} minutes")
        print(f"Standard deviation: {delays.std():.2f} minutes")
        print(f"Min delay: {delays.min():.2f} minutes")
        print(f"Max delay: {delays.max():.2f} minutes")

        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        print(f"\nDelay Percentiles:")
        for p in percentiles:
            value = np.percentile(delays, p)
            print(f"  {p}th percentile: {value:.2f} minutes")

        # Zero delays analysis
        zero_delays = (delays == 0).sum()
        zero_pct = (zero_delays / len(delays)) * 100
        print(f"\nOn-time trains (0 delay): {zero_delays} ({zero_pct:.1f}%)")

        # Delay categories
        delayed_trains = delays > 0
        minor_delays = (delays > 0) & (delays <= 5)
        moderate_delays = (delays > 5) & (delays <= 15)
        major_delays = delays > 15

        print(f"Delayed trains (>0 min): {delayed_trains.sum()} ({(delayed_trains.sum() / len(delays) * 100):.1f}%)")
        print(f"Minor delays (0-5 min): {minor_delays.sum()} ({(minor_delays.sum() / len(delays) * 100):.1f}%)")
        print(
            f"Moderate delays (5-15 min): {moderate_delays.sum()} ({(moderate_delays.sum() / len(delays) * 100):.1f}%)")
        print(f"Major delays (>15 min): {major_delays.sum()} ({(major_delays.sum() / len(delays) * 100):.1f}%)")

        # Create delay distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Histogram of all delays
        axes[0, 0].hist(delays, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution of All Delays')
        axes[0, 0].set_xlabel('Delay (minutes)')
        axes[0, 0].set_ylabel('Frequency')

        # Histogram excluding zero delays
        non_zero_delays = delays[delays > 0]
        axes[0, 1].hist(non_zero_delays, bins=30, alpha=0.7, color='red')
        axes[0, 1].set_title('Distribution of Non-Zero Delays')
        axes[0, 1].set_xlabel('Delay (minutes)')
        axes[0, 1].set_ylabel('Frequency')

        # Box plot
        axes[1, 0].boxplot(delays)
        axes[1, 0].set_title('Delay Distribution (Box Plot)')
        axes[1, 0].set_ylabel('Delay (minutes)')

        # Delays by rush hour
        if 'rush_hour' in self.train_delays.columns:
            delay_by_rush = self.train_delays.groupby('rush_hour')['delay_min'].mean()
            axes[1, 1].bar(delay_by_rush.index, delay_by_rush.values)
            axes[1, 1].set_title('Average Delay by Rush Hour Period')
            axes[1, 1].set_xlabel('Rush Hour Period')
            axes[1, 1].set_ylabel('Average Delay (minutes)')

        plt.tight_layout()
        plt.savefig('delay_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def merge_weather_delays(self):
        """Merge weather and delay data by timestamp"""
        print("\n" + "=" * 50)
        print("MERGING WEATHER AND DELAY DATA")
        print("=" * 50)

        # Round timestamps to nearest 15 minutes for matching
        self.train_delays['timestamp_rounded'] = self.train_delays['timestamp'].dt.round('15min')
        self.weather_data['timestamp_rounded'] = self.weather_data['timestamp'].dt.round('15min')

        # Merge on rounded timestamp
        self.merged_data = pd.merge(
            self.train_delays,
            self.weather_data,
            on='timestamp_rounded',
            how='inner',
            suffixes=('_delay', '_weather')
        )

        print(f"Successfully merged {len(self.merged_data)} records")
        print(f"Date range: {self.merged_data['timestamp_delay'].min()} to {self.merged_data['timestamp_delay'].max()}")

        # Check for missing values in key weather columns
        weather_cols = ['temp_fahrenheit', 'humidity', 'pressure', 'wind_speed',
                        'rain_1h', 'snow_1h', 'visibility']

        print(f"\nMissing values in weather features:")
        for col in weather_cols:
            if col in self.merged_data.columns:
                missing = self.merged_data[col].isnull().sum()
                missing_pct = (missing / len(self.merged_data)) * 100
                print(f"  {col}: {missing} ({missing_pct:.1f}%)")

    def calculate_correlations(self):
        """Calculate correlations between weather features and delays"""
        print("\n" + "=" * 50)
        print("WEATHER-DELAY CORRELATIONS")
        print("=" * 50)

        # Define weather features to analyze
        weather_features = [
            'temp_fahrenheit', 'feels_like_fahrenheit', 'temp_min_fahrenheit', 'temp_max_fahrenheit',
            'pressure', 'humidity', 'sea_level_pressure', 'ground_level_pressure',
            'visibility', 'wind_speed', 'wind_direction', 'wind_gust',
            'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h', 'cloudiness',
            'weather_severity_score'
        ]

        # Filter to available columns
        available_weather_features = [col for col in weather_features if col in self.merged_data.columns]

        # Calculate Pearson correlations
        correlations = {}
        significant_correlations = {}

        for feature in available_weather_features:
            # Remove missing values for correlation calculation
            valid_data = self.merged_data[[feature, 'delay_min']].dropna()

            if len(valid_data) > 10:  # Need sufficient data points
                corr_coef, p_value = stats.pearsonr(valid_data[feature], valid_data['delay_min'])
                correlations[feature] = corr_coef

                if p_value < 0.05:  # Significant correlation
                    significant_correlations[feature] = (corr_coef, p_value)

        # Sort correlations by absolute value
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        print("Correlations with delay_min (sorted by absolute value):")
        print("-" * 60)
        for feature, corr in sorted_correlations:
            significance = " *" if feature in significant_correlations else ""
            print(f"{feature:30s}: {corr:8.4f}{significance}")

        print(f"\n* indicates statistically significant (p < 0.05)")
        print(f"Significant correlations found: {len(significant_correlations)}")

        # Create correlation heatmap
        if len(available_weather_features) > 0:
            # Create correlation matrix
            corr_data = self.merged_data[available_weather_features + ['delay_min']].corr()

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                        fmt='.3f', square=True, linewidths=0.5)
            plt.title('Weather Features Correlation Matrix')
            plt.tight_layout()
            plt.savefig('weather_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

        return correlations, significant_correlations

    def analyze_categorical_weather_impact(self):
        """Analyze delay patterns by categorical weather conditions"""
        print("\n" + "=" * 50)
        print("CATEGORICAL WEATHER IMPACT ANALYSIS")
        print("=" * 50)

        categorical_features = ['weather_main', 'weather_description', 'is_precipitation',
                                'is_snow', 'is_extreme_temp', 'is_high_humidity', 'is_high_wind']

        for feature in categorical_features:
            if feature in self.merged_data.columns:
                print(f"\n{feature.upper()}:")
                delay_by_category = self.merged_data.groupby(feature)['delay_min'].agg(['count', 'mean', 'std'])
                print(delay_by_category.round(2))

    def determine_prediction_timeframes(self):
        """Analyze data frequency and suggest prediction timeframes"""
        print("\n" + "=" * 50)
        print("PREDICTION TIMEFRAME ANALYSIS")
        print("=" * 50)

        # Analyze data collection frequency
        time_diffs = self.merged_data['timestamp_delay'].diff().dropna()

        print(f"Data collection patterns:")
        print(f"  Most common interval: {time_diffs.mode().iloc[0]}")
        print(f"  Average interval: {time_diffs.mean()}")
        print(f"  Min interval: {time_diffs.min()}")
        print(f"  Max interval: {time_diffs.max()}")

        # Count observations by time interval
        interval_counts = time_diffs.value_counts().head(10)
        print(f"\nTop intervals in dataset:")
        for interval, count in interval_counts.items():
            print(f"  {interval}: {count} occurrences")

        # Suggest prediction timeframes
        print(f"\nSUGGESTED PREDICTION TIMEFRAMES:")
        print(f"1. Short-term (15-30 min ahead): Real-time operational decisions")
        print(f"2. Medium-term (1-2 hours ahead): Service planning and passenger alerts")
        print(f"3. Long-term (4+ hours ahead): Strategic service adjustments")

    def generate_summary_report(self, correlations, significant_correlations):
        """Generate a summary report of findings"""
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)

        print(f"Dataset Overview:")
        print(f"  Total merged records: {len(self.merged_data)}")
        print(f"  Average delay: {self.merged_data['delay_min'].mean():.2f} minutes")
        print(
            f"  Delayed trains: {(self.merged_data['delay_min'] > 0).sum()} ({(self.merged_data['delay_min'] > 0).mean() * 100:.1f}%)")

        print(f"\nTop Weather Correlations:")
        top_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, corr in top_correlations:
            print(f"  {feature}: {corr:.4f}")

        print(f"\nStatistically Significant Features: {len(significant_correlations)}")
        for feature, (corr, p_val) in significant_correlations.items():
            print(f"  {feature}: r={corr:.4f}, p={p_val:.4f}")

        print(f"\nNext Steps Recommendations:")
        print(f"1. Focus feature engineering on top correlated weather variables")
        print(f"2. Create lagged features for weather conditions (15-60 min)")
        print(f"3. Engineer threshold-based features for significant weather variables")
        print(f"4. Consider station-specific weather impacts")


def main():
    """Main execution function"""
    # Initialize explorer
    explorer = WeatherDelayExplorer(engine)

    # Run analysis pipeline
    explorer.load_data()
    explorer.analyze_target_distribution()
    explorer.merge_weather_delays()
    correlations, significant_correlations = explorer.calculate_correlations()
    explorer.analyze_categorical_weather_impact()
    explorer.determine_prediction_timeframes()
    explorer.generate_summary_report(correlations, significant_correlations)

    print(f"\nAnalysis complete! Check generated plots:")
    print(f"  - delay_distribution_analysis.png")
    print(f"  - weather_correlation_heatmap.png")


if __name__ == "__main__":
    # Update DATABASE_URL with your actual connection string
    main()