import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')


class WeatherFeatureEngineer:
    def __init__(self, engine, top_weather_features=None):
        """
        Initialize with database connection and top correlated features

        Args:
            engine: SQLAlchemy database engine
            top_weather_features: List of top correlated weather features from exploration
        """
        self.engine = engine
        self.merged_data = None
        self.feature_data = None

        # Default top features (update based on your correlation analysis results)
        self.top_weather_features = top_weather_features or [
            'temp_fahrenheit', 'feels_like_fahrenheit', 'humidity', 'pressure',
            'wind_speed', 'visibility', 'rain_1h', 'snow_1h', 'weather_severity_score'
        ]

        self.engineered_features = []
        self.feature_importance_scores = {}

    def load_and_merge_data(self):
        """Load and merge weather and delay data with timezone handling"""
        print("Loading and merging data...")

        # Load train delays
        train_delays = pd.read_sql("""
            SELECT * FROM train_delays 
            WHERE delay_min IS NOT NULL
            ORDER BY timestamp
        """, self.engine)

        # Load weather data
        weather_data = pd.read_sql("""
            SELECT * FROM weather_data 
            ORDER BY timestamp
        """, self.engine)

        # Handle timezone conversion (UTC to EST for train_delays)
        train_delays['timestamp'] = pd.to_datetime(train_delays['timestamp'])
        train_delays['timestamp'] = train_delays['timestamp'].dt.tz_localize('UTC').dt.tz_convert(
            'US/Eastern').dt.tz_localize(None)

        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

        # Round timestamps to nearest 15 minutes for matching
        train_delays['timestamp_rounded'] = train_delays['timestamp'].dt.round('15min')
        weather_data['timestamp_rounded'] = weather_data['timestamp'].dt.round('15min')

        # Merge datasets
        self.merged_data = pd.merge(
            train_delays, weather_data,
            on='timestamp_rounded',
            how='inner',
            suffixes=('_delay', '_weather')
        ).sort_values('timestamp_delay')

        print(f"Merged dataset: {len(self.merged_data)} records")
        return self.merged_data

    def create_temporal_features(self):
        """Create time-based features"""
        print("Creating temporal features...")

        df = self.merged_data.copy()

        # Basic time features
        df['hour'] = df['timestamp_delay'].dt.hour
        df['day_of_week'] = df['timestamp_delay'].dt.dayofweek  # 0=Monday
        df['month'] = df['timestamp_delay'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Rush hour indicators
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

        # Time since rush hour start
        df['minutes_since_rush_start'] = 0
        morning_rush_mask = df['is_morning_rush'] == 1
        evening_rush_mask = df['is_evening_rush'] == 1

        df.loc[morning_rush_mask, 'minutes_since_rush_start'] = (df.loc[morning_rush_mask, 'hour'] - 7) * 60 + df.loc[
            morning_rush_mask, 'timestamp_delay'].dt.minute
        df.loc[evening_rush_mask, 'minutes_since_rush_start'] = (df.loc[evening_rush_mask, 'hour'] - 17) * 60 + df.loc[
            evening_rush_mask, 'timestamp_delay'].dt.minute

        temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend',
                             'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
                             'minutes_since_rush_start']

        self.engineered_features.extend(temporal_features)
        return df

    def create_weather_threshold_features(self, df):
        """Create threshold-based weather features"""
        print("Creating weather threshold features...")

        # Temperature thresholds
        if 'temp_fahrenheit' in self.top_weather_features:
            df['temp_freezing'] = (df['temp_fahrenheit'] <= 32).astype(int)
            df['temp_very_cold'] = (df['temp_fahrenheit'] <= 20).astype(int)
            df['temp_hot'] = (df['temp_fahrenheit'] >= 85).astype(int)
            df['temp_very_hot'] = (df['temp_fahrenheit'] >= 95).astype(int)

            # Temperature bins
            df['temp_bin'] = pd.cut(df['temp_fahrenheit'],
                                    bins=[-np.inf, 20, 32, 50, 70, 85, np.inf],
                                    labels=['very_cold', 'cold', 'cool', 'mild', 'warm', 'hot'])

            # One-hot encode temperature bins
            temp_dummies = pd.get_dummies(df['temp_bin'], prefix='temp')
            df = pd.concat([df, temp_dummies], axis=1)

            threshold_features = ['temp_freezing', 'temp_very_cold', 'temp_hot', 'temp_very_hot']
            threshold_features.extend(temp_dummies.columns.tolist())

        # Humidity thresholds
        if 'humidity' in self.top_weather_features:
            df['humidity_high'] = (df['humidity'] >= 80).astype(int)
            df['humidity_very_high'] = (df['humidity'] >= 90).astype(int)
            df['humidity_low'] = (df['humidity'] <= 30).astype(int)
            threshold_features.extend(['humidity_high', 'humidity_very_high', 'humidity_low'])

        # Wind thresholds
        if 'wind_speed' in self.top_weather_features:
            df['wind_high'] = (df['wind_speed'] >= 15).astype(int)  # mph
            df['wind_very_high'] = (df['wind_speed'] >= 25).astype(int)
            threshold_features.extend(['wind_high', 'wind_very_high'])

        # Visibility thresholds
        if 'visibility' in self.top_weather_features:
            df['visibility_poor'] = (df['visibility'] <= 5000).astype(int)  # meters
            df['visibility_very_poor'] = (df['visibility'] <= 1000).astype(int)
            threshold_features.extend(['visibility_poor', 'visibility_very_poor'])

        # Precipitation thresholds
        if 'rain_1h' in self.top_weather_features:
            df['rain_light'] = ((df['rain_1h'] > 0) & (df['rain_1h'] <= 2.5)).astype(int)
            df['rain_moderate'] = ((df['rain_1h'] > 2.5) & (df['rain_1h'] <= 7.5)).astype(int)
            df['rain_heavy'] = (df['rain_1h'] > 7.5).astype(int)
            threshold_features.extend(['rain_light', 'rain_moderate', 'rain_heavy'])

        if 'snow_1h' in self.top_weather_features:
            df['snow_light'] = ((df['snow_1h'] > 0) & (df['snow_1h'] <= 1)).astype(int)
            df['snow_moderate'] = ((df['snow_1h'] > 1) & (df['snow_1h'] <= 4)).astype(int)
            df['snow_heavy'] = (df['snow_1h'] > 4).astype(int)
            threshold_features.extend(['snow_light', 'snow_moderate', 'snow_heavy'])

        self.engineered_features.extend([f for f in threshold_features if f in df.columns])
        return df

    def create_weather_interaction_features(self, df):
        """Create interaction features between top weather variables"""
        print("Creating weather interaction features...")

        interaction_features = []

        # Temperature + Precipitation interactions
        if 'temp_fahrenheit' in df.columns and 'rain_1h' in df.columns:
            df['temp_rain_interaction'] = df['temp_fahrenheit'] * df['rain_1h'].fillna(0)
            interaction_features.append('temp_rain_interaction')

        if 'temp_fahrenheit' in df.columns and 'snow_1h' in df.columns:
            df['temp_snow_interaction'] = df['temp_fahrenheit'] * df['snow_1h'].fillna(0)
            interaction_features.append('temp_snow_interaction')

        # Wind + Precipitation interactions
        if 'wind_speed' in df.columns and 'rain_1h' in df.columns:
            df['wind_rain_interaction'] = df['wind_speed'] * df['rain_1h'].fillna(0)
            interaction_features.append('wind_rain_interaction')

        # Temperature + Humidity interactions
        if 'temp_fahrenheit' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temp_fahrenheit'] * df['humidity']
            interaction_features.append('temp_humidity_interaction')

        # Rush hour + Weather interactions
        if 'is_rush_hour' in df.columns:
            for weather_feature in ['temp_fahrenheit', 'rain_1h', 'snow_1h', 'wind_speed']:
                if weather_feature in df.columns:
                    interaction_name = f'rush_{weather_feature}_interaction'
                    df[interaction_name] = df['is_rush_hour'] * df[weather_feature].fillna(0)
                    interaction_features.append(interaction_name)

        self.engineered_features.extend(interaction_features)
        return df

    def create_lag_features(self, df):
        """Create lagged weather features (past conditions)"""
        print("Creating lagged weather features...")

        # Sort by station and timestamp for proper lagging
        df = df.sort_values(['station_name', 'timestamp_delay'])

        lag_features = []
        lag_periods = [1, 2, 4]  # 15min, 30min, 1hr ago

        for feature in self.top_weather_features:
            if feature in df.columns:
                for lag in lag_periods:
                    lag_feature_name = f'{feature}_lag_{lag * 15}min'
                    df[lag_feature_name] = df.groupby('station_name')[feature].shift(lag)
                    lag_features.append(lag_feature_name)

                    # Weather change features
                    change_feature_name = f'{feature}_change_{lag * 15}min'
                    df[change_feature_name] = df[feature] - df[lag_feature_name]
                    lag_features.append(change_feature_name)

        self.engineered_features.extend(lag_features)
        return df

    def create_rolling_features(self, df):
        """Create rolling window features"""
        print("Creating rolling window features...")

        # Sort by station and timestamp
        df = df.sort_values(['station_name', 'timestamp_delay'])

        rolling_features = []
        windows = [4, 8]  # 1hr, 2hr windows (15min intervals)

        for feature in self.top_weather_features:
            if feature in df.columns:
                for window in windows:
                    # Rolling mean
                    mean_feature_name = f'{feature}_rolling_mean_{window * 15}min'
                    df[mean_feature_name] = df.groupby('station_name')[feature].rolling(window,
                                                                                        min_periods=1).mean().reset_index(
                        0, drop=True)
                    rolling_features.append(mean_feature_name)

                    # Rolling std
                    std_feature_name = f'{feature}_rolling_std_{window * 15}min'
                    df[std_feature_name] = df.groupby('station_name')[feature].rolling(window,
                                                                                       min_periods=1).std().reset_index(
                        0, drop=True)
                    rolling_features.append(std_feature_name)

                    # Rolling min/max
                    min_feature_name = f'{feature}_rolling_min_{window * 15}min'
                    max_feature_name = f'{feature}_rolling_max_{window * 15}min'
                    df[min_feature_name] = df.groupby('station_name')[feature].rolling(window,
                                                                                       min_periods=1).min().reset_index(
                        0, drop=True)
                    df[max_feature_name] = df.groupby('station_name')[feature].rolling(window,
                                                                                       min_periods=1).max().reset_index(
                        0, drop=True)
                    rolling_features.extend([min_feature_name, max_feature_name])

        self.engineered_features.extend(rolling_features)
        return df

    def create_composite_weather_scores(self, df):
        """Create composite weather severity scores"""
        print("Creating composite weather scores...")

        composite_features = []

        # Custom weather severity score
        weather_severity = 0

        # Temperature component
        if 'temp_fahrenheit' in df.columns:
            # Extreme temperatures increase severity
            temp_severity = np.where(df['temp_fahrenheit'] <= 32, (32 - df['temp_fahrenheit']) / 10, 0)
            temp_severity += np.where(df['temp_fahrenheit'] >= 90, (df['temp_fahrenheit'] - 90) / 10, 0)
            weather_severity += temp_severity

        # Precipitation component
        if 'rain_1h' in df.columns:
            rain_severity = df['rain_1h'].fillna(0) * 2  # Rain has moderate impact
            weather_severity += rain_severity

        if 'snow_1h' in df.columns:
            snow_severity = df['snow_1h'].fillna(0) * 3  # Snow has higher impact
            weather_severity += snow_severity

        # Wind component
        if 'wind_speed' in df.columns:
            wind_severity = np.where(df['wind_speed'] > 15, (df['wind_speed'] - 15) / 5, 0)
            weather_severity += wind_severity

        # Visibility component
        if 'visibility' in df.columns:
            visibility_severity = np.where(df['visibility'] < 5000, (5000 - df['visibility']) / 1000, 0)
            weather_severity += visibility_severity

        df['custom_weather_severity'] = weather_severity
        composite_features.append('custom_weather_severity')

        # Comfort index (temperature + humidity interaction)
        if 'temp_fahrenheit' in df.columns and 'humidity' in df.columns:
            # Simple heat index approximation
            df['comfort_index'] = df['temp_fahrenheit'] + (0.5 * df['humidity'])
            composite_features.append('comfort_index')

        self.engineered_features.extend(composite_features)
        return df

    def run_feature_engineering_pipeline(self):
        """Run the complete feature engineering pipeline"""
        print("=" * 60)
        print("WEATHER FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        # Load data
        df = self.load_and_merge_data()

        # Create all feature types
        df = self.create_temporal_features()
        df = self.create_weather_threshold_features(df)
        df = self.create_weather_interaction_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_composite_weather_scores(df)

        self.feature_data = df

        print(f"\nFeature engineering complete!")
        print(f"Total engineered features: {len(self.engineered_features)}")
        print(f"Dataset shape: {df.shape}")

        return df

    def evaluate_feature_importance(self):
        """Evaluate feature importance using Random Forest"""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE EVALUATION")
        print("=" * 60)

        # Prepare features for modeling
        feature_columns = self.top_weather_features + self.engineered_features

        # Filter to available columns and remove non-numeric
        available_features = []
        for col in feature_columns:
            if col in self.feature_data.columns:
                try:
                    # Get the dtype using pandas select_dtypes method
                    col_dtype = str(self.feature_data[col].dtype)
                    if any(dtype in col_dtype for dtype in ['int', 'float', 'bool']):
                        available_features.append(col)
                except (AttributeError, KeyError):
                    # Skip columns that cause issues
                    print(f"Warning: Skipping column {col} due to dtype issues")
                    continue

        X = self.feature_data[available_features].fillna(0)
        y = self.feature_data['delay_min']

        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)

        print(f"Features for importance evaluation: {len(available_features)}")
        print(f"Samples: {len(X)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate model performance
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f} minutes")

        print(f"\nTop 20 Most Important Features:")
        print("-" * 50)
        for i, row in feature_importance.head(20).iterrows():
            print(f"{row['feature']:40s}: {row['importance']:.4f}")

        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.feature_importance_scores = feature_importance
        return feature_importance

    def generate_feature_summary(self):
        """Generate summary of created features"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)

        feature_categories = {
            'Original Weather Features': self.top_weather_features,
            'Temporal Features': [f for f in self.engineered_features if
                                  any(x in f for x in ['hour', 'day', 'month', 'rush', 'weekend'])],
            'Threshold Features': [f for f in self.engineered_features if any(
                x in f for x in ['temp_', 'humidity_', 'wind_', 'visibility_', 'rain_', 'snow_'])],
            'Interaction Features': [f for f in self.engineered_features if 'interaction' in f],
            'Lag Features': [f for f in self.engineered_features if 'lag' in f or 'change' in f],
            'Rolling Features': [f for f in self.engineered_features if 'rolling' in f],
            'Composite Features': [f for f in self.engineered_features if
                                   any(x in f for x in ['severity', 'comfort', 'index'])]
        }

        for category, features in feature_categories.items():
            available_features = [f for f in features if f in self.feature_data.columns]
            print(f"\n{category}: {len(available_features)} features")
            if available_features:
                print("  " + ", ".join(available_features[:5]))
                if len(available_features) > 5:
                    print(f"  ... and {len(available_features) - 5} more")

        print(f"\nNext Steps:")
        print(f"1. Use top features from importance ranking for initial models")
        print(f"2. Experiment with feature selection techniques")
        print(f"3. Consider station-specific feature engineering")
        print(f"4. Test different prediction timeframes with these features")


def main():
    """Main execution function"""
    # Update with your database connection
    DATABASE_URL = "postgresql://neondb_owner:npg_VOXZBcRohC81@ep-spring-truth-ae312q45.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    engine = create_engine(DATABASE_URL)

    # Define top weather features from your correlation analysis
    # Update this list based on your exploration results
    top_features = [
        'temp_fahrenheit', 'feels_like_fahrenheit', 'humidity',
        'pressure', 'wind_speed', 'visibility', 'rain_1h',
        'snow_1h', 'weather_severity_score'
    ]

    # Initialize feature engineer
    engineer = WeatherFeatureEngineer(engine, top_features)

    # Run feature engineering pipeline
    engineered_data = engineer.run_feature_engineering_pipeline()

    # Evaluate feature importance
    feature_importance = engineer.evaluate_feature_importance()

    # Generate summary
    engineer.generate_feature_summary()

    print(f"\nFeature engineering complete! Generated files:")
    print(f"  - feature_importance.png")
    print(f"  - Engineered dataset ready for modeling")


if __name__ == "__main__":
    main()