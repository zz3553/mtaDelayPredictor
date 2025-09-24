import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')


class HybridWeatherFeatureEngineer:
    def __init__(self, engine):
        """
        Hybrid approach: Combine focused categorical features with powerful temporal features

        Best of both worlds:
        1. Focused interactions (humidity, temp-humidity, pressure-humidity, visibility)
        2. Rolling window features (weather persistence - the key to success)
        3. Lag features (past conditions matter more than current)
        4. Eliminate only true redundancy, keep predictive variations
        """
        self.engine = engine
        self.merged_data = None
        self.feature_data = None
        self.engineered_features = []

        # Keep the most predictive weather features (based on your results)
        self.core_weather_features = [
            'wind_speed',  # #1 predictor in focused model
            'feels_like_fahrenheit',  # Keep as primary temp measure
            'humidity',  # Strong delay predictor
            'pressure',  # For weather systems
            'visibility',  # For fog/mist detection
            'cloudiness',  # Weather system indicator
            'rain_1h'  # For binary rain features
        ]

    def load_and_merge_data(self):
        """Load and merge data with timezone handling"""
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

        # Handle timezone conversion
        train_delays['timestamp'] = pd.to_datetime(train_delays['timestamp'])
        train_delays['timestamp'] = train_delays['timestamp'].dt.tz_localize('UTC').dt.tz_convert(
            'US/Eastern').dt.tz_localize(None)
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

        # Round and merge
        train_delays['timestamp_rounded'] = train_delays['timestamp'].dt.round('15min')
        weather_data['timestamp_rounded'] = weather_data['timestamp'].dt.round('15min')

        self.merged_data = pd.merge(
            train_delays, weather_data,
            on='timestamp_rounded',
            how='inner',
            suffixes=('_delay', '_weather')
        ).sort_values('timestamp_delay')

        print(f"Merged dataset: {len(self.merged_data)} records")
        return self.merged_data

    def create_temporal_features(self, df):
        """Essential temporal features that performed well"""
        print("Creating temporal features...")

        # Basic time features
        df['hour'] = df['timestamp_delay'].dt.hour
        df['day_of_week'] = df['timestamp_delay'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

        # Minutes since rush hour start (this was important in previous model)
        df['minutes_since_rush_start'] = 0
        morning_rush_mask = df['is_morning_rush'] == 1
        evening_rush_mask = df['is_evening_rush'] == 1

        df.loc[morning_rush_mask, 'minutes_since_rush_start'] = (df.loc[morning_rush_mask, 'hour'] - 7) * 60 + df.loc[
            morning_rush_mask, 'timestamp_delay'].dt.minute
        df.loc[evening_rush_mask, 'minutes_since_rush_start'] = (df.loc[evening_rush_mask, 'hour'] - 17) * 60 + df.loc[
            evening_rush_mask, 'timestamp_delay'].dt.minute

        temporal_features = ['hour', 'day_of_week', 'is_weekend', 'is_morning_rush',
                             'is_evening_rush', 'is_rush_hour', 'minutes_since_rush_start']

        self.engineered_features.extend(temporal_features)
        return df

    def create_rolling_window_features(self, df):
        """
        CRITICAL: Rolling window features - these were the top performers
        Focus on wind_speed and rain_1h which showed highest importance
        """
        print("Creating rolling window features (KEY TO SUCCESS)...")

        # Sort by station and timestamp for proper rolling calculations
        df = df.sort_values(['station_name', 'timestamp_delay'])

        rolling_features = []

        # Focus on the most important weather variables for rolling windows
        key_variables = ['rain_1h', 'wind_speed', 'feels_like_fahrenheit', 'humidity', 'pressure']
        windows = [4, 8]  # 60min (4*15), 120min (8*15)

        for variable in key_variables:
            if variable in df.columns:
                for window in windows:
                    window_min = window * 15

                    # Rolling minimum (this was #1 and #2 in your original model!)
                    min_feature_name = f'{variable}_rolling_min_{window_min}min'
                    df[min_feature_name] = df.groupby('station_name')[variable].rolling(window,
                                                                                        min_periods=1).min().reset_index(
                        0, drop=True)
                    rolling_features.append(min_feature_name)

                    # Rolling mean
                    mean_feature_name = f'{variable}_rolling_mean_{window_min}min'
                    df[mean_feature_name] = df.groupby('station_name')[variable].rolling(window,
                                                                                         min_periods=1).mean().reset_index(
                        0, drop=True)
                    rolling_features.append(mean_feature_name)

                    # Rolling standard deviation (temperature volatility was important)
                    std_feature_name = f'{variable}_rolling_std_{window_min}min'
                    df[std_feature_name] = df.groupby('station_name')[variable].rolling(window,
                                                                                        min_periods=1).std().reset_index(
                        0, drop=True)
                    rolling_features.append(std_feature_name)

                    # Rolling max (for completeness)
                    max_feature_name = f'{variable}_rolling_max_{window_min}min'
                    df[max_feature_name] = df.groupby('station_name')[variable].rolling(window,
                                                                                        min_periods=1).max().reset_index(
                        0, drop=True)
                    rolling_features.append(max_feature_name)

        self.engineered_features.extend(rolling_features)
        print(f"Created {len(rolling_features)} rolling window features")
        return df

    def create_lag_features(self, df):
        """
        Lag features: Past weather conditions (15-60 min ago)
        Your original model showed these were very predictive
        """
        print("Creating lag features (weather history matters)...")

        # Sort by station and timestamp for proper lagging
        df = df.sort_values(['station_name', 'timestamp_delay'])

        lag_features = []
        lag_periods = [1, 2, 4]  # 15min, 30min, 1hr ago

        # Focus on key variables that showed importance
        key_lag_variables = ['wind_speed', 'rain_1h', 'feels_like_fahrenheit', 'humidity', 'pressure']

        for variable in key_lag_variables:
            if variable in df.columns:
                for lag in lag_periods:
                    lag_minutes = lag * 15

                    # Lag feature
                    lag_feature_name = f'{variable}_lag_{lag_minutes}min'
                    df[lag_feature_name] = df.groupby('station_name')[variable].shift(lag)
                    lag_features.append(lag_feature_name)

                    # Weather change features (current vs past)
                    change_feature_name = f'{variable}_change_{lag_minutes}min'
                    df[change_feature_name] = df[variable] - df[lag_feature_name]
                    lag_features.append(change_feature_name)

        self.engineered_features.extend(lag_features)
        print(f"Created {len(lag_features)} lag features")
        return df

    def create_focused_categorical_features(self, df):
        """
        Best categorical features from focused approach
        Keep the most impactful ones, skip redundant categories
        """
        print("Creating focused categorical features...")

        categorical_features = []

        # 1. Key humidity features (humidity was strong predictor)
        if 'humidity' in df.columns:
            df['humidity_high'] = (df['humidity'] >= 80).astype(int)
            df['humidity_very_high'] = (df['humidity'] >= 90).astype(int)
            df['humidity_extreme'] = ((df['humidity'] <= 25) | (df['humidity'] >= 95)).astype(int)
            categorical_features.extend(['humidity_high', 'humidity_very_high', 'humidity_extreme'])

        # 2. Temperature-humidity interactions (heat index was important)
        if 'feels_like_fahrenheit' in df.columns and 'humidity' in df.columns:
            df['heat_index_simple'] = df['feels_like_fahrenheit'] + 0.5 * (df['humidity'] - 50)
            df['temp_humidity_ratio'] = df['feels_like_fahrenheit'] / (df['humidity'] + 1)
            df['hot_humid'] = ((df['feels_like_fahrenheit'] >= 80) & (df['humidity'] >= 70)).astype(int)
            df['cold_humid'] = ((df['feels_like_fahrenheit'] <= 50) & (df['humidity'] >= 80)).astype(int)
            categorical_features.extend(['heat_index_simple', 'temp_humidity_ratio', 'hot_humid', 'cold_humid'])

        # 3. Weather system features (pressure-humidity combinations)
        if 'pressure' in df.columns and 'humidity' in df.columns:
            df['approaching_storm'] = ((df['pressure'] <= 1012) & (df['humidity'] >= 80)).astype(int)
            df['stable_clear'] = ((df['pressure'] >= 1020) & (df['humidity'] <= 60)).astype(int)
            df['weather_stability_index'] = (df['pressure'] - 1013) / 10 - (df['humidity'] - 50) / 20
            categorical_features.extend(['approaching_storm', 'stable_clear', 'weather_stability_index'])

        # 4. Visibility features (for fog/mist conditions)
        if 'visibility' in df.columns:
            df['visibility_poor'] = (df['visibility'] <= 5000).astype(int)
            df['fog_conditions'] = ((df['visibility'] <= 3000) & (df['humidity'] >= 85)).astype(int)
            if 'humidity' in df.columns:
                df['air_clarity_index'] = df['visibility'] / (df['humidity'] + 1)
            categorical_features.extend(['visibility_poor', 'fog_conditions', 'air_clarity_index'])

        # 5. Smart precipitation features (for sparse summer data)
        if 'rain_1h' in df.columns:
            df['has_rain'] = df['rain_1h'].notna().astype(int)
            df['rain_amount'] = df['rain_1h'].fillna(0)
            df['rain_heavy'] = (df['rain_amount'] > 5).astype(int)

            # Rain storm conditions (this was #2 most important in focused model)
            if 'humidity' in df.columns:
                df['rain_storm_conditions'] = ((df['has_rain'] == 1) & (df['humidity'] >= 85)).astype(int)
                categorical_features.append('rain_storm_conditions')

            categorical_features.extend(['has_rain', 'rain_amount', 'rain_heavy'])

        # 6. Rush hour interactions (these performed well)
        if 'is_rush_hour' in df.columns:
            for weather_var in ['feels_like_fahrenheit', 'wind_speed', 'humidity']:
                if weather_var in df.columns:
                    interaction_name = f'rush_{weather_var}_interaction'
                    df[interaction_name] = df['is_rush_hour'] * df[weather_var].fillna(0)
                    categorical_features.append(interaction_name)

        self.engineered_features.extend(categorical_features)
        print(f"Created {len(categorical_features)} focused categorical features")
        return df

    def run_hybrid_feature_engineering(self):
        """Run the hybrid feature engineering pipeline"""
        print("=" * 60)
        print("HYBRID WEATHER FEATURE ENGINEERING")
        print("Combining best of both approaches:")
        print("✅ Rolling windows (weather persistence)")
        print("✅ Lag features (past conditions)")
        print("✅ Focused interactions (smart categoricals)")
        print("✅ Temporal patterns (rush hour effects)")
        print("=" * 60)

        # Load data
        df = self.load_and_merge_data()

        # Create features in order of importance
        df = self.create_temporal_features(df)  # Rush hour patterns
        df = self.create_rolling_window_features(df)  # Weather persistence (KEY!)
        df = self.create_lag_features(df)  # Past conditions matter
        df = self.create_focused_categorical_features(df)  # Smart interactions

        self.feature_data = df

        print(f"\nHybrid feature engineering complete!")
        print(f"Total engineered features: {len(self.engineered_features)}")
        print(f"Dataset shape: {df.shape}")

        # Feature category breakdown
        rolling_features = [f for f in self.engineered_features if 'rolling' in f]
        lag_features = [f for f in self.engineered_features if 'lag' in f or 'change' in f]
        categorical_features = [f for f in self.engineered_features if
                                any(x in f for x in ['humidity_', 'temp_', 'weather_', 'fog_', 'rain_', 'rush_'])]
        temporal_features = [f for f in self.engineered_features if
                             any(x in f for x in ['hour', 'day_', 'weekend', 'rush_hour', 'minutes_since'])]

        print(f"\nFeature Breakdown:")
        print(f"  Rolling window features: {len(rolling_features)} (weather persistence)")
        print(f"  Lag features: {len(lag_features)} (past conditions)")
        print(f"  Categorical features: {len(categorical_features)} (smart interactions)")
        print(f"  Temporal features: {len(temporal_features)} (time patterns)")

        return df

    def evaluate_hybrid_model(self):
        """Evaluate the hybrid model performance"""
        print("\n" + "=" * 60)
        print("HYBRID MODEL EVALUATION")
        print("=" * 60)

        # Use core weather features + all engineered features
        all_features = self.core_weather_features + self.engineered_features

        # Filter to available numeric columns
        available_features = []
        for col in all_features:
            if col in self.feature_data.columns:
                try:
                    col_dtype = str(self.feature_data[col].dtype)
                    if any(dtype in col_dtype for dtype in ['int', 'float', 'bool']):
                        available_features.append(col)
                except:
                    continue

        X = self.feature_data[available_features].fillna(0)
        y = self.feature_data['delay_min']

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)

        print(f"Features for evaluation: {len(available_features)}")
        print(f"Samples: {len(X)}")

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Evaluate
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n🎯 HYBRID MODEL PERFORMANCE:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f} minutes")

        # Compare to previous models
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"  Original model:  R² = 0.6354, RMSE = 6.30")
        print(f"  Focused model:   R² = 0.4319, RMSE = 6.90")
        print(f"  Hybrid model:    R² = {r2:.4f}, RMSE = {np.sqrt(mse):.2f}")

        if r2 > 0.6354:
            print(f"  ✅ IMPROVEMENT! Hybrid approach beats original model")
        elif r2 > 0.4319:
            print(f"  ✅ RECOVERY! Better than focused-only approach")
        else:
            print(f"  ⚠️  Still optimizing - check feature combinations")

        # Feature importance with categories
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 20 Most Important Hybrid Features:")
        print("-" * 70)
        for i, row in feature_importance.head(20).iterrows():
            category = self._categorize_feature(row['feature'])
            print(f"{row['feature']:40s}: {row['importance']:.4f} ({category})")

        # Category performance summary
        self._analyze_category_performance(feature_importance)

        return feature_importance, r2, np.sqrt(mse)

    def _categorize_feature(self, feature_name):
        """Categorize features for reporting"""
        if 'rolling' in feature_name:
            return 'Rolling Window'
        elif 'lag' in feature_name or 'change' in feature_name:
            return 'Lag/Change'
        elif any(x in feature_name for x in ['hour', 'day_', 'weekend', 'minutes_since']):
            return 'Temporal'
        elif any(x in feature_name for x in ['rush_', '_interaction']):
            return 'Rush Interaction'
        elif any(x in feature_name for x in ['humidity_', 'temp_humidity', 'heat_', 'hot_', 'cold_']):
            return 'Humidity/Temp'
        elif any(x in feature_name for x in ['weather_', 'storm', 'stable', 'pressure']):
            return 'Weather System'
        elif any(x in feature_name for x in ['visibility', 'fog', 'air_clarity']):
            return 'Visibility'
        elif 'rain' in feature_name:
            return 'Precipitation'
        else:
            return 'Core Weather'

    def _analyze_category_performance(self, feature_importance):
        """Analyze which feature categories are most important"""
        print(f"\n📈 FEATURE CATEGORY ANALYSIS:")
        print("-" * 50)

        # Group by category and sum importance
        category_importance = {}
        for _, row in feature_importance.iterrows():
            category = self._categorize_feature(row['feature'])
            if category not in category_importance:
                category_importance[category] = 0
            category_importance[category] += row['importance']

        # Sort and display
        sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        for category, total_importance in sorted_categories:
            print(f"  {category:20s}: {total_importance:.4f}")

        # Insights
        top_category = sorted_categories[0][0]
        print(f"\n💡 KEY INSIGHT: '{top_category}' features are most predictive")

        if 'Rolling Window' in [cat[0] for cat in sorted_categories[:3]]:
            print("   ✅ Rolling windows confirmed as crucial for weather-delay prediction")
        if 'Lag/Change' in [cat[0] for cat in sorted_categories[:3]]:
            print("   ✅ Past weather conditions more important than current conditions")


def main():
    """Main execution function"""
    # Update with your database connection
    DATABASE_URL = "postgresql://neondb_owner:npg_VOXZBcRohC81@ep-spring-truth-ae312q45.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    engine = create_engine(DATABASE_URL)

    # Initialize hybrid feature engineer
    engineer = HybridWeatherFeatureEngineer(engine)

    # Run hybrid feature engineering
    engineered_data = engineer.run_hybrid_feature_engineering()

    # Evaluate hybrid model
    feature_importance, r2, rmse = engineer.evaluate_hybrid_model()

    print(f"\n" + "=" * 60)
    print("🎯 HYBRID FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Combined the best of both approaches:")
    print(f"• Weather persistence (rolling windows)")
    print(f"• Historical patterns (lag features)")
    print(f"• Smart interactions (categorical features)")
    print(f"• Temporal effects (rush hour patterns)")
    print(f"\nFinal Model: R² = {r2:.4f}, RMSE = {rmse:.2f} minutes")
    print(f"Ready for production deployment! 🚀")


if __name__ == "__main__":
    main()