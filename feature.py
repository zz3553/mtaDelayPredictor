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

    def run_seasonal_optimized_feature_engineering(self):
        """Run optimized feature engineering for all-season data collection"""
        print("=" * 60)
        print("SEASONAL WEATHER FEATURE ENGINEERING")
        print("Optimized for year-round data collection:")
        print("✅ Core temporal patterns (always important)")
        print("✅ Weather persistence (120min rolling windows)")
        print("✅ Seasonal adaptability (temperature ranges)")
        print("✅ Precipitation variety (rain, snow, mixed)")
        print("✅ Equipment stress indicators")
        print("=" * 60)

        # Load data
        df = self.load_and_merge_data()

        # Create seasonal-optimized features in the correct order
        df = self.create_essential_temporal_features(df)  # Basic temporal features
        df = self.create_core_weather_persistence_features(df)  # Rolling windows and lag features
        df = self.create_seasonal_transition_features(df)  # Spring/fall rates (after lag exists)
        df = self.create_seasonal_weather_features(df)  # Season-specific patterns
        df = self.create_equipment_stress_features(df)  # Equipment stress indicators

        self.feature_data = df

        print(f"\nSeasonal feature engineering complete!")
        print(f"Total optimized features: {len(self.engineered_features)}")
        print(f"Dataset shape: {df.shape}")

        return df

    def create_essential_temporal_features(self, df):
        """Essential temporal features that work year-round with specific seasonal patterns"""
        print("Creating essential temporal features...")

        # Core time features
        df['hour'] = df['timestamp_delay'].dt.hour
        df['day_of_week'] = df['timestamp_delay'].dt.dayofweek
        df['month'] = df['timestamp_delay'].dt.month  # Critical for seasonal patterns
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Rush hour patterns
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        df['minutes_since_rush_start'] = 0

        morning_rush_mask = df['is_morning_rush'] == 1
        evening_rush_mask = df['is_evening_rush'] == 1
        df.loc[morning_rush_mask, 'minutes_since_rush_start'] = (df.loc[morning_rush_mask, 'hour'] - 7) * 60 + df.loc[
            morning_rush_mask, 'timestamp_delay'].dt.minute
        df.loc[evening_rush_mask, 'minutes_since_rush_start'] = (df.loc[evening_rush_mask, 'hour'] - 17) * 60 + df.loc[
            evening_rush_mask, 'timestamp_delay'].dt.minute

        # Specific seasonal indicators (more precise than transition seasons)
        df['is_winter'] = (df['month'].isin([12, 1, 2])).astype(int)
        df['is_spring'] = (df['month'].isin([3, 4, 5])).astype(int)
        df['is_summer'] = (df['month'].isin([6, 7, 8])).astype(int)
        df['is_fall'] = (df['month'].isin([9, 10, 11])).astype(int)

        # Temperature deviation from seasonal norms (NYC-specific)
        df['temp_deviation_seasonal'] = 0
        seasonal_norms = {1: 35, 2: 38, 3: 48, 4: 58, 5: 68, 6: 76,
                          7: 80, 8: 78, 9: 71, 10: 60, 11: 50, 12: 40}

        for month, norm_temp in seasonal_norms.items():
            month_mask = df['month'] == month
            if 'feels_like_fahrenheit' in df.columns:
                df.loc[month_mask, 'temp_deviation_seasonal'] = abs(
                    df.loc[month_mask, 'feels_like_fahrenheit'] - norm_temp)

        # NOTE: Spring/fall rate features will be added AFTER lag features are created

        temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_morning_rush',
                             'is_evening_rush', 'is_rush_hour', 'minutes_since_rush_start',
                             'is_winter', 'is_spring', 'is_summer', 'is_fall', 'temp_deviation_seasonal']

        self.engineered_features.extend(temporal_features)
        return df

    def create_seasonal_transition_features(self, df):
        """Create seasonal transition features AFTER lag features exist"""
        print("Creating seasonal transition features...")

        transition_features = []

        # Now we can safely create these features since lag features exist
        if 'feels_like_fahrenheit_change_60min' in df.columns:
            # Seasonal transition features (spring warming vs fall cooling)
            df['spring_warming_rate'] = (df['is_spring'] * df['feels_like_fahrenheit_change_60min']).clip(lower=0)
            df['fall_cooling_rate'] = (df['is_fall'] * abs(df['feels_like_fahrenheit_change_60min'].clip(upper=0)))
            transition_features.extend(['spring_warming_rate', 'fall_cooling_rate'])
        else:
            # Fallback if lag features don't exist
            df['spring_warming_rate'] = 0
            df['fall_cooling_rate'] = 0
            transition_features.extend(['spring_warming_rate', 'fall_cooling_rate'])

        self.engineered_features.extend(transition_features)
        return df

    def create_core_weather_persistence_features(self, df):
        """Core weather persistence features that matter most"""
        print("Creating core weather persistence features...")

        df = df.sort_values(['station_name', 'timestamp_delay'])

        # Focus on the most critical variables and time windows
        critical_variables = ['feels_like_fahrenheit', 'humidity', 'wind_speed', 'pressure']
        precipitation_variables = ['rain_1h', 'snow_1h'] if 'snow_1h' in df.columns else ['rain_1h']

        # 120-minute windows (most predictive from your analysis)
        window = 8  # 120 minutes
        window_name = '120min'

        persistence_features = []

        for variable in critical_variables:
            if variable in df.columns:
                # Most important statistics from your analysis
                df[f'{variable}_rolling_mean_{window_name}'] = df.groupby('station_name')[variable].rolling(window,
                                                                                                            min_periods=1).mean().reset_index(
                    0, drop=True)
                df[f'{variable}_rolling_std_{window_name}'] = df.groupby('station_name')[variable].rolling(window,
                                                                                                           min_periods=1).std().reset_index(
                    0, drop=True)
                df[f'{variable}_rolling_min_{window_name}'] = df.groupby('station_name')[variable].rolling(window,
                                                                                                           min_periods=1).min().reset_index(
                    0, drop=True)
                df[f'{variable}_rolling_max_{window_name}'] = df.groupby('station_name')[variable].rolling(window,
                                                                                                           min_periods=1).max().reset_index(
                    0, drop=True)

                persistence_features.extend([
                    f'{variable}_rolling_mean_{window_name}',
                    f'{variable}_rolling_std_{window_name}',
                    f'{variable}_rolling_min_{window_name}',
                    f'{variable}_rolling_max_{window_name}'
                ])

        # Precipitation persistence (critical for all seasons)
        for variable in precipitation_variables:
            if variable in df.columns:
                df[f'{variable}_rolling_min_60min'] = df.groupby('station_name')[variable].rolling(4,
                                                                                                   min_periods=1).min().reset_index(
                    0, drop=True)
                df[f'{variable}_rolling_max_120min'] = df.groupby('station_name')[variable].rolling(window,
                                                                                                    min_periods=1).max().reset_index(
                    0, drop=True)
                persistence_features.extend([f'{variable}_rolling_min_60min', f'{variable}_rolling_max_120min'])

        # Weather change detection (60-min lag) - but only if we have enough data
        for variable in critical_variables:
            if variable in df.columns:
                lag_feature = f'{variable}_lag_60min'
                change_feature = f'{variable}_change_60min'
                df[lag_feature] = df.groupby('station_name')[variable].shift(4)
                df[change_feature] = df[variable] - df[lag_feature]
                persistence_features.extend([lag_feature, change_feature])

        # Fill NaN values that may result from rolling/lag calculations
        for feature in persistence_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)

        self.engineered_features.extend(persistence_features)
        print(f"Created {len(persistence_features)} weather persistence features")
        return df

    def create_seasonal_weather_features(self, df):
        """Weather features that adapt to specific seasonal patterns"""
        print("Creating specific seasonal weather features...")

        seasonal_features = []

        # Temperature features that work year-round
        if 'feels_like_fahrenheit' in df.columns:
            df['temp_extreme_cold'] = (df['feels_like_fahrenheit'] <= 20).astype(int)  # Winter
            df['temp_freezing'] = (df['feels_like_fahrenheit'] <= 32).astype(int)  # Winter/Spring
            df['temp_hot'] = (df['feels_like_fahrenheit'] >= 85).astype(int)  # Summer
            df['temp_extreme_hot'] = (df['feels_like_fahrenheit'] >= 95).astype(int)  # Summer

            seasonal_features.extend(['temp_extreme_cold', 'temp_freezing', 'temp_hot', 'temp_extreme_hot'])

        # Spring-specific weather patterns
        if 'is_spring' in df.columns and 'feels_like_fahrenheit' in df.columns:
            # Spring warming spells (rapid temperature increases)
            df['spring_warm_spell'] = ((df['is_spring'] == 1) & (df['feels_like_fahrenheit'] >= 70) &
                                       (df['feels_like_fahrenheit_change_60min'] > 5)).astype(int)

            # Spring freeze risk (warm days followed by potential freezing)
            df['spring_freeze_risk'] = ((df['is_spring'] == 1) & (df['feels_like_fahrenheit'] <= 35) &
                                        (df['feels_like_fahrenheit_rolling_max_120min'] > 50)).astype(int)

            seasonal_features.extend(['spring_warm_spell', 'spring_freeze_risk'])

        # Fall-specific weather patterns
        if 'is_fall' in df.columns and 'feels_like_fahrenheit' in df.columns:
            # Fall cooling trends (consistent temperature drops)
            df['fall_cooling_trend'] = ((df['is_fall'] == 1) &
                                        (df['feels_like_fahrenheit_change_60min'] < -3)).astype(int)

            # Early winter preparation (fall + cold conditions)
            df['early_winter_conditions'] = ((df['is_fall'] == 1) & (df['feels_like_fahrenheit'] <= 40)).astype(int)

            seasonal_features.extend(['fall_cooling_trend', 'early_winter_conditions'])

        # Humidity features (important all seasons)
        if 'humidity' in df.columns:
            df['humidity_high'] = (df['humidity'] >= 80).astype(int)
            df['humidity_very_high'] = (df['humidity'] >= 90).astype(int)
            seasonal_features.extend(['humidity_high', 'humidity_very_high'])

        # Enhanced precipitation features for all seasons
        if 'rain_1h' in df.columns:
            df['has_rain'] = df['rain_1h'].notna().astype(int)
            df['rain_amount'] = df['rain_1h'].fillna(0)
            df['rain_heavy'] = (df['rain_amount'] > 7).astype(int)
            seasonal_features.extend(['has_rain', 'rain_amount', 'rain_heavy'])

        if 'snow_1h' in df.columns:
            df['has_snow'] = df['snow_1h'].notna().astype(int)
            df['snow_amount'] = df['snow_1h'].fillna(0)
            df['snow_heavy'] = (df['snow_amount'] > 2).astype(int)
            seasonal_features.extend(['has_snow', 'snow_amount', 'snow_heavy'])

        # Weather system features
        if 'pressure' in df.columns and 'humidity' in df.columns:
            df['approaching_storm'] = ((df['pressure'] <= 1012) & (df['humidity'] >= 80)).astype(int)
            df['stable_clear'] = ((df['pressure'] >= 1020) & (df['humidity'] <= 60)).astype(int)
            df['weather_stability_index'] = (df['pressure'] - 1013) / 10 - (df['humidity'] - 50) / 20
            seasonal_features.extend(['approaching_storm', 'stable_clear', 'weather_stability_index'])

        # Season-specific weather system interactions
        if 'is_spring' in df.columns and 'approaching_storm' in df.columns:
            df['spring_storm'] = (df['is_spring'] * df['approaching_storm']).astype(int)
            seasonal_features.append('spring_storm')

        if 'is_fall' in df.columns and 'stable_clear' in df.columns:
            df['fall_high_pressure'] = (df['is_fall'] * df['stable_clear']).astype(int)
            seasonal_features.append('fall_high_pressure')

        # Multi-season interaction features
        if 'feels_like_fahrenheit' in df.columns and 'humidity' in df.columns:
            df['heat_index_simple'] = df['feels_like_fahrenheit'] + 0.5 * (df['humidity'] - 50)
            df['temp_humidity_ratio'] = df['feels_like_fahrenheit'] / (df['humidity'] + 1)
            seasonal_features.extend(['heat_index_simple', 'temp_humidity_ratio'])

        # Critical rush hour interaction (your #3 most important feature)
        if 'is_rush_hour' in df.columns and 'wind_speed' in df.columns:
            df['rush_wind_speed_interaction'] = df['is_rush_hour'] * df['wind_speed'].fillna(0)
            seasonal_features.append('rush_wind_speed_interaction')

        self.engineered_features.extend(seasonal_features)
        print(f"Created {len(seasonal_features)} specific seasonal weather features")
        return df

    def create_equipment_stress_features(self, df):
        """Features that capture equipment stress across seasons"""
        print("Creating equipment stress features...")

        equipment_features = []

        # Multi-season equipment stress
        if 'feels_like_fahrenheit' in df.columns and 'humidity' in df.columns:
            # Cold + wet = rail/electrical issues
            df['cold_wet_stress'] = ((df['feels_like_fahrenheit'] <= 35) & (df['humidity'] >= 80)).astype(int)
            # Hot + humid = expansion/cooling issues
            df['hot_humid_stress'] = ((df['feels_like_fahrenheit'] >= 85) & (df['humidity'] >= 70)).astype(int)
            equipment_features.extend(['cold_wet_stress', 'hot_humid_stress'])

        # Visibility-based operational stress
        if 'visibility' in df.columns and 'humidity' in df.columns:
            df['poor_visibility'] = (df['visibility'] <= 5000).astype(int)
            df['fog_conditions'] = ((df['visibility'] <= 3000) & (df['humidity'] >= 85)).astype(int)
            equipment_features.extend(['poor_visibility', 'fog_conditions'])

        # Precipitation + temperature combinations (freeze/thaw, rain/snow mix)
        if 'feels_like_fahrenheit' in df.columns:
            if 'rain_amount' in df.columns:  # Use rain_amount instead of rain_1h for consistency
                df['freezing_rain_risk'] = ((df['feels_like_fahrenheit'] <= 34) & (df['rain_amount'] > 0)).astype(int)
                equipment_features.append('freezing_rain_risk')

            if 'has_snow' in df.columns and 'has_rain' in df.columns:
                df['mixed_precipitation'] = ((df['has_snow'] == 1) & (df['has_rain'] == 1)).astype(int)
                equipment_features.append('mixed_precipitation')

        # Fill any NaN values
        for feature in equipment_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0).astype(int)

        self.engineered_features.extend(equipment_features)
        print(f"Created {len(equipment_features)} equipment stress features")
        return df

    # Include the old method name for backward compatibility
    def run_hybrid_feature_engineering(self):
        """Backward compatibility - calls the seasonal optimized version"""
        print("Note: run_hybrid_feature_engineering() is now run_seasonal_optimized_feature_engineering()")
        return self.run_seasonal_optimized_feature_engineering()

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

        # SAVE THE MODEL AND DATASET FOR FEATURE ANALYSIS
        print(f"\n💾 SAVING MODEL AND DATASET FOR ANALYSIS...")

        import pickle

        # Save the trained Random Forest model
        with open('mta_delay_model.pkl', 'wb') as f:
            pickle.dump(rf, f)

        # Save feature columns for reference
        with open('feature_columns.pkl', 'wb') as f:
            pickle.dump(available_features, f)

        # SAVE THE COMPLETE DATASET FOR FEATURE SELECTION ANALYSIS
        with open('mta_features_dataset.csv', 'w') as f:
            self.feature_data.to_csv(f, index=False)

        print(f"✅ Saved for analysis:")
        print(f"   - Model: mta_delay_model.pkl")
        print(f"   - Features: feature_columns.pkl")
        print(f"   - Dataset: mta_features_dataset.csv ({self.feature_data.shape})")
        print(f"   - Ready for feature selection analysis!")

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