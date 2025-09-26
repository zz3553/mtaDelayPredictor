import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime, timedelta
import requests
from sqlalchemy import create_engine
from typing import Dict, List

# Page config
st.set_page_config(
    page_title="MTA Delay Predictor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)


class MTADelayPredictor:
    def __init__(self, model_path: str, db_url: str, weather_api_key: str):
        """Initialize the MTA Delay Prediction Service"""
        self.model = self._load_model(model_path)
        self.engine = create_engine(db_url) if db_url else None
        self.weather_api_key = weather_api_key
        self.feature_columns = self._get_feature_columns()

        # NYC Station coordinates
        self.stations = {
            # Queens
            "Astoria-Ditmars Blvd": {"lat": 40.7839, "lon": -73.9126},
            "Astoria Blvd-Hoyt Av": {"lat": 40.7788, "lon": -73.9179},
            "30 Av": {"lat": 40.7711, "lon": -73.9216},
            "Broadway": {"lat": 40.7645, "lon": -73.9254},
            "36 Av": {"lat": 40.7628, "lon": -73.9295},
            "39 Av-Dutch Kills": {"lat": 40.7584, "lon": -73.9329},
            "Queensboro Plaza": {"lat": 40.7505, "lon": -73.9402},

            # Manhattan
            "Lexington Av/59 St": {"lat": 40.7626, "lon": -73.9672},
            "5 Av/59 St": {"lat": 40.7648, "lon": -73.9734},
            "57 St-7 Av": {"lat": 40.7640, "lon": -73.9798},
            "49 St": {"lat": 40.7599, "lon": -73.9840},
            "Times Sq-42 St": {"lat": 40.7554, "lon": -73.9877},
            "34 St-Herald Sq": {"lat": 40.7495, "lon": -73.9879},
            "28 St": {"lat": 40.7443, "lon": -73.9904},
            "23 St": {"lat": 40.7413, "lon": -73.9901},
            "14 St-Union Sq": {"lat": 40.7358, "lon": -73.9906},
            "8 St-NYU": {"lat": 40.7302, "lon": -73.9926},
            "Prince St": {"lat": 40.7246, "lon": -73.9972},
            "Canal St": {"lat": 40.7196, "lon": -74.0018},
            "City Hall": {"lat": 40.7130, "lon": -74.0062},
            "Cortlandt St": {"lat": 40.7093, "lon": -74.0116},
            "Rector St": {"lat": 40.7073, "lon": -74.0132},
            "Whitehall St-South Ferry": {"lat": 40.7032, "lon": -74.0129},

            # Brooklyn
            "Court St": {"lat": 40.6942, "lon": -73.9900},
            "Jay St-MetroTech": {"lat": 40.6924, "lon": -73.9873},
            "DeKalb Av": {"lat": 40.6908, "lon": -73.9818},
            "Atlantic Av-Barclays Ctr": {"lat": 40.6844, "lon": -73.9796},
            "Union St": {"lat": 40.6784, "lon": -73.9830},
            "9 St": {"lat": 40.6732, "lon": -73.9850},
            "Prospect Av": {"lat": 40.6655, "lon": -73.9868},
            "25 St": {"lat": 40.6603, "lon": -73.9893},
            "36 St": {"lat": 40.6551, "lon": -73.9942},
            "45 St": {"lat": 40.6477, "lon": -74.0011},
            "53 St": {"lat": 40.6416, "lon": -74.0084},
            "59 St": {"lat": 40.6369, "lon": -74.0112},
            "8 Av": {"lat": 40.6358, "lon": -74.0041},
            "Fort Hamilton Pkwy": {"lat": 40.6318, "lon": -74.0040},
            "New Utrecht Av": {"lat": 40.6288, "lon": -74.0004},
            "18 Av": {"lat": 40.6225, "lon": -73.9928},
            "20 Av": {"lat": 40.6186, "lon": -73.9897},
            "Bay Pkwy": {"lat": 40.6138, "lon": -73.9868},
            "Kings Hwy": {"lat": 40.6087, "lon": -73.9825},
            "Avenue U": {"lat": 40.6030, "lon": -73.9785},
            "86 St": {"lat": 40.5968, "lon": -73.9739},
            "Gravesend-86 St": {"lat": 40.5954, "lon": -73.9715},
            "Coney Island-Stillwell Av": {"lat": 40.5772, "lon": -73.9815}
        }

    @st.cache_resource
    def _load_model(_self, model_path: str):
        """Load the trained model with caching"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Failed to load model from {model_path}: {e}")
            return None

    def _get_feature_columns(self) -> List[str]:
        """Exact feature list from your trained model"""
        return [
            'wind_speed',
            'feels_like_fahrenheit',
            'humidity',
            'pressure',
            'visibility',
            'cloudiness',
            'rain_1h',
            'hour',
            'day_of_week',
            'is_weekend',
            'is_morning_rush',
            'is_evening_rush',
            'is_rush_hour',
            'minutes_since_rush_start',
            'rain_1h_rolling_min_60min',
            'rain_1h_rolling_mean_60min',
            'rain_1h_rolling_std_60min',
            'rain_1h_rolling_max_60min',
            'rain_1h_rolling_min_120min',
            'rain_1h_rolling_mean_120min',
            'rain_1h_rolling_std_120min',
            'rain_1h_rolling_max_120min',
            'wind_speed_rolling_min_60min',
            'wind_speed_rolling_mean_60min',
            'wind_speed_rolling_std_60min',
            'wind_speed_rolling_max_60min',
            'wind_speed_rolling_min_120min',
            'wind_speed_rolling_mean_120min',
            'wind_speed_rolling_std_120min',
            'wind_speed_rolling_max_120min',
            'feels_like_fahrenheit_rolling_min_60min',
            'feels_like_fahrenheit_rolling_mean_60min',
            'feels_like_fahrenheit_rolling_std_60min',
            'feels_like_fahrenheit_rolling_max_60min',
            'feels_like_fahrenheit_rolling_min_120min',
            'feels_like_fahrenheit_rolling_mean_120min',
            'feels_like_fahrenheit_rolling_std_120min',
            'feels_like_fahrenheit_rolling_max_120min',
            'humidity_rolling_min_60min',
            'humidity_rolling_mean_60min',
            'humidity_rolling_std_60min',
            'humidity_rolling_max_60min',
            'humidity_rolling_min_120min',
            'humidity_rolling_mean_120min',
            'humidity_rolling_std_120min',
            'humidity_rolling_max_120min',
            'pressure_rolling_min_60min',
            'pressure_rolling_mean_60min',
            'pressure_rolling_std_60min',
            'pressure_rolling_max_60min',
            'pressure_rolling_min_120min',
            'pressure_rolling_mean_120min',
            'pressure_rolling_std_120min',
            'pressure_rolling_max_120min',
            'wind_speed_lag_15min',
            'wind_speed_change_15min',
            'wind_speed_lag_30min',
            'wind_speed_change_30min',
            'wind_speed_lag_60min',
            'wind_speed_change_60min',
            'rain_1h_lag_15min',
            'rain_1h_change_15min',
            'rain_1h_lag_30min',
            'rain_1h_change_30min',
            'rain_1h_lag_60min',
            'rain_1h_change_60min',
            'feels_like_fahrenheit_lag_15min',
            'feels_like_fahrenheit_change_15min',
            'feels_like_fahrenheit_lag_30min',
            'feels_like_fahrenheit_change_30min',
            'feels_like_fahrenheit_lag_60min',
            'feels_like_fahrenheit_change_60min',
            'humidity_lag_15min',
            'humidity_change_15min',
            'humidity_lag_30min',
            'humidity_change_30min',
            'humidity_lag_60min',
            'humidity_change_60min',
            'pressure_lag_15min',
            'pressure_change_15min',
            'pressure_lag_30min',
            'pressure_change_30min',
            'pressure_lag_60min',
            'pressure_change_60min',
            'humidity_high',
            'humidity_very_high',
            'humidity_extreme',
            'heat_index_simple',
            'temp_humidity_ratio',
            'hot_humid',
            'cold_humid',
            'approaching_storm',
            'stable_clear',
            'weather_stability_index',
            'visibility_poor',
            'fog_conditions',
            'air_clarity_index',
            'rain_storm_conditions',
            'has_rain',
            'rain_amount',
            'rain_heavy',
            'rush_feels_like_fahrenheit_interaction',
            'rush_wind_speed_interaction',
            'rush_humidity_interaction',
        ]

    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def get_current_weather(_self, lat: float, lon: float) -> Dict:
        """Fetch current weather data with caching"""
        try:
            if not _self.weather_api_key or _self.weather_api_key == "demo_mode":
                # Demo mode with simulated data
                return _self._get_demo_weather(lat, lon)

            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': _self.weather_api_key,
                'units': 'imperial'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            weather_data = {
                'temp_fahrenheit': data['main']['temp'],
                'feels_like_fahrenheit': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'visibility': data.get('visibility', 10000),
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'cloudiness': data['clouds']['all'],
                'rain_1h': data.get('rain', {}).get('1h', None),
                'timestamp': datetime.now()
            }

            return weather_data

        except Exception as e:
            st.error(f"Failed to fetch weather data: {e}")
            return _self._get_demo_weather(lat, lon)

    def _get_demo_weather(self, lat: float, lon: float) -> Dict:
        """Generate realistic demo weather data"""
        base_temp = 75 + (lat - 40.75) * 2  # Vary by location
        current_hour = datetime.now().hour

        # Simulate daily temperature cycle
        temp_variation = 10 * np.sin((current_hour - 6) * np.pi / 12)
        temp = base_temp + temp_variation + np.random.normal(0, 3)

        return {
            'temp_fahrenheit': temp,
            'feels_like_fahrenheit': temp + np.random.normal(0, 2),
            'humidity': max(30, min(95, 65 + np.random.normal(0, 15))),
            'pressure': 1013 + np.random.normal(0, 5),
            'visibility': max(1000, 15000 + np.random.normal(0, 3000)),
            'wind_speed': max(0, np.random.exponential(5)),
            'cloudiness': max(0, min(100, np.random.normal(50, 25))),
            'rain_1h': np.random.choice([None, None, None, 1.2, 3.5], p=[0.8, 0.1, 0.05, 0.03, 0.02]),
            'timestamp': datetime.now()
        }

    @st.cache_data(ttl=900)
    def get_historical_weather(_self, station_name: str, hours_back: int = 2) -> pd.DataFrame:
        """Get historical weather data or simulate it"""
        try:
            if not _self.engine:
                return _self._simulate_historical_weather(hours_back)

            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)

            query = """
                SELECT timestamp, temp_fahrenheit, feels_like_fahrenheit, humidity, 
                       pressure, visibility, wind_speed, cloudiness, rain_1h
                FROM weather_data 
                WHERE timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp DESC
                LIMIT %s
            """

            df = pd.read_sql(query, _self.engine, params=[start_time, end_time, hours_back * 4])

            if len(df) < 4:  # If insufficient data, simulate
                return _self._simulate_historical_weather(hours_back)

            return df

        except Exception as e:
            st.warning(f"Using simulated historical data: {e}")
            return _self._simulate_historical_weather(hours_back)

    def _simulate_historical_weather(self, hours_back: int) -> pd.DataFrame:
        """Simulate historical weather data for demo"""
        timestamps = [datetime.now() - timedelta(minutes=15 * i) for i in range(hours_back * 4)]

        # Simulate realistic weather progression
        base_temp = 75
        base_humidity = 65

        data = []
        for i, ts in enumerate(timestamps):
            temp_trend = -2 * i / len(timestamps)  # Slight cooling trend
            humidity_trend = 5 * i / len(timestamps)  # Slight humidity increase

            data.append({
                'timestamp': ts,
                'temp_fahrenheit': base_temp + temp_trend + np.random.normal(0, 2),
                'feels_like_fahrenheit': base_temp + temp_trend + np.random.normal(0, 2),
                'humidity': max(30, min(95, base_humidity + humidity_trend + np.random.normal(0, 5))),
                'pressure': 1013 + np.random.normal(0, 3),
                'visibility': max(5000, 15000 + np.random.normal(0, 2000)),
                'wind_speed': max(0, np.random.exponential(4)),
                'cloudiness': max(0, min(100, 40 + np.random.normal(0, 20))),
                'rain_1h': np.random.choice([None, None, None, 0.5], p=[0.9, 0.05, 0.03, 0.02])
            })

        return pd.DataFrame(data)

    def engineer_features(self, current_weather: Dict, historical_weather: pd.DataFrame,
                          station_name: str) -> pd.DataFrame:
        """Engineer ALL 104 features that the trained model expects"""

        df = pd.DataFrame([current_weather])

        # Fill any None values in current weather with 0
        df = df.fillna(0)

        # Ensure we have rain_1h column (even if None/NaN)
        if 'rain_1h' not in df.columns:
            df['rain_1h'] = 0.0

        # Temporal features
        now = datetime.now()
        df['hour'] = now.hour
        df['day_of_week'] = now.weekday()
        df['is_weekend'] = int(df['day_of_week'].iloc[0] >= 5)
        df['is_morning_rush'] = int(7 <= df['hour'].iloc[0] <= 9)
        df['is_evening_rush'] = int(17 <= df['hour'].iloc[0] <= 19)
        df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']

        # Minutes since rush hour start
        if df['is_morning_rush'].iloc[0]:
            df['minutes_since_rush_start'] = (df['hour'].iloc[0] - 7) * 60 + now.minute
        elif df['is_evening_rush'].iloc[0]:
            df['minutes_since_rush_start'] = (df['hour'].iloc[0] - 17) * 60 + now.minute
        else:
            df['minutes_since_rush_start'] = 0

        # Ensure all core weather values are numeric
        core_weather_defaults = {
            'feels_like_fahrenheit': 70.0,
            'humidity': 50.0,
            'pressure': 1013.0,
            'wind_speed': 0.0,
            'visibility': 10000.0,
            'cloudiness': 50.0,
            'rain_1h': 0.0
        }

        for col, default_val in core_weather_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)

        # Core variables for calculations
        core_vars = ['feels_like_fahrenheit', 'humidity', 'pressure', 'wind_speed', 'visibility', 'rain_1h']

        # Create complete historical data with proper columns and fill NaN values
        if len(historical_weather) >= 8:
            hist_data = historical_weather.copy()

            # Fill NaN values in historical data
            for col, default_val in core_weather_defaults.items():
                if col not in hist_data.columns:
                    hist_data[col] = default_val
                else:
                    hist_data[col] = hist_data[col].fillna(default_val)

            # Rolling window calculations (60min=4 periods, 120min=8 periods)
            windows = [4, 8]
            window_names = ['60min', '120min']

            for var in core_vars:
                for window, window_name in zip(windows, window_names):
                    data_slice = hist_data[var].head(window)
                    data_slice = data_slice.fillna(df[var].iloc[0])

                    df[f'{var}_rolling_min_{window_name}'] = float(data_slice.min())
                    df[f'{var}_rolling_max_{window_name}'] = float(data_slice.max())
                    df[f'{var}_rolling_mean_{window_name}'] = float(data_slice.mean())
                    df[f'{var}_rolling_std_{window_name}'] = float(data_slice.std()) if len(data_slice) > 1 else 0.0

            # Lag features (15min=1 period, 30min=2 periods, 60min=4 periods)
            lags = [1, 2, 4]
            lag_names = ['15min', '30min', '60min']

            for var in core_vars:
                current_val = float(df[var].iloc[0])

                for lag, lag_name in zip(lags, lag_names):
                    if len(hist_data) > lag:
                        lag_val = float(hist_data[var].iloc[lag])
                        if pd.isna(lag_val):
                            lag_val = current_val
                    else:
                        lag_val = current_val

                    df[f'{var}_lag_{lag_name}'] = lag_val
                    df[f'{var}_change_{lag_name}'] = current_val - lag_val

        else:
            # Insufficient historical data - use current values as defaults
            for var in core_vars:
                current_val = float(df[var].iloc[0])

                # Rolling features
                for window_name in ['60min', '120min']:
                    df[f'{var}_rolling_min_{window_name}'] = current_val
                    df[f'{var}_rolling_max_{window_name}'] = current_val
                    df[f'{var}_rolling_mean_{window_name}'] = current_val
                    df[f'{var}_rolling_std_{window_name}'] = 0.0

                # Lag features
                for lag_name in ['15min', '30min', '60min']:
                    df[f'{var}_lag_{lag_name}'] = current_val
                    df[f'{var}_change_{lag_name}'] = 0.0

        # Convert all numeric columns to float
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Categorical features - using safe operations
        humidity_val = float(df['humidity'].iloc[0])
        temp_val = float(df['feels_like_fahrenheit'].iloc[0])
        pressure_val = float(df['pressure'].iloc[0])
        visibility_val = float(df['visibility'].iloc[0])
        rain_val = float(df['rain_1h'].iloc[0])

        # Core categorical features
        df['temp_humidity_ratio'] = temp_val / max(humidity_val + 1, 1)
        df['humidity_high'] = int(humidity_val >= 80)
        df['humidity_very_high'] = int(humidity_val >= 90)
        df['humidity_extreme'] = int((humidity_val <= 25) or (humidity_val >= 95))

        # Weather interaction features
        df['heat_index_simple'] = temp_val + 0.5 * (humidity_val - 50)
        df['hot_humid'] = int((temp_val >= 80) and (humidity_val >= 70))
        df['cold_humid'] = int((temp_val <= 50) and (humidity_val >= 70))

        # Weather system features
        df['approaching_storm'] = int((pressure_val <= 1012) and (humidity_val >= 80))
        df['stable_clear'] = int((pressure_val >= 1020) and (humidity_val <= 60))
        df['weather_stability_index'] = (pressure_val - 1013) / 10 - (humidity_val - 50) / 20

        # Visibility features
        df['visibility_poor'] = int(visibility_val <= 5000)
        df['fog_conditions'] = int((visibility_val <= 5000) and (humidity_val >= 85))
        df['air_clarity_index'] = visibility_val / max(humidity_val + 1, 1)

        # Precipitation features
        df['has_rain'] = int(rain_val > 0)
        df['rain_amount'] = rain_val
        df['rain_heavy'] = int(rain_val > 7)
        df['rain_storm_conditions'] = int((rain_val > 0) and (humidity_val >= 85))

        # Rush hour interactions
        rush_hour_val = float(df['is_rush_hour'].iloc[0])
        wind_val = float(df['wind_speed'].iloc[0])

        df['rush_feels_like_fahrenheit_interaction'] = rush_hour_val * temp_val
        df['rush_humidity_interaction'] = rush_hour_val * humidity_val
        df['rush_wind_speed_interaction'] = rush_hour_val * wind_val

        # Final cleanup - ensure all values are numeric
        df = df.fillna(0.0)

        # Convert all columns to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            except:
                df[col] = 0.0

        # Create a DataFrame with ALL required features, initialized to 0
        required_features = self.feature_columns
        final_df = pd.DataFrame(0.0, index=[0], columns=required_features)

        # Fill in the features we calculated
        for col in df.columns:
            if col in required_features:
                final_df[col] = df[col].iloc[0]

        # Ensure all values are float
        return final_df.astype(float)

    def predict_delay(self, station_name: str) -> Dict:
        """Make delay prediction for a specific station"""
        if not self.model:
            return {"error": "Model not loaded"}

        try:
            station_info = self.stations.get(station_name)
            if not station_info:
                return {"error": f"Station {station_name} not found"}

            lat, lon = station_info['lat'], station_info['lon']

            # Get weather data
            current_weather = self.get_current_weather(lat, lon)
            historical_weather = self.get_historical_weather(station_name)

            # Engineer features
            features_df = self.engineer_features(current_weather, historical_weather, station_name)

            # Make prediction
            predicted_delay = self.model.predict(features_df)[0]

            # Calculate confidence
            confidence = self._calculate_confidence(features_df, current_weather)

            return {
                'station_name': station_name,
                'predicted_delay_minutes': float(predicted_delay),
                'confidence_score': confidence,
                'weather_conditions': current_weather,
                'prediction_timestamp': datetime.now(),
                'recommendation': self._get_recommendation(predicted_delay)
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _calculate_confidence(self, features_df: pd.DataFrame, weather_data: Dict) -> float:
        """Calculate prediction confidence"""
        confidence = 0.85

        # Reduce confidence for extreme conditions
        temp = weather_data['feels_like_fahrenheit']
        humidity = weather_data['humidity']

        if temp < 30 or temp > 95:
            confidence -= 0.15
        if humidity > 90:
            confidence -= 0.1
        if weather_data.get('rain_1h'):
            confidence -= 0.05

        return max(0.5, confidence)

    def _get_recommendation(self, predicted_delay: float) -> str:
        """Get travel recommendation based on predicted delay"""
        if predicted_delay <= 2:
            return "Normal service expected"
        elif predicted_delay <= 5:
            return "Allow a few extra minutes"
        elif predicted_delay <= 10:
            return "Consider alternative routes"
        else:
            return "Significant delays expected - plan accordingly"


# Initialize the app
def initialize_app():
    """Initialize the Streamlit app"""
    st.title("🚇 MTA Delay Predictor")
    st.markdown("### Real-time delay predictions using weather data")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model path
        model_path = st.text_input(
            "Model Path",
            value="./mta_delay_model.pkl",
            help="Path to your trained model pickle file"
        )

        # Database URL
        db_url = st.text_input(
            "Database URL (Optional)",
            value="",
            help="PostgreSQL connection string for historical weather data"
        )

        # Weather API key
        weather_api_key = st.text_input(
            "Weather API Key (Optional)",
            value="demo_mode",
            type="password",
            help="OpenWeatherMap API key. Leave as 'demo_mode' for simulated data"
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.metric("R² Score", "57.4%")
        st.metric("RMSE", "6.36 min")
        st.metric("Training Data", "Aug-Sep 2025")

    # Initialize predictor
    if 'predictor' not in st.session_state:
        try:
            st.session_state.predictor = MTADelayPredictor(model_path, db_url, weather_api_key)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize predictor: {e}")
            st.stop()

    return st.session_state.predictor


def main():
    """Main Streamlit app"""
    predictor = initialize_app()

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎯 Make Predictions")

        # Station selection
        station_name = st.selectbox(
            "Select MTA Station",
            options=list(predictor.stations.keys()),
            help="Choose a station to get delay predictions"
        )

        # Prediction button
        if st.button("🔮 Predict Delays", type="primary"):
            with st.spinner("Analyzing weather conditions and predicting delays..."):
                result = predictor.predict_delay(station_name)

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Display prediction results
                    st.success("✅ Prediction Complete!")

                    # Main metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric(
                            "Predicted Delay",
                            f"{result['predicted_delay_minutes']:.1f} min",
                            delta=f"{result['predicted_delay_minutes'] - 1.77:.1f} vs avg"
                        )
                    with col_b:
                        st.metric(
                            "Confidence",
                            f"{result['confidence_score']:.0%}"
                        )
                    with col_c:
                        st.metric(
                            "Recommendation",
                            result['recommendation']
                        )

                    # Weather conditions
                    st.subheader("🌤️ Current Weather Conditions")
                    weather = result['weather_conditions']

                    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                    with col_w1:
                        st.metric("Temperature", f"{weather['feels_like_fahrenheit']:.1f}°F")
                    with col_w2:
                        st.metric("Humidity", f"{weather['humidity']:.0f}%")
                    with col_w3:
                        st.metric("Wind Speed", f"{weather['wind_speed']:.1f} mph")
                    with col_w4:
                        st.metric("Visibility", f"{weather['visibility'] / 1000:.1f} km")

                    # Store results for batch processing
                    if 'results' not in st.session_state:
                        st.session_state.results = []
                    st.session_state.results.append(result)

        # Batch predictions
        st.header("📊 Batch Predictions")
        if st.button("🚇 Predict All Stations"):
            with st.spinner("Processing all stations..."):
                batch_results = []
                progress_bar = st.progress(0)

                for i, station in enumerate(predictor.stations.keys()):
                    result = predictor.predict_delay(station)
                    if "error" not in result:
                        batch_results.append({
                            'Station': result['station_name'],
                            'Predicted Delay (min)': result['predicted_delay_minutes'],
                            'Confidence': result['confidence_score'],
                            'Temperature (°F)': result['weather_conditions']['feels_like_fahrenheit'],
                            'Humidity (%)': result['weather_conditions']['humidity'],
                            'Recommendation': result['recommendation']
                        })
                    progress_bar.progress((i + 1) / len(predictor.stations))

                if batch_results:
                    df_results = pd.DataFrame(batch_results)
                    st.dataframe(df_results, use_container_width=True)

                    # Visualization
                    fig = px.bar(
                        df_results,
                        x='Station',
                        y='Predicted Delay (min)',
                        color='Confidence',
                        title="Predicted Delays Across All Stations",
                        color_continuous_scale="RdYlGn"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("📈 Live Dashboard")

        # Real-time clock
        placeholder = st.empty()
        with placeholder.container():
            current_time = datetime.now()
            st.metric("Current Time", current_time.strftime("%H:%M:%S"))

            # Rush hour indicator
            hour = current_time.hour
            if 7 <= hour <= 9:
                st.success("🌅 Morning Rush Hour")
            elif 17 <= hour <= 19:
                st.warning("🌆 Evening Rush Hour")
            else:
                st.info("⏰ Off-Peak Hours")

        # Recent predictions
        if 'results' in st.session_state and st.session_state.results:
            st.subheader("📋 Recent Predictions")
            recent = st.session_state.results[-5:]  # Last 5 predictions

            for result in reversed(recent):
                with st.expander(f"{result['station_name']}: {result['predicted_delay_minutes']:.1f} min"):
                    st.write(f"**Confidence:** {result['confidence_score']:.0%}")
                    st.write(f"**Time:** {result['prediction_timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Recommendation:** {result['recommendation']}")

        # Performance info
        st.subheader("🎯 Model Performance")
        st.info("""
        **Hybrid Weather Model**
        - R² Score: 57.4%
        - RMSE: 6.36 minutes
        - Features: 97 engineered
        - Training: Aug-Sep 2025
        """)

        # Tips
        st.subheader("💡 Tips")
        st.info("""
        - **Morning rush:** 7-9 AM
        - **Evening rush:** 5-7 PM
        - **Weather impacts:** High humidity and temperature changes increase delays
        - **Best accuracy:** During rush hours with recent weather data
        """)


if __name__ == "__main__":
    main()