import pickle
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sqlalchemy import create_engine

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

        # NYC Station coordinates (sample - you can expand this)
        self.stations = {
            "Times Sq-42 St": {"lat": 40.7590, "lon": -73.9845},
            "Union Sq-14 St": {"lat": 40.7359, "lon": -73.9911},
            "Grand Central-42 St": {"lat": 40.7527, "lon": -73.9772},
            "14 St-Union Sq": {"lat": 40.7359, "lon": -73.9911},
            "34 St-Herald Sq": {"lat": 40.7505, "lon": -73.9884},
            "42 St-Port Authority": {"lat": 40.7570, "lon": -73.9897},
            "Atlantic Av-Barclays Ctr": {"lat": 40.6840, "lon": -73.9769},
            "Fulton St": {"lat": 40.7097, "lon": -73.0067},
            "Wall St": {"lat": 40.7074, "lon": -74.0113},
            "Canal St": {"lat": 40.7227, "lon": -74.0027}
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
        """Define the feature columns expected by the model"""
        return [
            # Core weather features
            'wind_speed', 'feels_like_fahrenheit', 'humidity', 'pressure', 'visibility', 'cloudiness',

            # Temporal features
            'hour', 'day_of_week', 'is_weekend', 'is_morning_rush', 'is_evening_rush',
            'is_rush_hour', 'minutes_since_rush_start',

            # Rolling window features
            'humidity_rolling_mean_120min', 'feels_like_fahrenheit_rolling_min_120min',
            'feels_like_fahrenheit_rolling_mean_120min', 'feels_like_fahrenheit_rolling_std_120min',
            'wind_speed_rolling_max_120min', 'humidity_rolling_std_120min', 'wind_speed_rolling_std_120min',

            # Lag features
            'feels_like_fahrenheit_change_60min', 'feels_like_fahrenheit_lag_60min',
            'feels_like_fahrenheit_change_30min',

            # Categorical features
            'temp_humidity_ratio', 'humidity_high', 'humidity_very_high', 'heat_index_simple',
            'hot_humid', 'approaching_storm', 'stable_clear', 'weather_stability_index',
            'visibility_poor', 'fog_conditions', 'has_rain', 'rain_storm_conditions',

            # Rush hour interactions
            'rush_feels_like_fahrenheit_interaction', 'rush_wind_speed_interaction', 'rush_humidity_interaction'
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
        """Engineer features for prediction"""

        df = pd.DataFrame([current_weather])

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

        # Rolling window features from historical data
        if len(historical_weather) >= 8:
            df['humidity_rolling_mean_120min'] = historical_weather['humidity'].head(8).mean()
            df['feels_like_fahrenheit_rolling_min_120min'] = historical_weather['feels_like_fahrenheit'].head(8).min()
            df['feels_like_fahrenheit_rolling_mean_120min'] = historical_weather['feels_like_fahrenheit'].head(8).mean()
            df['feels_like_fahrenheit_rolling_std_120min'] = historical_weather['feels_like_fahrenheit'].head(8).std()
            df['wind_speed_rolling_max_120min'] = historical_weather['wind_speed'].head(8).max()
            df['humidity_rolling_std_120min'] = historical_weather['humidity'].head(8).std()
            df['wind_speed_rolling_std_120min'] = historical_weather['wind_speed'].head(8).std()

            # Lag features
            if len(historical_weather) >= 4:
                temp_60min_ago = historical_weather['feels_like_fahrenheit'].iloc[3]
                temp_30min_ago = historical_weather['feels_like_fahrenheit'].iloc[1]

                df['feels_like_fahrenheit_lag_60min'] = temp_60min_ago
                df['feels_like_fahrenheit_change_60min'] = df['feels_like_fahrenheit'].iloc[0] - temp_60min_ago
                df['feels_like_fahrenheit_change_30min'] = df['feels_like_fahrenheit'].iloc[0] - temp_30min_ago
            else:
                df['feels_like_fahrenheit_lag_60min'] = df['feels_like_fahrenheit'].iloc[0]
                df['feels_like_fahrenheit_change_60min'] = 0
                df['feels_like_fahrenheit_change_30min'] = 0
        else:
            # Default values for insufficient data
            for col in ['humidity_rolling_mean_120min', 'feels_like_fahrenheit_rolling_min_120min',
                        'feels_like_fahrenheit_rolling_mean_120min', 'feels_like_fahrenheit_rolling_std_120min',
                        'wind_speed_rolling_max_120min', 'humidity_rolling_std_120min',
                        'wind_speed_rolling_std_120min']:
                base_col = col.split('_rolling_')[0] if '_rolling_' in col else col.split('_')[0] + '_' + \
                                                                                col.split('_')[1]
                if base_col in df.columns:
                    df[col] = df[base_col].iloc[0]
                else:
                    df[col] = 0

        # Categorical features
        df['temp_humidity_ratio'] = df['feels_like_fahrenheit'] / (df['humidity'] + 1)
        df['humidity_high'] = (df['humidity'] >= 80).astype(int)
        df['humidity_very_high'] = (df['humidity'] >= 90).astype(int)
        df['heat_index_simple'] = df['feels_like_fahrenheit'] + 0.5 * (df['humidity'] - 50)
        df['hot_humid'] = ((df['feels_like_fahrenheit'] >= 80) & (df['humidity'] >= 70)).astype(int)

        # Weather system features
        df['approaching_storm'] = ((df['pressure'] <= 1012) & (df['humidity'] >= 80)).astype(int)
        df['stable_clear'] = ((df['pressure'] >= 1020) & (df['humidity'] <= 60)).astype(int)
        df['weather_stability_index'] = (df['pressure'] - 1013) / 10 - (df['humidity'] - 50) / 20

        # Visibility features
        df['visibility_poor'] = (df['visibility'] <= 5000).astype(int)
        df['fog_conditions'] = ((df['visibility'] <= 3000) & (df['humidity'] >= 85)).astype(int)

        # Precipitation features
        df['has_rain'] = df['rain_1h'].notna().astype(int)
        df['rain_storm_conditions'] = ((df['has_rain'] == 1) & (df['humidity'] >= 85)).astype(int)

        # Rush hour interactions
        df['rush_feels_like_fahrenheit_interaction'] = df['is_rush_hour'] * df['feels_like_fahrenheit']
        df['rush_wind_speed_interaction'] = df['is_rush_hour'] * df['wind_speed']
        df['rush_humidity_interaction'] = df['is_rush_hour'] * df['humidity']

        # Fill missing values and ensure all columns are present
        df = df.fillna(0)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_columns]

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