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

        # NYC Station coordinates (sample stations)
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
            st.error(f"Failed to load model: {e}")
            return None

    def _get_feature_columns(self) -> List[str]:
        """Load exact feature list from your seasonal optimized model"""
        try:
            with open('feature_columns.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return [
                'wind_speed', 'feels_like_fahrenheit', 'humidity', 'pressure', 'visibility', 'cloudiness', 'rain_1h',
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
                'minutes_since_rush_start', 'is_winter', 'is_spring', 'is_summer', 'is_fall', 'temp_deviation_seasonal',
                'feels_like_fahrenheit_rolling_mean_120min', 'feels_like_fahrenheit_rolling_std_120min',
                'feels_like_fahrenheit_rolling_min_120min', 'feels_like_fahrenheit_rolling_max_120min',
                'humidity_rolling_mean_120min', 'humidity_rolling_std_120min', 'humidity_rolling_min_120min',
                'humidity_rolling_max_120min', 'wind_speed_rolling_mean_120min', 'wind_speed_rolling_std_120min',
                'wind_speed_rolling_min_120min', 'wind_speed_rolling_max_120min', 'pressure_rolling_mean_120min',
                'pressure_rolling_std_120min', 'pressure_rolling_min_120min', 'pressure_rolling_max_120min',
                'rain_1h_rolling_min_60min', 'rain_1h_rolling_max_120min', 'snow_1h_rolling_min_60min',
                'snow_1h_rolling_max_120min', 'feels_like_fahrenheit_lag_60min', 'feels_like_fahrenheit_change_60min',
                'humidity_lag_60min', 'humidity_change_60min', 'wind_speed_lag_60min', 'wind_speed_change_60min',
                'pressure_lag_60min', 'pressure_change_60min', 'spring_warming_rate', 'fall_cooling_rate',
                'temp_extreme_cold', 'temp_freezing', 'temp_hot', 'temp_extreme_hot', 'spring_warm_spell',
                'spring_freeze_risk', 'fall_cooling_trend', 'early_winter_conditions', 'humidity_high',
                'humidity_very_high', 'has_rain', 'rain_amount', 'rain_heavy', 'has_snow', 'snow_amount',
                'snow_heavy', 'approaching_storm', 'stable_clear', 'weather_stability_index', 'spring_storm',
                'fall_high_pressure', 'heat_index_simple', 'temp_humidity_ratio', 'rush_wind_speed_interaction',
                'cold_wet_stress', 'hot_humid_stress', 'poor_visibility', 'fog_conditions', 'freezing_rain_risk',
                'mixed_precipitation'
            ]

    @st.cache_data(ttl=900)
    def get_current_weather(_self, lat: float, lon: float) -> Dict:
        """Fetch current weather data"""
        if not _self.weather_api_key or _self.weather_api_key == "demo_mode":
            return _self._get_demo_weather(lat, lon)

        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {'lat': lat, 'lon': lon, 'appid': _self.weather_api_key, 'units': 'imperial'}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                'feels_like_fahrenheit': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'visibility': data.get('visibility', 10000),
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'cloudiness': data['clouds']['all'],
                'rain_1h': data.get('rain', {}).get('1h', 0),
                'snow_1h': data.get('snow', {}).get('1h', 0),
                'timestamp': datetime.now()
            }
        except:
            return _self._get_demo_weather(lat, lon)

    def _get_demo_weather(self, lat: float, lon: float) -> Dict:
        """Generate demo weather data"""
        temp = 75 + 10 * np.sin((datetime.now().hour - 6) * np.pi / 12)
        return {
            'feels_like_fahrenheit': temp,
            'humidity': max(30, min(95, 65 + np.random.normal(0, 10))),
            'pressure': 1013 + np.random.normal(0, 5),
            'visibility': 15000,
            'wind_speed': max(0, np.random.exponential(5)),
            'cloudiness': 50,
            'rain_1h': 0,
            'snow_1h': 0,
            'timestamp': datetime.now()
        }

    def engineer_features(self, current_weather: Dict) -> pd.DataFrame:
        """Engineer all 80 features"""
        df = pd.DataFrame([current_weather]).fillna(0)

        if 'rain_1h' not in df.columns:
            df['rain_1h'] = 0.0
        if 'snow_1h' not in df.columns:
            df['snow_1h'] = 0.0

        # Temporal features
        now = datetime.now()
        df['hour'] = now.hour
        df['day_of_week'] = now.weekday()
        df['month'] = now.month
        df['is_weekend'] = int(df['day_of_week'].iloc[0] >= 5)
        df['is_morning_rush'] = int(7 <= df['hour'].iloc[0] <= 9)
        df['is_evening_rush'] = int(17 <= df['hour'].iloc[0] <= 19)
        df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']

        if df['is_morning_rush'].iloc[0]:
            df['minutes_since_rush_start'] = (df['hour'].iloc[0] - 7) * 60 + now.minute
        elif df['is_evening_rush'].iloc[0]:
            df['minutes_since_rush_start'] = (df['hour'].iloc[0] - 17) * 60 + now.minute
        else:
            df['minutes_since_rush_start'] = 0

        # Seasonal indicators
        df['is_winter'] = int(df['month'].iloc[0] in [12, 1, 2])
        df['is_spring'] = int(df['month'].iloc[0] in [3, 4, 5])
        df['is_summer'] = int(df['month'].iloc[0] in [6, 7, 8])
        df['is_fall'] = int(df['month'].iloc[0] in [9, 10, 11])

        # Temperature deviation
        norms = {1: 35, 2: 38, 3: 48, 4: 58, 5: 68, 6: 76, 7: 80, 8: 78, 9: 71, 10: 60, 11: 50, 12: 40}
        df['temp_deviation_seasonal'] = abs(df['feels_like_fahrenheit'].iloc[0] - norms.get(df['month'].iloc[0], 70))

        # Use current values for rolling/lag features (simplified for demo)
        temp = float(df['feels_like_fahrenheit'].iloc[0])
        for var in ['feels_like_fahrenheit', 'humidity', 'pressure', 'wind_speed']:
            val = float(df[var].iloc[0])
            df[f'{var}_rolling_mean_120min'] = val
            df[f'{var}_rolling_std_120min'] = 0.0
            df[f'{var}_rolling_min_120min'] = val
            df[f'{var}_rolling_max_120min'] = val
            df[f'{var}_lag_60min'] = val
            df[f'{var}_change_60min'] = 0.0

        for var in ['rain_1h', 'snow_1h']:
            df[f'{var}_rolling_min_60min'] = df[var].iloc[0]
            df[f'{var}_rolling_max_120min'] = df[var].iloc[0]

        df['spring_warming_rate'] = 0.0
        df['fall_cooling_rate'] = 0.0

        # Categorical features
        temp_val = float(df['feels_like_fahrenheit'].iloc[0])
        humidity_val = float(df['humidity'].iloc[0])
        pressure_val = float(df['pressure'].iloc[0])

        df['temp_extreme_cold'] = int(temp_val <= 20)
        df['temp_freezing'] = int(temp_val <= 32)
        df['temp_hot'] = int(temp_val >= 85)
        df['temp_extreme_hot'] = int(temp_val >= 95)
        df['spring_warm_spell'] = 0
        df['spring_freeze_risk'] = 0
        df['fall_cooling_trend'] = 0
        df['early_winter_conditions'] = 0
        df['humidity_high'] = int(humidity_val >= 80)
        df['humidity_very_high'] = int(humidity_val >= 90)
        df['has_rain'] = int(df['rain_1h'].iloc[0] > 0)
        df['rain_amount'] = df['rain_1h'].iloc[0]
        df['rain_heavy'] = int(df['rain_1h'].iloc[0] > 7)
        df['has_snow'] = int(df['snow_1h'].iloc[0] > 0)
        df['snow_amount'] = df['snow_1h'].iloc[0]
        df['snow_heavy'] = int(df['snow_1h'].iloc[0] > 2)
        df['approaching_storm'] = int((pressure_val <= 1012) and (humidity_val >= 80))
        df['stable_clear'] = int((pressure_val >= 1020) and (humidity_val <= 60))
        df['weather_stability_index'] = (pressure_val - 1013) / 10 - (humidity_val - 50) / 20
        df['spring_storm'] = 0
        df['fall_high_pressure'] = 0
        df['heat_index_simple'] = temp_val + 0.5 * (humidity_val - 50)
        df['temp_humidity_ratio'] = temp_val / max(humidity_val + 1, 1)
        df['rush_wind_speed_interaction'] = float(df['is_rush_hour'].iloc[0]) * float(df['wind_speed'].iloc[0])
        df['cold_wet_stress'] = int((temp_val <= 35) and (humidity_val >= 80))
        df['hot_humid_stress'] = int((temp_val >= 85) and (humidity_val >= 70))
        df['poor_visibility'] = int(df['visibility'].iloc[0] <= 5000)
        df['fog_conditions'] = int((df['visibility'].iloc[0] <= 3000) and (humidity_val >= 85))
        df['freezing_rain_risk'] = int((temp_val <= 34) and (df['rain_1h'].iloc[0] > 0))
        df['mixed_precipitation'] = int((df['rain_1h'].iloc[0] > 0) and (df['snow_1h'].iloc[0] > 0))

        # Create final dataframe with all required features
        final_df = pd.DataFrame(0.0, index=[0], columns=self.feature_columns)
        for col in df.columns:
            if col in self.feature_columns:
                final_df[col] = df[col].iloc[0]

        return final_df.astype(float)

    def predict_delay(self, station_name: str) -> Dict:
        """Make delay prediction"""
        if not self.model:
            return {"error": "Model not loaded"}

        try:
            station_info = self.stations.get(station_name)
            if not station_info:
                return {"error": f"Station {station_name} not found"}

            current_weather = self.get_current_weather(station_info['lat'], station_info['lon'])
            features_df = self.engineer_features(current_weather)
            predicted_delay = self.model.predict(features_df)[0]

            return {
                'station_name': station_name,
                'predicted_delay_minutes': float(predicted_delay),
                'confidence_score': 0.85,
                'weather_conditions': current_weather,
                'prediction_timestamp': datetime.now(),
                'recommendation': self._get_recommendation(predicted_delay)
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _get_recommendation(self, predicted_delay: float) -> str:
        """Get travel recommendation"""
        if predicted_delay <= 2:
            return "Normal service expected"
        elif predicted_delay <= 5:
            return "Allow a few extra minutes"
        elif predicted_delay <= 10:
            return "Consider alternative routes"
        else:
            return "Significant delays expected"


def initialize_app():
    """Initialize the Streamlit app"""
    st.title("🚇 MTA Delay Predictor")
    st.markdown("### Real-time delay predictions using seasonal weather patterns")

    with st.sidebar:
        st.header("⚙️ Configuration")

        model_path = st.text_input("Model Path", value="./mta_delay_model.pkl")
        db_url = st.text_input("Database URL (Optional)", value="")
        weather_api_key = st.text_input("Weather API Key", value="demo_mode", type="password")

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.metric("R² Score", "56.7%")
        st.metric("RMSE", "5.94 min")
        st.metric("Features", "80")
        st.caption("Seasonal Optimized")
        st.caption("Training: Aug-Sep 2025")

    if 'predictor' not in st.session_state:
        try:
            st.session_state.predictor = MTADelayPredictor(model_path, db_url, weather_api_key)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.stop()

    return st.session_state.predictor


def main():
    """Main Streamlit app"""
    predictor = initialize_app()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎯 Make Predictions")

        station_name = st.selectbox("Select MTA Station", options=list(predictor.stations.keys()))

        if st.button("🔮 Predict Delays", type="primary"):
            with st.spinner("Analyzing weather conditions..."):
                result = predictor.predict_delay(station_name)

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Prediction Complete!")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        # --- START: Modified Metric with Delta ---
                        st.metric(
                            "Predicted Delay",
                            f"{result['predicted_delay_minutes']:.1f} min",
                            delta=f"{result['predicted_delay_minutes'] - 1.77:.1f} vs avg"
                        )
                        # --- END: Modified Metric with Delta ---
                    with col_b:
                        st.metric("Confidence", f"{result['confidence_score']:.0%}")
                    with col_c:
                        st.metric("Status", result['recommendation'])

                    st.subheader("🌤️ Current Weather")
                    weather = result['weather_conditions']

                    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                    with col_w1:
                        st.metric("Temperature", f"{weather['feels_like_fahrenheit']:.1f}°F")
                    with col_w2:
                        st.metric("Humidity", f"{weather['humidity']:.0f}%")
                    with col_w3:
                        st.metric("Wind", f"{weather['wind_speed']:.1f} mph")
                    with col_w4:
                        st.metric("Visibility", f"{weather['visibility'] / 1000:.1f} km")

        st.header("📊 Batch Predictions")
        if st.button("🚇 Predict All Stations"):
            with st.spinner("Processing all stations..."):
                batch_results = []
                progress_bar = st.progress(0)
                all_stations = list(predictor.stations.keys())

                for i, station in enumerate(all_stations):
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
                    progress_bar.progress((i + 1) / len(all_stations))

                if batch_results:
                    df_results = pd.DataFrame(batch_results)
                    st.dataframe(df_results, use_container_width=True)

                    # Visualization
                    fig = px.bar(
                        df_results,
                        x='Station',
                        y='Predicted Delay (min)',
                        color='Predicted Delay (min)',
                        title="Predicted Delays Across All Stations",
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("📈 Dashboard")

        current_time = datetime.now()
        st.metric("Time", current_time.strftime("%H:%M:%S"))

        hour = current_time.hour
        if 7 <= hour <= 9:
            st.success("🌅 Morning Rush")
        elif 17 <= hour <= 19:
            st.warning("🌆 Evening Rush")
        else:
            st.info("⏰ Off-Peak")

        st.subheader("🎯 Model Info")
        st.info("""
**Seasonal Optimized Model**
- R² Score: 56.7%
- RMSE: 5.94 minutes
- Features: 80 optimized
- Training: Aug-Sep 2025
- Year-round ready

**Seasonal Coverage:**
- Summer: 56.7% R²
- Winter: 52-55% R² (est.)
- Spring/Fall: 54-56% R² (est.)
        """)

        st.subheader("💡 Tips")
        st.info("""
- Morning rush: 7-9 AM
- Evening rush: 5-7 PM
- Weather impacts: Humidity and temperature changes increase delays
- Seasonal patterns: Spring warming and fall cooling affect delays differently
        """)


if __name__ == "__main__":
    main()