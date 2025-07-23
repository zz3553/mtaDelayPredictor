#!/usr/bin/env python
# coding: utf-8

"""
Gathering weather data - runs daily to collect previous day's data
"""
import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# Change to script directory to ensure consistent paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


try:
    log_message("Starting weather data collection...")

    # NYC coordinates
    lat, lon = 40.7128, -74.0060

    # Get date for 2 days ago (to ensure data is available)
    delay_date = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
    start_date = delay_date
    end_date = delay_date  # Same day, hourly granularity

    log_message(f"Collecting weather data for {delay_date}")

    # Open-Meteo API with temperature in Fahrenheit
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,snowfall,relative_humidity_2m,windspeed_10m"
        f"&temperature_unit=fahrenheit"
        f"&timezone=America/New_York"
    )

    # Fetch and convert to DataFrame
    log_message("Fetching weather data from API...")
    response = requests.get(weather_url)
    response.raise_for_status()  # Raise exception for HTTP errors

    weather_data = response.json()
    weather_df = pd.DataFrame(weather_data['hourly'])
    weather_df['time'] = pd.to_datetime(weather_df['time'])

    log_message(f"Retrieved {len(weather_df)} weather records")

    # === Database config ===
    DB_USER = "postgres"
    DB_PASSWORD = "commiteveryday"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "train_delays"

    # === Create DB engine ===
    log_message("Connecting to database...")
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    create_weather_hourly = """
    CREATE TABLE IF NOT EXISTS weather_hourly (
        id SERIAL PRIMARY KEY,
        time TIMESTAMP,
        temperature_f REAL,
        precipitation REAL,
        snowfall REAL,
        humidity REAL,
        windspeed REAL
    );
    """

    with engine.connect() as conn:
        conn.execute(text(create_weather_hourly))
        log_message("Weather table created/verified.")

    # === Insert weather data ===
    if not weather_df.empty:
        weather_cols = ['time', 'temperature_2m', 'precipitation', 'snowfall', 'relative_humidity_2m', 'windspeed_10m']

        df_weather = weather_df[weather_cols].copy()
        df_weather.columns = ['time', 'temperature_f', 'precipitation', 'snowfall', 'humidity', 'windspeed']

        df_weather.to_sql('weather_hourly', engine, if_exists='append', index=False)
        log_message(f"Successfully inserted {len(df_weather)} weather records into database.")
    else:
        log_message("WARNING: No weather data found.")

    log_message("Weather data collection completed successfully.")

except Exception as e:
    log_message(f"ERROR: Weather data collection failed: {str(e)}")
    sys.exit(1)