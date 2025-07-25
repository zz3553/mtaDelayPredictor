#!/usr/bin/env python
# coding: utf-8

"""
Getting realtime data feed from the MTA and separating based on N/S Bound trains
"""

from nyct_gtfs import NYCTFeed
import pandas as pd
import re
import requests
import zipfile
import io
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from nyct_gtfs.gtfs_static_types import Stations
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mta_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    # === Get current working directory (GitHub Actions will be in the repo root) ===
    PROJECT_DIR = os.getcwd()
    DATA_DIR = os.path.join(PROJECT_DIR, "data_files")

    # Create data_files directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.info(f"Working directory: {PROJECT_DIR}")
    logging.info(f"Data directory: {DATA_DIR}")

    # === Get MTA feed data ===
    logging.info("Getting MTA feed data...")
    feed = NYCTFeed("N")
    trains = feed.filter_trips(line_id=['N'], underway=True)

    # filtering based on n/s bound trains
    northbound_trains = [train for train in trains if train.direction == 'N']
    southbound_trains = [train for train in trains if train.direction == 'S']

    # gathering all the individual trips
    n_bound_trip_ids = [trip.trip_id for trip in northbound_trains]
    s_bound_trip_ids = [trip.trip_id for trip in southbound_trains]

    logging.info(f"Found {len(n_bound_trip_ids)} northbound and {len(s_bound_trip_ids)} southbound trips")

    # === Download GTFS data ===
    logging.info("Downloading GTFS data...")
    url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"
    response = requests.get(url)
    response.raise_for_status()

    # Extract stop_times.txt from the zip file
    logging.info("Extracting stop_times.txt...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        file_list = zip_file.namelist()

        stop_times_file_path = None
        for file_path in file_list:
            if file_path.endswith('stop_times.txt'):
                stop_times_file_path = file_path
                break

        if stop_times_file_path is None:
            raise FileNotFoundError("stop_times.txt not found in the archive")

        with zip_file.open(stop_times_file_path) as stop_times_file:
            stop_times = pd.read_csv(stop_times_file)

    # Generate filename with current datetime
    current_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M")
    filename = os.path.join(DATA_DIR, f'stop_times_{current_datetime}.csv')

    # Save to CSV
    logging.info(f"Saving stop_times data to {filename}")
    stop_times.to_csv(filename, index=False)

    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        logging.info(f"Successfully saved stop_times data. File size: {file_size:,} bytes, Rows: {len(stop_times):,}")
    else:
        logging.error(f"Failed to create file at {filename}")

    # Process trip data
    if n_bound_trip_ids or s_bound_trip_ids:
        trip_dfs = {}
        all_trip_ids = n_bound_trip_ids + s_bound_trip_ids

        for trip in all_trip_ids:
            escaped_trip = re.escape(trip)
            filtered = stop_times[stop_times['trip_id'].str.contains(escaped_trip, na=False)]
            trip_dfs[trip] = filtered

        logging.info(f"Processed {len(trip_dfs)} trip DataFrames")
    else:
        logging.warning("No trip IDs found - skipping trip processing")

    # === Process delay calculations ===
    stations = Stations()


    def process_trains(train_list, direction_label):
        results = []

        for train in train_list:
            trip_id = train.trip_id
            current_time = train.last_position_update
            stop_id = train.location

            if stop_id and trip_id:
                sched = stop_times[
                    stop_times.trip_id.str.contains(trip_id, na=False) &
                    (stop_times.stop_id == stop_id)
                    ][["arrival_time", "departure_time"]]

                if not sched.empty:
                    try:
                        scheduled_time = datetime.strptime(sched["arrival_time"].iloc[0], "%H:%M:%S").time()
                        scheduled_dt = pd.Timestamp.combine(current_time.date(), scheduled_time)
                        delay = (current_time - scheduled_dt).total_seconds() / 60.0

                        results.append({
                            "trip_id": trip_id,
                            "stop_id_raw": stop_id,
                            "stop_id": stations.get_station_name(stop_id),
                            "timestamp": current_time,
                            "delay_min": round(delay, 2),
                            "status": "on_time" if abs(delay) < 1 else "delayed" if delay > 0 else "early",
                            "direction": direction_label
                        })
                    except Exception as e:
                        logging.error(f"Error parsing time for {trip_id} at {stop_id}: {e}")

        df = pd.DataFrame(results)

        if not df.empty:
            # Add stop_sequence by fuzzy match
            def get_stop_sequence(row):
                match = stop_times[
                    stop_times.trip_id.str.contains(row["trip_id"], na=False) &
                    (stop_times.stop_id == row["stop_id_raw"])
                    ]
                return match.iloc[0]["stop_sequence"] if not match.empty else None

            df["stop_sequence"] = df.apply(get_stop_sequence, axis=1)
            df = df.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

        return df


    # Process both directions
    northbound_df = process_trains(northbound_trains, direction_label="N")
    southbound_df = process_trains(southbound_trains, direction_label="S")

    # Combine dataframes
    all_delays_df = pd.concat([northbound_df, southbound_df]).sort_values(by=["trip_id", "stop_sequence"]).reset_index(
        drop=True)


    # Rush hour classification
    def classify_rush_hour(ts):
        hour = ts.hour
        if 7 <= hour < 10:
            return "morning"
        elif 16 <= hour < 19:
            return "evening"
        else:
            return None


    all_delays_df["rush_hour"] = all_delays_df["timestamp"].apply(classify_rush_hour)

    logging.info(f"Processed {len(all_delays_df)} delay records")

    # === Database operations ===
    # Get environment variables for database connection
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise ValueError("Missing required database environment variables")

    # Create database connection
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    # Create table
    create_train_delays = """
    CREATE TABLE IF NOT EXISTS train_delays (
        id SERIAL PRIMARY KEY,
        trip_id TEXT,
        stop_id TEXT,
        station_name TEXT,
        timestamp TIMESTAMP,
        delay_min REAL,
        status TEXT,
        direction TEXT,
        stop_sequence INTEGER,
        rush_hour TEXT
    );
    """

    with engine.connect() as conn:
        conn.execute(text(create_train_delays))
        logging.info("Database table created/verified")

    # Insert data
    if not all_delays_df.empty:
        delay_cols = ['trip_id', 'stop_id_raw', 'stop_id', 'timestamp', 'delay_min', 'status', 'direction',
                      'stop_sequence', 'rush_hour']
        df = all_delays_df[delay_cols].copy()
        df.columns = ['trip_id', 'stop_id', 'station_name', 'timestamp', 'delay_min', 'status', 'direction',
                      'stop_sequence', 'rush_hour']
        df.to_sql('train_delays', engine, if_exists='append', index=False)
        logging.info(f"Successfully inserted {len(df)} delay records into database")
    else:
        logging.warning("No delay data to insert into database")

    logging.info("MTA data collection completed successfully")

except Exception as e:
    logging.error(f"MTA data collection failed: {str(e)}")
    sys.exit(1)