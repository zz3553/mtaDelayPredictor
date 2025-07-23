#!/usr/bin/env python
# coding: utf-8

# In[34]:


"""
Getting realtime data feed from the MTA and separating based on N/S Bound trains
"""

from nyct_gtfs import NYCTFeed

feed = NYCTFeed("N")
trains = feed.filter_trips(line_id=['N'], underway=True)

# filtering based on n/s bound trains
northbound_trains = [train for train in trains if train.direction == 'N']
southbound_trains = [train for train in trains if train.direction == 'S']

# gathering all the individual trips
n_bound_trip_ids = [trip.trip_id for trip in northbound_trains]
s_bound_trip_ids = [trip.trip_id for trip in southbound_trains]

n_bound_trip_ids


# In[35]:
"""
Creating data frame to hold scheduled times for each trip
Downloads GTFS data from https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip
"""
import pandas as pd
import re
import requests
import zipfile
import io
import os
from datetime import datetime

# Define the absolute path to your project directory
PROJECT_DIR = "/Users/mitchel/Desktop/beep/mtaDelayPredictor"
DATA_DIR = os.path.join(PROJECT_DIR, "data_files")

# Create data_files directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Download and extract the GTFS data
print("Downloading GTFS data...")
url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"
response = requests.get(url)
response.raise_for_status()  # Raise an exception for bad status codes

# Extract stop_times.txt from the zip file
print("Extracting stop_times.txt...")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
    # First, let's see what files are in the archive
    file_list = zip_file.namelist()
    # print("Files in archive:", file_list)

    # Find the stop_times.txt file (it might be in a different path)
    stop_times_file_path = None
    for file_path in file_list:
        if file_path.endswith('stop_times.txt'):
            stop_times_file_path = file_path
            break

    if stop_times_file_path is None:
        raise FileNotFoundError("stop_times.txt not found in the archive")

    # print(f"Found stop_times.txt at: {stop_times_file_path}")

    with zip_file.open(stop_times_file_path) as stop_times_file:
        # Read the content and create a DataFrame
        stop_times = pd.read_csv(stop_times_file)

# Generate filename with current datetime
current_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M")
filename = os.path.join(DATA_DIR, f'stop_times_{current_datetime}.csv')

# Get the absolute path for clarity
abs_filename = os.path.abspath(filename)

# Save to CSV with datetime appended
print(f"Saving to {abs_filename}...")
stop_times.to_csv(filename, index=False)

# Verify the file was created
if os.path.exists(filename):
    file_size = os.path.getsize(filename)
    print(f"Successfully saved stop_times data to {abs_filename}")
    print(f"File size: {file_size:,} bytes")
    print(f"Number of rows: {len(stop_times):,}")
else:
    print(f"ERROR: File was not created at {abs_filename}")
    print(f"Current working directory: {os.getcwd()}")
    print(
        f"Contents of data_files directory: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'Directory does not exist'}")

if n_bound_trip_ids or s_bound_trip_ids:
    trip_dfs = {}
    all_trip_ids = n_bound_trip_ids + s_bound_trip_ids

    for trip in all_trip_ids:
        escaped_trip = re.escape(trip)
        filtered = stop_times[stop_times['trip_id'].str.contains(escaped_trip, na=False)]
        trip_dfs[trip] = filtered

    print(f"Processed {len(trip_dfs)} trip DataFrames")
else:
    print("No trip IDs defined - skipping trip processing")
# In[36]:


"""
Combining static and real time schedule to compute delay information
"""
import pandas as pd
from nyct_gtfs.gtfs_static_types import Stations

stations = Stations()


# === Shared delay calculation logic ===
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
                        "delay_min": round(delay,2),
                        "status": "on_time" if abs(delay) < 1 else "delayed" if delay > 0 else "early",
                        "direction": direction_label
                    })
                except Exception as e:
                    print(f"Error parsing time for {trip_id} at {stop_id}: {e}")

    df = pd.DataFrame(results)

    # Add stop_sequence by fuzzy match
    def get_stop_sequence(row):
        match = stop_times[
            stop_times.trip_id.str.contains(row["trip_id"], na=False) &
            (stop_times.stop_id == row["stop_id_raw"])
        ]
        if not match.empty:
            return match.iloc[0]["stop_sequence"]
        else:
            return None

    df["stop_sequence"] = df.apply(get_stop_sequence, axis=1)
    df = df.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

    return df


# === Run for both directions ===
northbound_df = process_trains(northbound_trains, direction_label="N")
southbound_df = process_trains(southbound_trains, direction_label="S")

# === Combine for full view ===
all_delays_df = pd.concat([northbound_df, southbound_df]).sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

# === Rush hour logic ===
def classify_rush_hour(ts):
    hour = ts.hour
    if 7 <= hour < 10:
        return "morning"
    elif 16 <= hour < 19:
        return "evening"
    else:
        return None
    
# === Log ===
all_delays_df["rush_hour"] = all_delays_df["timestamp"].apply(classify_rush_hour)

all_delays_df


# In[37]:


import datetime
import pandas as pd
from sqlalchemy import create_engine, text

# === Database config ===
DB_USER = "postgres"
DB_PASSWORD = "commiteveryday"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "train_delays"

# === Create DB engine ===
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# === Create tables ===
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
    print("Table created.")

# === Clean and insert delay data ===
if not all_delays_df.empty:
    delay_cols = ['trip_id', 'stop_id_raw', 'stop_id', 'timestamp', 'delay_min', 'status', 'direction', 'stop_sequence', 'rush_hour']
    df = all_delays_df[delay_cols].copy()
    df.columns = ['trip_id', 'stop_id', 'station_name', 'timestamp', 'delay_min', 'status', 'direction', 'stop_sequence', 'rush_hour']
    df.to_sql('train_delays', engine, if_exists='append', index=False)
    print(f"Inserted {len(df)} delay records.")
else:
    print("No delay data found.")

