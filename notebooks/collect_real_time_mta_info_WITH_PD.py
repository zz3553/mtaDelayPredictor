#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

n_bound_trip_ids + s_bound_trip_ids


# In[6]:


"""
Creating data frame to hold scheduled times for each trip
Downloads GTFS data from https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip
"""
import pandas as pd
import re
import requests
import zipfile
import io

# Download and extract the GTFS data
print("Downloading GTFS data...")
url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"
response = requests.get(url)
response.raise_for_status()  # Raise an exception for bad status codes

# Extract stop_times.txt from the zip file
print("Extracting stop_times.txt...")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
    # Find the stop_times.txt file (it might be in a different path)
    stop_times_file_path = None
    for file_path in zip_file.namelist():
        if file_path.endswith('stop_times.txt'):
            stop_times_file_path = file_path
            break

    if stop_times_file_path is None:
        raise FileNotFoundError("stop_times.txt not found in the archive")

    with zip_file.open(stop_times_file_path) as stop_times_file:
        # Read the content and create a DataFrame
        stop_times = pd.read_csv(stop_times_file)

print(f"Successfully loaded stop_times data with {len(stop_times):,} rows")

# Process trip data if trip IDs are defined
trip_dfs = {}
if 'n_bound_trip_ids' in globals() and 's_bound_trip_ids' in globals():
    all_trip_ids = n_bound_trip_ids + s_bound_trip_ids

    for trip in all_trip_ids:
        escaped_trip = re.escape(trip)
        filtered = stop_times[stop_times['trip_id'].str.contains(escaped_trip, na=False)]
        trip_dfs[trip] = filtered

    print(f"Processed {len(trip_dfs)} trip DataFrames")
else:
    print("Trip IDs not defined - skipping trip processing. Define n_bound_trip_ids and s_bound_trip_ids variables.")

trip_dfs


# In[ ]:


"""
Combining static and real time schedule to compute delay information
"""
import pandas as pd
from nyct_gtfs.gtfs_static_types import Stations
from datetime import datetime

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


# In[ ]:


import datetime
import pandas as pd
from sqlalchemy import create_engine, text

# === Database config ===
DB_USER = "neondb_owner"
DB_PASSWORD = ""  # Replace with the password you revealed
DB_HOST = "ep-spring-truth-ae312q45-pooler.c-2.us-east-2.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "neondb"


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


# In[ ]:
#
#
# """
# Build Hourly Features + Labels from Postgres
# """
# import pandas as pd
#
# # === Get hourly aggregated delay data ===
# delay_sql = """
#     SELECT
#         DATE_TRUNC('hour', timestamp) AS hour,
#         COUNT(*) FILTER (WHERE delay_min > 1) * 1.0 / COUNT(*) AS delay_rate,
#         COUNT(*) AS total_trips,
#         MAX(rush_hour) AS rush_hour  -- could be 'morning', 'evening', or NULL
#     FROM train_delays
#     GROUP BY hour
#     ORDER BY hour
# """
#
# # === Get matching weather data ===
# weather_sql = """
#     SELECT
#         time AS hour,
#         temperature_f,
#         precipitation,
#         snowfall,
#         humidity,
#         windspeed
#     FROM weather_hourly
# """
#
# # === Load from Postgres ===
# df_delays = pd.read_sql(delay_sql, engine)
# df_weather = pd.read_sql(weather_sql, engine)
#
# # === Merge both datasets on hour
# df = pd.merge(df_delays, df_weather, on="hour", how="inner")
#
# # === Add time-based features
# df["hour_of_day"] = df["hour"].dt.hour
# df["day_of_week"] = df["hour"].dt.dayofweek
# df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
#
# # === Drop incomplete rows
# df = df.dropna()
#
# # === Show preview
# print(df)
#
