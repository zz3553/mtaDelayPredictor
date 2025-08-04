import json
import csv
import requests
import zipfile
import io
from datetime import datetime
from sqlalchemy import create_engine, text
from nyct_gtfs import NYCTFeed
from nyct_gtfs.gtfs_static_types import Stations

# Database configuration
DB_USER = "neondb_owner"
DB_PASSWORD = "npg_J73HnAiwErpq"
DB_HOST = "ep-spring-truth-ae312q45-pooler.c-2.us-east-2.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "neondb"


def lambda_handler(event, context):
    """
    AWS Lambda handler for MTA delay tracking
    """
    try:
        print("Starting MTA delay tracking...")

        # Step 1: Get realtime data from MTA
        print("Fetching realtime MTA data...")
        feed = NYCTFeed("N")
        trains = feed.filter_trips(line_id=['N'], underway=True)

        # Filter by direction
        northbound_trains = [train for train in trains if train.direction == 'N']
        southbound_trains = [train for train in trains if train.direction == 'S']

        if not northbound_trains and not southbound_trains:
            print("No trains currently underway")
            return {
                'statusCode': 200,
                'body': json.dumps('No trains currently underway')
            }

        print(f"Found {len(northbound_trains)} northbound and {len(southbound_trains)} southbound trains")

        # Step 2: Download GTFS static data
        print("Downloading GTFS static data...")
        url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Extract stop_times.txt
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            stop_times_file_path = None
            for file_path in zip_file.namelist():
                if file_path.endswith('stop_times.txt'):
                    stop_times_file_path = file_path
                    break

            if stop_times_file_path is None:
                raise FileNotFoundError("stop_times.txt not found in the archive")

            with zip_file.open(stop_times_file_path) as stop_times_file:
                # Read CSV without pandas - use built-in csv module
                stop_times_data = []
                csv_content = stop_times_file.read().decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(csv_content))
                for row in csv_reader:
                    stop_times_data.append(row)

        print(f"Loaded {len(stop_times_data):,} stop times records")

        # Step 3: Process delay calculations
        stations = Stations()

        def process_trains(train_list, direction_label):
            results = []

            for train in train_list:
                trip_id = train.trip_id
                current_time = train.last_position_update
                stop_id = train.location

                if stop_id and trip_id:
                    # Find matching schedule without pandas
                    matching_schedule = None
                    for row in stop_times_data:
                        if trip_id in row.get('trip_id', '') and row.get('stop_id') == stop_id:
                            matching_schedule = row
                            break

                    if matching_schedule:
                        try:
                            scheduled_time = datetime.strptime(matching_schedule["arrival_time"], "%H:%M:%S").time()
                            scheduled_dt = datetime.combine(current_time.date(), scheduled_time)
                            delay = (current_time - scheduled_dt).total_seconds() / 60.0

                            results.append({
                                "trip_id": trip_id,
                                "stop_id_raw": stop_id,
                                "stop_id": stations.get_station_name(stop_id),
                                "timestamp": current_time,
                                "delay_min": round(delay, 2),
                                "status": "on_time" if abs(delay) < 1 else "delayed" if delay > 0 else "early",
                                "direction": direction_label,
                                "stop_sequence": matching_schedule.get("stop_sequence")
                            })
                        except Exception as e:
                            print(f"Error parsing time for {trip_id} at {stop_id}: {e}")

            return results

        # Process both directions
        northbound_results = process_trains(northbound_trains, "N")
        southbound_results = process_trains(southbound_trains, "S")

        # Combine results
        all_results = northbound_results + southbound_results

        if not all_results:
            print("No delay data to process")
            return {
                'statusCode': 200,
                'body': json.dumps('No delay data found')
            }

        # Add rush hour classification
        def classify_rush_hour(ts):
            hour = ts.hour
            if 7 <= hour < 10:
                return "morning"
            elif 16 <= hour < 19:
                return "evening"
            else:
                return None

        for result in all_results:
            result["rush_hour"] = classify_rush_hour(result["timestamp"])

        # Step 4: Save to database
        print("Connecting to database...")
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

        # Create table if not exists
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
            conn.commit()

        # Insert data
        with engine.connect() as conn:
            conn.execute(text(create_train_delays))
            conn.commit()

            # Insert each record individually (without pandas to_sql)
            insert_sql = """
                INSERT INTO train_delays (trip_id, stop_id, station_name, timestamp, delay_min, status, direction, stop_sequence, rush_hour)
                VALUES (:trip_id, :stop_id, :station_name, :timestamp, :delay_min, :status, :direction, :stop_sequence, :rush_hour)
            """

            for result in all_results:
                conn.execute(text(insert_sql), {
                    'trip_id': result['trip_id'],
                    'stop_id': result['stop_id_raw'],
                    'station_name': result['stop_id'],
                    'timestamp': result['timestamp'],
                    'delay_min': result['delay_min'],
                    'status': result['status'],
                    'direction': result['direction'],
                    'stop_sequence': result['stop_sequence'],
                    'rush_hour': result['rush_hour']
                })
            conn.commit()

        records_inserted = len(all_results)
        print(f"Successfully inserted {records_inserted} delay records")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'MTA delay tracking completed successfully',
                'records_inserted': records_inserted,
                'timestamp': datetime.now().isoformat()
            })
        }

    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }