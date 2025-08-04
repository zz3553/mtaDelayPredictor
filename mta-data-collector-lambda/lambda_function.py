import json
import csv

import boto3
import urllib3  # Replace requests with urllib3
import zipfile
import re
import io
import logging
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict
from typing import Dict, List, Optional

from botocore.exceptions import NoCredentialsError, ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from zoneinfo import ZoneInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_neon_db_credentials_from_aws_sm():
    """Retrieve database credentials from AWS Secrets Manager"""
    secret_name = "neon_db_credentials"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        logger.info(f"Retrieving secret: {secret_name}")
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        logger.info("Secret retrieved successfully")

        # Parse the JSON secret
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret

    except ClientError as e:
        # Handle specific AWS errors
        error_code = e.response['Error']['Code']
        logger.error(f"AWS Secrets Manager error ({error_code}): {str(e)}")

        if error_code == 'DecryptionFailureException':
            raise e
        elif error_code == 'InternalServiceErrorException':
            raise e
        elif error_code == 'InvalidParameterException':
            raise e
        elif error_code == 'InvalidRequestException':
            raise e
        elif error_code == 'ResourceNotFoundException':
            logger.error(f"Secret {secret_name} not found")
            raise e
        else:
            raise e

    except NoCredentialsError:
        logger.error("AWS credentials not found")
        raise

    except Exception as e:
        logger.error(f"Unexpected error retrieving secret: {str(e)}")
        raise

class SubwayDelayTracker:
    def __init__(self):
        self.stop_times_data = []
        self.stop_times_index = defaultdict(list)  # Index by trip_id for faster lookups
        self.trip_stop_index = defaultdict(dict)  # Index by trip_id -> stop_id for O(1) lookups
        self.db_engine = None
        self.http = urllib3.PoolManager()  # Initialize urllib3 pool manager

    def initialize_database(self, db_config: Dict[str, str] = None):
        """Initialize database connection with configuration"""
        try:
            if not db_config:
                aws_sm_response = get_neon_db_credentials_from_aws_sm()
                db_config = {
                    'user': aws_sm_response['username'],
                    'password': aws_sm_response['password'],
                    'host': aws_sm_response['host'],
                    'port': aws_sm_response['port'],
                    'database': aws_sm_response['dbname']
                }

            connection_string = f"postgresql+pg8000://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            self.db_engine = create_engine(connection_string, pool_pre_ping=True, pool_recycle=3600)

            self.log_with_time("Database connection initialized successfully")
            return True

        except Exception as e:
            self.log_with_time(f"Failed to initialize database: {str(e)}", "ERROR")
            return False

    def create_tables(self):
        """Create database tables if they don't exist - original schema only"""
        create_train_delays_sql = """
        CREATE TABLE IF NOT EXISTS train_delays (
            id SERIAL PRIMARY KEY,
            trip_id TEXT,
            stop_id TEXT,
            station_name TEXT,
            timestamp TIMESTAMP,
            delay_min DOUBLE PRECISION,
            status TEXT,
            direction TEXT,
            stop_sequence BIGINT,
            rush_hour TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_train_delays_timestamp ON train_delays(timestamp);
        CREATE INDEX IF NOT EXISTS idx_train_delays_trip_id ON train_delays(trip_id);
        CREATE INDEX IF NOT EXISTS idx_train_delays_direction ON train_delays(direction);
        CREATE INDEX IF NOT EXISTS idx_train_delays_status ON train_delays(status);
        """

        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(create_train_delays_sql))
                conn.commit()

            self.log_with_time("Database tables created/verified successfully")
            return True

        except SQLAlchemyError as e:
            self.log_with_time(f"Database table creation failed: {str(e)}", "ERROR")
            return False

    def insert_delay_data(self, delay_records: List[Dict]) -> int:
        """Insert delay data into database with batch processing"""
        if not delay_records:
            self.log_with_time("No delay records to insert")
            return 0

        if not self.db_engine:
            self.log_with_time("Database not initialized", "ERROR")
            return 0

        try:
            # Prepare data for insertion using original schema only
            insert_sql = """
            INSERT INTO train_delays 
            (trip_id, stop_id, station_name, timestamp, delay_min, status, direction, 
             stop_sequence, rush_hour)
            VALUES 
            (:trip_id, :stop_id_raw, :stop_id, :timestamp, :delay_min, :status, :direction,
             :stop_sequence, :rush_hour)
            """

            # Convert timestamp strings back to datetime objects for database
            processed_records = []
            for record in delay_records:
                processed_record = {
                    'trip_id': record['trip_id'],
                    'stop_id_raw': record['stop_id_raw'],
                    'stop_id': record['stop_id'],
                    'timestamp': datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')) if isinstance(
                        record['timestamp'], str) else record['timestamp'],
                    'delay_min': record['delay_min'],
                    'status': record['status'],
                    'direction': record['direction'],
                    'stop_sequence': record['stop_sequence'],
                    'rush_hour': record['rush_hour']
                }
                processed_records.append(processed_record)

            # Batch insert
            with self.db_engine.begin() as conn:
                result = conn.execute(text(insert_sql), processed_records)
                conn.commit()
                inserted_count = result.rowcount

            self.log_with_time(f"Successfully inserted {inserted_count} delay records into database")
            return inserted_count

        except SQLAlchemyError as e:
            self.log_with_time(f"Database insertion failed: {str(e)}", "ERROR")
            return 0
        except Exception as e:
            self.log_with_time(f"Unexpected error during database insertion: {str(e)}", "ERROR")
            return 0

    def log_with_time(self, message: str, level: str = "INFO"):
        """Enhanced logging with different levels"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        if level == "ERROR":
            logger.error(f"[{timestamp}] {message}")
        elif level == "WARNING":
            logger.warning(f"[{timestamp}] {message}")
        else:
            logger.info(f"[{timestamp}] {message}")

    def download_and_parse_gtfs(self, url: str = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip") -> bool:
        """Download and parse GTFS data with error handling - using urllib3 instead of requests"""
        try:
            self.log_with_time("Downloading GTFS data...")

            # Replace requests with urllib3
            response = self.http.request('GET', url, timeout=30)

            # Check if request was successful
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: Failed to download GTFS data")

            self.log_with_time("Extracting stop_times.txt...")
            with zipfile.ZipFile(io.BytesIO(response.data)) as zip_file:
                stop_times_file_path = None
                for file_path in zip_file.namelist():
                    if file_path.endswith('stop_times.txt'):
                        stop_times_file_path = file_path
                        break

                if not stop_times_file_path:
                    raise FileNotFoundError("stop_times.txt not found in the archive")

                with zip_file.open(stop_times_file_path) as stop_times_file:
                    # Parse CSV manually instead of using pandas
                    content = stop_times_file.read().decode('utf-8')
                    csv_reader = csv.DictReader(io.StringIO(content))

                    self.stop_times_data = []
                    for row in csv_reader:
                        self.stop_times_data.append(row)

            self.log_with_time(f"Successfully loaded stop_times data with {len(self.stop_times_data):,} rows")
            self._build_indexes()
            return True

        except Exception as e:
            self.log_with_time(f"Error downloading/parsing GTFS data: {str(e)}", "ERROR")
            return False

    def _build_indexes(self):
        """Build indexes for faster data lookup"""
        self.log_with_time("Building indexes for faster lookup...")

        for row in self.stop_times_data:
            trip_id = row.get('trip_id', '').strip()
            stop_id = row.get('stop_id', '').strip()

            # Ensure we have valid data
            if not trip_id or not stop_id:
                continue

            # Index by trip_id
            self.stop_times_index[trip_id].append(row)

            # Index by trip_id -> stop_id for O(1) lookups
            self.trip_stop_index[trip_id][stop_id] = row

        self.log_with_time(f"Built indexes for {len(self.stop_times_index)} trips")

    def find_matching_trips(self, trip_ids: List[str]) -> Dict[str, List[Dict]]:
        """Find matching trips using regex with optimized lookup"""
        trip_data = {}

        for trip_id in trip_ids:
            try:
                # First try exact match (faster)
                if trip_id in self.stop_times_index:
                    trip_data[trip_id] = self.stop_times_index[trip_id]
                else:
                    # Fall back to regex matching
                    escaped_trip = re.escape(trip_id)
                    pattern = re.compile(escaped_trip)

                    matching_rows = []
                    for stored_trip_id, rows in self.stop_times_index.items():
                        if pattern.search(stored_trip_id):
                            matching_rows.extend(rows)

                    trip_data[trip_id] = matching_rows

            except Exception as e:
                self.log_with_time(f"Error processing trip {trip_id}: {str(e)}", "ERROR")
                trip_data[trip_id] = []

        return trip_data

    def parse_time_string(self, time_str: str) -> Optional[dt_time]:
        """Parse time string with error handling"""
        try:
            return datetime.strptime(time_str, "%H:%M:%S").time()
        except ValueError:
            try:
                # Handle times like "24:30:00" (next day)
                parts = time_str.split(':')
                hour = int(parts[0]) % 24
                minute = int(parts[1])
                second = int(parts[2])
                return dt_time(hour, minute, second)
            except (ValueError, IndexError):
                return None

    def calculate_delay(self, scheduled_time_str: str, current_time: datetime) -> Optional[float]:
        """Calculate delay with proper timezone handling"""
        try:
            scheduled_time = self.parse_time_string(scheduled_time_str)
            if not scheduled_time:
                return None

            # Ensure current_time is timezone-aware
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=ZoneInfo('UTC'))

            # Convert to Eastern for local time comparison
            local_current = current_time.astimezone(ZoneInfo('US/Eastern'))

            # Create scheduled datetime in Eastern timezone
            scheduled_dt = datetime.combine(local_current.date(), scheduled_time)
            scheduled_dt = scheduled_dt.replace(tzinfo=ZoneInfo('US/Eastern'))

            # Handle day boundary crossings
            if scheduled_dt < local_current - timedelta(hours=12):
                scheduled_dt += timedelta(days=1)
            elif scheduled_dt > local_current + timedelta(hours=12):
                scheduled_dt -= timedelta(days=1)

            delay_seconds = (local_current - scheduled_dt).total_seconds()
            return delay_seconds / 60.0

        except Exception as e:
            logger.error(f"Error calculating delay for {scheduled_time_str}: {str(e)}")
            return None

    def classify_rush_hour(self, timestamp: datetime) -> Optional[str]:
        """Classify rush hour periods"""
        hour = timestamp.hour
        if 7 <= hour < 10:
            return "morning"
        elif 16 <= hour < 19:
            return "evening"
        else:
            return None

    def get_station_name(self, stop_id: str, stations) -> str:
        """Get station name with fallback"""
        try:
            return stations.get_station_name(stop_id) if stations else stop_id
        except Exception:
            return stop_id

    def process_trains(self, train_list: List, direction_label: str, stations=None) -> List[Dict]:
        """Process trains and calculate delays"""
        results = []
        processed_count = 0
        error_count = 0

        for train in train_list:
            try:
                trip_id = getattr(train, 'trip_id', None)
                current_time = getattr(train, 'last_position_update', None)
                stop_id = getattr(train, 'location', None)

                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=ZoneInfo('UTC'))

                    # Convert to Eastern Time
                    current_time = current_time.astimezone(ZoneInfo('US/Eastern'))

                if not all([trip_id, current_time, stop_id]):
                    continue

                # Fast lookup using indexes
                schedule_data = None

                # Try exact match first
                if trip_id in self.trip_stop_index and stop_id in self.trip_stop_index[trip_id]:
                    schedule_data = self.trip_stop_index[trip_id][stop_id]
                else:
                    # Fall back to pattern matching
                    escaped_trip = re.escape(trip_id)
                    pattern = re.compile(escaped_trip)

                    for stored_trip_id, stop_dict in self.trip_stop_index.items():
                        if pattern.search(stored_trip_id) and stop_id in stop_dict:
                            schedule_data = stop_dict[stop_id]
                            break

                if schedule_data:
                    arrival_time = schedule_data.get('arrival_time', '').strip()
                    departure_time = schedule_data.get('departure_time', '').strip()
                    stop_sequence = schedule_data.get('stop_sequence', '0')

                    # Use arrival_time, fallback to departure_time if needed
                    time_to_use = arrival_time if arrival_time else departure_time

                    if time_to_use:
                        delay = self.calculate_delay(time_to_use, current_time)

                        if delay is not None:
                            # Determine status
                            if abs(delay) < 1:
                                status = "on_time"
                            elif delay > 0:
                                status = "delayed"
                            else:
                                status = "early"

                            if delay > 100:
                                print(F"Logging incorrect calculating of delay info. "
                                            F"Current time: {current_time} "
                                            F"Time to use: {time_to_use} "
                                            F"Schedule data: {schedule_data} "
                                            F"Stop ID: {stop_id}"
                                            F"Train from GTFS: {train}")

                            result = {
                                "trip_id": trip_id,
                                "stop_id_raw": stop_id,
                                "stop_id": self.get_station_name(stop_id, stations),
                                "timestamp": current_time.isoformat(),
                                "delay_min": round(delay, 2),
                                "status": status,
                                "direction": direction_label,
                                "stop_sequence": int(stop_sequence) if stop_sequence.isdigit() else 0,
                                "rush_hour": self.classify_rush_hour(current_time)
                            }

                            results.append(result)
                            processed_count += 1

            except Exception as e:
                error_count += 1
                self.log_with_time(f"Error processing train {getattr(train, 'trip_id', 'unknown')}: {str(e)}", "ERROR")

        self.log_with_time(f"Processed {processed_count} trains, {error_count} errors for direction {direction_label}")

        # Sort results by trip_id and stop_sequence
        results.sort(key=lambda x: (x['trip_id'], x['stop_sequence']))

        return results


def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Initialize tracker
        tracker = SubwayDelayTracker()

        # Initialize database
        if not tracker.initialize_database():
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to initialize database connection'})
            }

        # Create tables
        if not tracker.create_tables():
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to create database tables'})
            }

        # Import NYCTFeed (assuming this is available in Lambda layer)
        try:
            from nyct_gtfs import NYCTFeed
            from nyct_gtfs.gtfs_static_types import Stations
        except ImportError as e:
            logger.error(f"Failed to import NYCT GTFS modules: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Missing required NYCT GTFS modules'})
            }

        # Download and parse GTFS data
        if not tracker.download_and_parse_gtfs():
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to download GTFS data'})
            }

        # Get real-time feed data
        line_id = event.get('line_id', ['N'])
        tracker.log_with_time(f"Processing line: {line_id}")

        feed = NYCTFeed("N")
        trains = feed.filter_trips(line_id=line_id, underway=True)

        # Filter by direction
        northbound_trains = [train for train in trains if getattr(train, 'direction', None) == 'N']
        southbound_trains = [train for train in trains if getattr(train, 'direction', None) == 'S']

        tracker.log_with_time(
            f"Found {len(northbound_trains)} northbound and {len(southbound_trains)} southbound trains")

        # Initialize stations
        try:
            stations = Stations()
        except Exception as e:
            tracker.log_with_time(f"Warning: Could not initialize stations: {str(e)}", "WARNING")
            stations = None

        # Process trains
        northbound_results = tracker.process_trains(northbound_trains, "N", stations)
        southbound_results = tracker.process_trains(southbound_trains, "S", stations)

        # Combine all results
        all_results = northbound_results + southbound_results

        # Insert into database
        inserted_count = 0
        if all_results:
            inserted_count = tracker.insert_delay_data(all_results)

        # Prepare response
        results = {
            'northbound': northbound_results,
            'southbound': southbound_results,
            'all_delays': all_results,
            'processing_time': datetime.now().isoformat(),
            'total_trains': len(northbound_trains) + len(southbound_trains),
            'records_processed': len(all_results),
            'records_inserted': inserted_count,
            'database_success': inserted_count > 0 if all_results else True
        }

        tracker.log_with_time(
            f"Successfully processed {len(all_results)} train delay records, inserted {inserted_count} to database")

        return {
            'statusCode': 200,
            'body': json.dumps(results, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

    except Exception as e:
        logger.error(f"Lambda execution error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


# For local testing
def main():
    """Main function for local testing"""
    try:
        from nyct_gtfs import NYCTFeed
        from nyct_gtfs.gtfs_static_types import Stations
    except ImportError:
        print("NYCT GTFS modules not available for local testing")
        return

    # Simulate Lambda event
    event = {'line_id': ['N']}
    context = {}

    result = lambda_handler(event, context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()