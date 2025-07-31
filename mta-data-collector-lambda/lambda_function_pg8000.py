import json
import csv
import urllib3
import zipfile
import re
import io
import logging
import sys
import traceback
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import boto3
from botocore.exceptions import ClientError

# Force immediate logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)


def log_and_print(message, level="INFO"):
    """Immediate logging that appears in CloudWatch faster"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    formatted_msg = f"[{timestamp}] {level}: {message}"
    print(formatted_msg)
    sys.stdout.flush()

    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


class SubwayDelayTracker:
    def __init__(self):
        log_and_print("🔧 SubwayDelayTracker.__init__() started")
        self.stop_times_data = []
        self.stop_times_index = defaultdict(list)
        self.trip_stop_index = defaultdict(dict)
        self.db_engine = None
        self.http = urllib3.PoolManager()
        self.start_time = datetime.now()
        log_and_print("✅ SubwayDelayTracker.__init__() completed")

    def log_execution_time(self, operation: str, context=None):
        """Log execution time since start"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = context.get_remaining_time_in_millis() if context else "N/A"
        log_and_print(f"⏱️ {operation} - Elapsed: {elapsed:.2f}s, Remaining: {remaining}ms")

    def get_rds_credentials_from_secrets(self, secret_name: str, context=None) -> Dict[str, str]:
        """Get RDS credentials from AWS Secrets Manager"""
        log_and_print(f"🔐 get_rds_credentials_from_secrets() started for: {secret_name}")
        self.log_execution_time("Secrets retrieval start", context)

        try:
            log_and_print("📡 Creating Secrets Manager client...")
            secrets_client = boto3.client('secretsmanager')

            log_and_print("📡 Making GetSecretValue API call...")
            response = secrets_client.get_secret_value(SecretId=secret_name)

            log_and_print("🔍 Parsing secret response...")
            secret = json.loads(response['SecretString'])

            credentials = {
                'username': secret.get('username'),
                'password': secret.get('password'),
                'host': secret.get('host'),
                'port': str(secret.get('port', 5432)),
                'dbname': secret.get('dbname', 'postgres')
            }

            log_and_print(f"✅ Successfully retrieved credentials for host: {credentials.get('host', 'unknown')}")
            self.log_execution_time("Secrets retrieval complete", context)
            return credentials

        except Exception as e:
            log_and_print(f"❌ Secrets Manager error: {str(e)}", "ERROR")
            log_and_print(f"📍 Secrets error traceback: {traceback.format_exc()}", "ERROR")
            self.log_execution_time("Secrets retrieval failed", context)
            raise

    def initialize_database(self, db_config: Dict[str, str] = None, context=None):
        """Initialize AWS RDS PostgreSQL database connection"""
        log_and_print("🔧 initialize_database() started")
        self.log_execution_time("DB initialization start", context)

        try:
            if not db_config:
                secret_name = os.getenv('RDS_SECRET_NAME')
                if secret_name:
                    log_and_print(f"🔐 Using Secrets Manager: {secret_name}")
                    db_config = self.get_rds_credentials_from_secrets(secret_name, context)
                else:
                    log_and_print("🌍 Using environment variables for DB config")
                    db_config = {
                        'username': os.getenv('RDS_USERNAME', 'postgres'),
                        'password': os.getenv('RDS_PASSWORD'),
                        'host': os.getenv('RDS_HOST'),
                        'port': os.getenv('RDS_PORT', '5432'),
                        'dbname': 'postgres'
                    }

            # Validate configuration
            log_and_print("✅ Validating database configuration...")
            required = ['username', 'password', 'host']
            missing = [k for k in required if not db_config.get(k)]
            if missing:
                raise ValueError(f"Missing RDS configuration: {', '.join(missing)}")

            log_and_print(f"🌐 Connecting to: {db_config['host']}:{db_config['port']}")

            # Create connection string
            connection_string = (
                f"postgresql+pg8000://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
            )

            log_and_print("🔌 Creating SQLAlchemy engine...")
            self.db_engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=3,
                max_overflow=2,
                connect_args={
                    "timeout": 10,
                    "application_name": "subway_delay_tracker_lambda"
                }
            )

            # Test connection
            log_and_print("🧪 Testing database connection...")
            connection_start = datetime.now()

            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                connection_time = (datetime.now() - connection_start).total_seconds()
                log_and_print(f"✅ Connected in {connection_time:.2f}s to: {version[:50]}...")

            log_and_print("🎉 Database connection initialized successfully")
            self.log_execution_time("DB initialization complete", context)
            return True

        except Exception as e:
            log_and_print(f"❌ Database initialization failed: {str(e)}", "ERROR")
            log_and_print(f"📍 DB error traceback: {traceback.format_exc()}", "ERROR")
            self.log_execution_time("DB initialization failed", context)
            return False

    def create_tables(self, context=None):
        """Create database tables"""
        log_and_print("🏗️ create_tables() started")
        self.log_execution_time("Table creation start", context)

        setup_sql = """
        CREATE SCHEMA IF NOT EXISTS subway_data;
        CREATE TABLE IF NOT EXISTS subway_data.train_delays (
            id BIGSERIAL PRIMARY KEY,
            trip_id TEXT NOT NULL,
            stop_id TEXT NOT NULL,
            station_name TEXT,
            timestamp TIMESTAMPTZ NOT NULL,
            delay_min DOUBLE PRECISION,
            status TEXT NOT NULL,
            direction CHAR(1) NOT NULL,
            stop_sequence INTEGER,
            rush_hour TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_train_delays_timestamp ON subway_data.train_delays(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_train_delays_trip_stop ON subway_data.train_delays(trip_id, stop_id);
        """

        try:
            log_and_print("📝 Executing table creation SQL...")
            sql_start = datetime.now()

            with self.db_engine.begin() as conn:
                conn.execute(text(setup_sql))
                conn.commit()

            sql_time = (datetime.now() - sql_start).total_seconds()
            log_and_print(f"✅ SQL execution completed in {sql_time:.2f}s")
            self.log_execution_time("Table creation complete", context)
            return True

        except Exception as e:
            log_and_print(f"❌ Table creation failed: {str(e)}", "ERROR")
            log_and_print(f"📍 Table error traceback: {traceback.format_exc()}", "ERROR")
            self.log_execution_time("Table creation failed", context)
            return False

    def download_and_parse_gtfs(self, url: str = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip",
                                context=None) -> bool:
        """Download and parse GTFS data - THIS IS LIKELY THE TIMEOUT CULPRIT"""
        log_and_print("📥 download_and_parse_gtfs() started - POTENTIAL TIMEOUT SOURCE")
        log_and_print(f"🔗 URL: {url}")
        self.log_execution_time("GTFS download start", context)

        try:
            # Step 1: Download
            log_and_print("🌐 Starting HTTP download...")
            download_start = datetime.now()

            # Use shorter timeout to fail fast if there are issues
            response = self.http.request('GET', url, timeout=45)

            download_duration = (datetime.now() - download_start).total_seconds()
            log_and_print(f"📊 Download completed in {download_duration:.2f} seconds")
            log_and_print(f"📦 Response status: {response.status}")
            log_and_print(f"📏 Response size: {len(response.data):,} bytes ({len(response.data) / 1024 / 1024:.2f} MB)")
            self.log_execution_time("GTFS download complete", context)

            if response.status != 200:
                raise Exception(f"HTTP {response.status}: Failed to download GTFS data")

            # Step 2: Extract
            log_and_print("📂 Extracting stop_times.txt from ZIP...")
            extract_start = datetime.now()

            with zipfile.ZipFile(io.BytesIO(response.data)) as zip_file:
                file_list = zip_file.namelist()
                log_and_print(f"📋 ZIP contains {len(file_list)} files")

                stop_times_file_path = None
                for file_path in file_list:
                    if file_path.endswith('stop_times.txt'):
                        stop_times_file_path = file_path
                        break

                if not stop_times_file_path:
                    raise FileNotFoundError("stop_times.txt not found in the archive")

                log_and_print(f"✅ Found stop_times.txt at: {stop_times_file_path}")

                with zip_file.open(stop_times_file_path) as stop_times_file:
                    log_and_print("📖 Reading stop_times.txt content...")
                    content = stop_times_file.read().decode('utf-8')
                    log_and_print(f"📄 stop_times.txt size: {len(content):,} characters")

                    # Step 3: Parse CSV
                    log_and_print("🔍 Parsing CSV data...")
                    csv_reader = csv.DictReader(io.StringIO(content))
                    headers = csv_reader.fieldnames
                    log_and_print(f"📋 CSV headers: {headers}")

                    self.stop_times_data = []
                    row_count = 0
                    parse_start = datetime.now()

                    for row in csv_reader:
                        self.stop_times_data.append(row)
                        row_count += 1

                        # Log progress and check remaining time
                        if row_count % 100000 == 0:
                            elapsed = (datetime.now() - parse_start).total_seconds()
                            log_and_print(f"📊 Processed {row_count:,} rows in {elapsed:.2f}s...")
                            self.log_execution_time(f"GTFS parsing - {row_count} rows", context)

            extract_duration = (datetime.now() - extract_start).total_seconds()
            log_and_print(f"✅ Extraction completed in {extract_duration:.2f} seconds")
            log_and_print(f"📊 Successfully loaded {len(self.stop_times_data):,} rows")
            self.log_execution_time("GTFS extraction complete", context)

            # Step 4: Build indexes
            log_and_print("🗂️ Building lookup indexes...")
            index_start = datetime.now()
            self._build_indexes(context)
            index_duration = (datetime.now() - index_start).total_seconds()
            log_and_print(f"✅ Indexes built in {index_duration:.2f} seconds")
            self.log_execution_time("GTFS indexing complete", context)

            return True

        except Exception as e:
            log_and_print(f"❌ GTFS download/parse failed: {str(e)}", "ERROR")
            log_and_print(f"📍 GTFS error traceback: {traceback.format_exc()}", "ERROR")
            self.log_execution_time("GTFS download failed", context)
            return False

    def _build_indexes(self, context=None):
        """Build indexes for faster data lookup"""
        log_and_print(f"🔨 Building indexes for {len(self.stop_times_data):,} rows...")

        processed = 0
        for row in self.stop_times_data:
            trip_id = row.get('trip_id', '').strip()
            stop_id = row.get('stop_id', '').strip()

            if not trip_id or not stop_id:
                continue

            self.stop_times_index[trip_id].append(row)
            self.trip_stop_index[trip_id][stop_id] = row
            processed += 1

            # Check progress and remaining time
            if processed % 200000 == 0:
                log_and_print(f"📊 Indexed {processed:,} rows...")
                if context:
                    remaining = context.get_remaining_time_in_millis()
                    if remaining < 30000:  # Less than 30 seconds remaining
                        log_and_print(f"⚠️ LOW TIME WARNING: Only {remaining}ms remaining!", "WARNING")

        log_and_print(f"✅ Built indexes for {len(self.stop_times_index):,} trips from {processed:,} rows")


def lambda_handler(event, context):
    """Step-by-step Lambda handler to isolate timeout location"""

    log_and_print("🚀 LAMBDA HANDLER STARTED")
    log_and_print(f"📋 Request ID: {context.aws_request_id}")
    log_and_print(f"⏱️ Initial remaining time: {context.get_remaining_time_in_millis()}ms")
    log_and_print(f"💾 Memory limit: {context.memory_limit_in_mb}MB")

    start_time = datetime.now()

    try:
        # STEP 1: Initialize tracker
        log_and_print("🔧 STEP 1: Initializing SubwayDelayTracker...")
        tracker = SubwayDelayTracker()
        log_and_print(f"⏱️ After tracker init - remaining: {context.get_remaining_time_in_millis()}ms")

        # STEP 2: Initialize database
        log_and_print("🗄️ STEP 2: Starting database initialization...")
        if not tracker.initialize_database(context=context):
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to initialize RDS database connection'})
            }
        log_and_print(f"⏱️ After DB init - remaining: {context.get_remaining_time_in_millis()}ms")

        # STEP 3: Create tables
        log_and_print("🏗️ STEP 3: Starting table creation...")
        if not tracker.create_tables(context=context):
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to create RDS database tables'})
            }
        log_and_print(f"⏱️ After table creation - remaining: {context.get_remaining_time_in_millis()}ms")

        # STEP 4: Import NYCT modules
        log_and_print("📦 STEP 4: Importing NYCT GTFS modules...")
        try:
            from nyct_gtfs import NYCTFeed
            from nyct_gtfs.gtfs_static_types import Stations
            log_and_print("✅ NYCT GTFS modules imported successfully")
        except ImportError as e:
            log_and_print(f"❌ NYCT GTFS modules not available: {str(e)}", "ERROR")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Missing nyct_gtfs module'})
            }
        log_and_print(f"⏱️ After GTFS imports - remaining: {context.get_remaining_time_in_millis()}ms")

        # STEP 5: Download GTFS data (MOST LIKELY TIMEOUT LOCATION)
        log_and_print("📥 STEP 5: DOWNLOADING GTFS DATA - WATCHING FOR TIMEOUT...")
        remaining_before_gtfs = context.get_remaining_time_in_millis()
        log_and_print(f"⚠️ CRITICAL: {remaining_before_gtfs}ms remaining before GTFS download")

        if not tracker.download_and_parse_gtfs(context=context):
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to download GTFS data'})
            }

        remaining_after_gtfs = context.get_remaining_time_in_millis()
        log_and_print(f"⏱️ After GTFS download - remaining: {remaining_after_gtfs}ms")
        log_and_print(f"📊 GTFS download used: {remaining_before_gtfs - remaining_after_gtfs}ms")

        # STEP 6: Process trains (if we get this far)
        log_and_print("🚇 STEP 6: Processing subway trains...")
        line_id = event.get('line_id', ['N'])
        log_and_print(f"🚇 Processing line: {line_id}")

        # Initialize NYCT Feed
        log_and_print("🔌 Initializing NYCT Feed...")
        feed_start = datetime.now()

        feed = NYCTFeed("N")
        feed_time = (datetime.now() - feed_start).total_seconds()
        log_and_print(f"✅ NYCT Feed initialized in {feed_time:.2f}s")
        log_and_print(f"⏱️ After NYCT Feed - remaining: {context.get_remaining_time_in_millis()}ms")

        # Filter trains
        log_and_print("🔍 Filtering trains...")
        trains = feed.filter_trips(line_id=line_id, underway=True)
        log_and_print(f"🚂 Found {len(trains)} total trains")
        log_and_print(f"⏱️ After train filtering - remaining: {context.get_remaining_time_in_millis()}ms")

        # Quick success response
        total_time = (datetime.now() - start_time).total_seconds()
        log_and_print(f"🎉 SUCCESS! Lambda completed in {total_time:.2f}s")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Step-by-step execution completed successfully',
                'trains_found': len(trains),
                'execution_time': total_time,
                'remaining_time_ms': context.get_remaining_time_in_millis(),
                'gtfs_processing_time': remaining_before_gtfs - remaining_after_gtfs
            })
        }

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        log_and_print(f"💥 LAMBDA FAILED after {error_time:.2f}s", "ERROR")
        log_and_print(f"❌ Error: {str(e)}", "ERROR")
        log_and_print(f"📍 Error traceback: {traceback.format_exc()}", "ERROR")
        log_and_print(f"⏱️ Remaining time at failure: {context.get_remaining_time_in_millis()}ms")

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'execution_time': error_time,
                'error_type': type(e).__name__
            })
        }


if __name__ == "__main__":
    print("🧪 Running local test...")