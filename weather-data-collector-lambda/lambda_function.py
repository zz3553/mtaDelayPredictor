import json
import os
import boto3
import requests
import pg8000
from datetime import datetime
import time
import logging

from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_neon_db_credentials_from_aws_sm():
    """Retrieve database credentials from AWS Secrets Manager"""
    secret_name = "neon_db_credentials_updated"
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


class WeatherDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

        aws_sm_response = get_neon_db_credentials_from_aws_sm()
        self.db_config = {
            'user': aws_sm_response['username'],
            'password': aws_sm_response['password'],
            'host': aws_sm_response['host'],
            'port': aws_sm_response['port'],
            'database': aws_sm_response['dbname']
        }

        self.init_database()

    def init_database(self):
        """Initialize PostgreSQL database with weather data table"""
        try:
            conn = pg8000.connect(
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                ssl_context=True
            )
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    location_name TEXT,
                    latitude REAL,
                    longitude REAL,

                    -- Temperature data (in Fahrenheit)
                    temp_fahrenheit REAL,
                    feels_like_fahrenheit REAL,
                    temp_min_fahrenheit REAL,
                    temp_max_fahrenheit REAL,

                    -- Weather conditions
                    weather_main TEXT,
                    weather_description TEXT,
                    weather_id INTEGER,

                    -- Atmospheric data
                    pressure INTEGER,
                    humidity INTEGER,
                    sea_level_pressure INTEGER,
                    ground_level_pressure INTEGER,
                    visibility INTEGER,

                    -- Wind data
                    wind_speed REAL,
                    wind_direction INTEGER,
                    wind_gust REAL,

                    -- Precipitation data
                    rain_1h REAL,
                    rain_3h REAL,
                    snow_1h REAL,
                    snow_3h REAL,

                    -- Cloud data
                    cloudiness INTEGER,

                    -- Additional fields for MTA analysis
                    is_precipitation BOOLEAN,
                    is_snow BOOLEAN,
                    is_extreme_temp BOOLEAN,
                    is_high_humidity BOOLEAN,
                    is_high_wind BOOLEAN,
                    weather_severity_score INTEGER,

                    -- Timestamps
                    sunrise_time TIMESTAMP,
                    sunset_time TIMESTAMP,
                    data_timestamp TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("PostgreSQL database initialized successfully")
        except pg8000.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def kelvin_to_fahrenheit(self, kelvin):
        """Convert Kelvin directly to Fahrenheit"""
        return (kelvin - 273.15) * 9 / 5 + 32

    def calculate_weather_severity_score(self, weather_data):
        """
        Calculate a weather severity score (1-10) for MTA delay correlation
        Higher scores indicate more severe weather conditions
        """
        score = 1

        # Temperature extremes
        temp_f = weather_data.get('temp_fahrenheit', 32)
        if temp_f < 14 or temp_f > 95:  # Extreme temperatures
            score += 3
        elif temp_f < 32 or temp_f > 86:  # Very cold/hot
            score += 2
        elif temp_f < 41 or temp_f > 77:  # Cold/warm
            score += 1

        # Precipitation
        rain_1h = weather_data.get('rain_1h', 0) or 0
        snow_1h = weather_data.get('snow_1h', 0) or 0

        if snow_1h > 5:  # Heavy snow
            score += 3
        elif snow_1h > 0:  # Any snow
            score += 2
        elif rain_1h > 10:  # Heavy rain
            score += 2
        elif rain_1h > 2:  # Moderate rain
            score += 1

        # Wind
        wind_speed = weather_data.get('wind_speed', 0) or 0
        if wind_speed > 15:  # High wind
            score += 2
        elif wind_speed > 10:  # Moderate wind
            score += 1

        # Humidity
        humidity = weather_data.get('humidity', 0) or 0
        if humidity > 90:  # Very high humidity
            score += 1

        # Visibility
        visibility = weather_data.get('visibility', 10000) or 10000
        if visibility < 1000:  # Poor visibility
            score += 2
        elif visibility < 5000:  # Reduced visibility
            score += 1

        return min(score, 10)  # Cap at 10

    def fetch_weather_data(self, lat, lon, location_name=None):
        """Fetch weather data from OpenWeatherMap API"""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def process_weather_data(self, raw_data, location_name=None):
        """Process raw weather data into structured format"""
        if not raw_data:
            return None

        # Extract basic info
        coord = raw_data.get('coord', {})
        main = raw_data.get('main', {})
        weather = raw_data.get('weather', [{}])[0]
        wind = raw_data.get('wind', {})
        rain = raw_data.get('rain', {})
        snow = raw_data.get('snow', {})
        clouds = raw_data.get('clouds', {})
        sys = raw_data.get('sys', {})

        # Temperature conversions
        temp_f = self.kelvin_to_fahrenheit(main.get('temp', 0))
        feels_like_f = self.kelvin_to_fahrenheit(main.get('feels_like', 0))
        temp_min_f = self.kelvin_to_fahrenheit(main.get('temp_min', 0))
        temp_max_f = self.kelvin_to_fahrenheit(main.get('temp_max', 0))

        processed_data = {
            'timestamp': datetime.now(),
            'location_name': location_name or raw_data.get('name', 'Unknown'),
            'latitude': coord.get('lat'),
            'longitude': coord.get('lon'),

            # Temperature
            'temp_fahrenheit': round(temp_f, 2),
            'feels_like_fahrenheit': round(feels_like_f, 2),
            'temp_min_fahrenheit': round(temp_min_f, 2),
            'temp_max_fahrenheit': round(temp_max_f, 2),

            # Weather conditions
            'weather_main': weather.get('main'),
            'weather_description': weather.get('description'),
            'weather_id': weather.get('id'),

            # Atmospheric
            'pressure': main.get('pressure'),
            'humidity': main.get('humidity'),
            'sea_level_pressure': main.get('sea_level'),
            'ground_level_pressure': main.get('grnd_level'),
            'visibility': raw_data.get('visibility'),

            # Wind
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'wind_gust': wind.get('gust'),

            # Precipitation
            'rain_1h': rain.get('1h'),
            'rain_3h': rain.get('3h'),
            'snow_1h': snow.get('1h'),
            'snow_3h': snow.get('3h'),

            # Clouds
            'cloudiness': clouds.get('all'),

            # Analysis flags
            'is_precipitation': bool(rain.get('1h', 0) or snow.get('1h', 0)),
            'is_snow': bool(snow.get('1h', 0)),
            'is_extreme_temp': temp_f < 23 or temp_f > 95,
            'is_high_humidity': main.get('humidity', 0) > 85,
            'is_high_wind': wind.get('speed', 0) > 12,

            # Timestamps
            'sunrise_time': datetime.fromtimestamp(sys.get('sunrise', 0)) if sys.get('sunrise') else None,
            'sunset_time': datetime.fromtimestamp(sys.get('sunset', 0)) if sys.get('sunset') else None,
            'data_timestamp': datetime.fromtimestamp(raw_data.get('dt', 0)) if raw_data.get('dt') else None
        }

        # Calculate severity score
        processed_data['weather_severity_score'] = self.calculate_weather_severity_score(processed_data)

        return processed_data

    def save_to_database(self, weather_data):
        """Save processed weather data to PostgreSQL database"""
        if not weather_data:
            return False

        try:
            conn = pg8000.connect(
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                ssl_context=True
            )
            cursor = conn.cursor()

            columns = list(weather_data.keys())
            placeholders = ', '.join(['%s' for _ in columns])
            column_names = ', '.join(columns)

            query = f"INSERT INTO weather_data ({column_names}) VALUES ({placeholders})"

            cursor.execute(query, list(weather_data.values()))
            conn.commit()
            logger.info(f"Weather data saved for {weather_data['location_name']}")
            return True
        except pg8000.Error as e:
            logger.error(f"Database error: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def collect_weather_data(self, locations):
        """
        Collect weather data for multiple locations
        locations: list of tuples [(lat, lon, name), ...]
        """
        results = []

        for lat, lon, name in locations:
            logger.info(f"Fetching weather data for {name}")

            # Fetch raw data
            raw_data = self.fetch_weather_data(lat, lon, name)

            if raw_data:
                # Process data
                processed_data = self.process_weather_data(raw_data, name)

                if processed_data:
                    # Save to database
                    if self.save_to_database(processed_data):
                        results.append(processed_data)

                    # Log summary
                    logger.info(f"{name}: {processed_data['temp_fahrenheit']:.1f}°F, "
                                f"{processed_data['weather_description']}, "
                                f"Severity: {processed_data['weather_severity_score']}/10")

            # Be respectful to the API (reduced sleep for Lambda efficiency)
            time.sleep(0.5)

        return results


def lambda_handler(event, context):
    """
    AWS Lambda handler function

    Expected event structure (optional):
    {
        "locations": [
            {"lat": 40.7831, "lon": -73.9712, "name": "NYC_Manhattan"},
            {"lat": 40.6782, "lon": -73.9442, "name": "NYC_Brooklyn"}
        ]
    }

    If no locations provided, uses default NYC locations
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'OPENWEATHER_API_KEY environment variable not set'
                })
            }

        # Initialize collector
        collector = WeatherDataCollector(api_key)

        # Get locations from event or use defaults
        if event and 'locations' in event:
            locations = [(loc['lat'], loc['lon'], loc['name']) for loc in event['locations']]
        else:
            # Default NYC area locations for MTA analysis
            locations = [
                (40.7831, -73.9712, "NYC_Manhattan"),
                (40.6782, -73.9442, "NYC_Brooklyn"),
                (40.7282, -73.7949, "NYC_Queens"),
                (40.7505, -73.9934, "NYC_Midtown_Manhattan"),
                (40.6892, -74.0445, "NYC_Lower_Manhattan")
            ]

        # Collect weather data
        results = collector.collect_weather_data(locations)

        # Prepare response
        response_data = {
            'message': f'Successfully collected weather data for {len(results)} locations',
            'locations_processed': len(results),
            'total_locations': len(locations),
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'location': result['location_name'],
                    'temperature_f': result['temp_fahrenheit'],
                    'conditions': result['weather_description'],
                    'severity_score': result['weather_severity_score']
                }
                for result in results
            ]
        }

        logger.info(f"Lambda execution completed successfully. Processed {len(results)} locations.")

        return {
            'statusCode': 200,
            'body': json.dumps(response_data, default=str)
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }