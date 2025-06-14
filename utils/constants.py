# App Constants
APP_NAME = "Pothole Detection ML"
APP_VERSION = "1.0.0"

# Model Configuration
SEQUENCE_LENGTH = 50
N_FEATURES = 6
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# Sensor Configuration
SAMPLING_RATE = 50  # Hz
GPS_ACCURACY_THRESHOLD = 10  # meters
ROUTE_DEVIATION_THRESHOLD = 100  # meters

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.7
DETECTION_COOLDOWN = 2.0  # seconds
POTHOLE_WINDOW = 3.0  # seconds for data collection

# Rule-based Thresholds (fallback)
ACCEL_Z_DROP_THRESHOLD = -2.5
ACCEL_Z_RISE_THRESHOLD = 3.0
GYRO_THRESHOLD = 1.5
ACCEL_MAGNITUDE_THRESHOLD = 15.0

# Database Configuration
DB_FILE = "database/locations.db"
LOCATION_DISTANCE_THRESHOLD = 0.00005  # ~5 meters in decimal degrees

# UI Constants
SEARCH_BAR_HEIGHT = 50
NAVIGATION_PANEL_HEIGHT = 120
