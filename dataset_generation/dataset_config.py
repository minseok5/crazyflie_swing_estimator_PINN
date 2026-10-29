"""Configuration for Crazyflie slung-payload dataset generation."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "crazyflie" / "cf21B_500_payload.xml"
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "crazyflie" / "cf21B_500.xml"
MODEL_GENERATOR = PROJECT_ROOT / "create_payload_model.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

PHYSICS_HZ = 500.0
DT = 1.0 / PHYSICS_HZ
DEFAULT_DURATION = 60.0
LOG_INTERVAL = 0.01

INITIAL_DRONE_POSITION = (0.0, 0.0, 0.30)
HOVER_POSITION = (0.0, 0.0, 1.5)
YAW_DES = 0.0
PAYLOAD_MASS = 0.010

ALTITUDE_TOL = 0.03
VERTICAL_VEL_TOL = 0.05
STABLE_DURATION = 1.0

CIRCLE_RADIUS = 0.5
CIRCLE_PERIOD = 10.0
CIRCLE_CENTER_X = 0.5
CIRCLE_CENTER_Y = 0.0

FIGURE8_AMPLITUDE_X = 0.5
FIGURE8_AMPLITUDE_Y = 0.5
FIGURE8_PERIOD = 10.0

STRAIGHT_DISTANCE = 1.0
STRAIGHT_SPEED = 0.2

DEFAULT_TRAJECTORY = "circle"
DEFAULT_SPEED_SCALE = 1.0
DEFAULT_EPISODES = 100
DEFAULT_WORKERS = 4
RANDOM_SEED = 42
INITIAL_SWING_ANGLES_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
SPEED_SCALES = (0.8, 1.0, 1.2)
TRAJECTORIES = ("circle", "figure8", "straight")

ESTIMATOR_INPUT_COLUMNS = (
    "acc_x_mps2", "acc_y_mps2", "acc_z_mps2",
    "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
)
ESTIMATOR_TARGET_COLUMNS = (
    "swing_angle_x_rad", "swing_angle_y_rad",
    "swing_rate_x_radps", "swing_rate_y_radps",
)
