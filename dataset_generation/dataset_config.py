"""Configuration for Crazyflie slung-payload dataset generation."""

from pathlib import Path

# ============================================================
# Shared simulation configuration
# ============================================================

from config.simulation_config import (
    PROJECT_ROOT,
    MODEL_PATH,
    BASE_MODEL_PATH,
    PHYSICS_HZ,
    DT,
    SIM_DURATION,
    GRAVITY,
    LOG_INTERVAL,
)

from config.vehicle_config import (
    DRONE_MASS,
    DRONE_INERTIA,
    INITIAL_DRONE_POSITION,
    HOVER_POSITION,
    PAYLOAD_MASS,
)

# ============================================================
# Dataset paths
# ============================================================

MODEL_GENERATOR = PROJECT_ROOT / "create_payload_model.py"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ============================================================
# Dataset simulation settings
# ============================================================

DEFAULT_DURATION = SIM_DURATION

YAW_DES = 0.0

ALTITUDE_TOL = 0.03
VERTICAL_VEL_TOL = 0.05
STABLE_DURATION = 1.0

# ============================================================
# Trajectory settings
# ============================================================

CIRCLE_RADIUS = 0.5
CIRCLE_PERIOD = 10.0
CIRCLE_CENTER_X = 0.5
CIRCLE_CENTER_Y = 0.0

FIGURE8_AMPLITUDE_X = 0.5
FIGURE8_AMPLITUDE_Y = 0.5
FIGURE8_PERIOD = 10.0

STRAIGHT_DISTANCE = 1.0
STRAIGHT_SPEED = 0.2

# ============================================================
# Dataset generation settings
# ============================================================

DEFAULT_TRAJECTORY = "circle"
DEFAULT_SPEED_SCALE = 1.0

DEFAULT_EPISODES = 100
DEFAULT_WORKERS = 4

RANDOM_SEED = 42

INITIAL_SWING_ANGLES_DEG = (
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
)

SPEED_SCALES = (0.8, 1.0, 1.2)

TRAJECTORIES = (
    "circle",
    "figure8",
    "straight",
)

# ============================================================
# Estimator data configuration
# ============================================================

ESTIMATOR_INPUT_COLUMNS = (
    "acc_x_mps2",
    "acc_y_mps2",
    "acc_z_mps2",
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",
)

ESTIMATOR_TARGET_COLUMNS = (
    "swing_angle_x_rad",
    "swing_angle_y_rad",
    "swing_rate_x_radps",
    "swing_rate_y_radps",
)