# config/simulation_config.py

from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MuJoCo model
MODEL_DIR = PROJECT_ROOT / "models" / "crazyflie"

MODEL_PATH = MODEL_DIR / "cf21B_500_payload.xml"
BASE_MODEL_PATH = MODEL_DIR / "cf21B_500.xml"

# Simulation frequency
PHYSICS_HZ = 500.0
DT = 1.0 / PHYSICS_HZ

# Simulation duration
SIM_DURATION = 60.0

# Gravity
GRAVITY = 9.81

# Logging
LOG_INTERVAL = 0.01

# Realtime simulation
REALTIME_SIMULATION = False