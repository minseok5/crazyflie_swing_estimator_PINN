# config/vehicle_config.py

# Crazyflie 2.1 Brushless
DRONE_MASS = 0.04338

DRONE_INERTIA = (
    2.3951e-5,
    2.3951e-5,
    3.2347e-5,
)

ARM_LENGTH = 0.035355

# Payload
PAYLOAD_MASS = 0.010

# Rope
ROPE_LENGTH = 0.20

# Initial position
INITIAL_DRONE_POSITION = (0.0, 0.0, 0.30)

# Desired hover position
HOVER_POSITION = (0.0, 0.0, 1.5)

# Real mass and inertia of CD is from MuJoCo model.
# In python configuration, we can control the condition of experiment and the value that is needed to controller.