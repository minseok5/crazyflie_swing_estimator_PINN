# config/actuator_config.py

# Motor confiuration
MOTOR_COUNT = 4

MOTOR_DIRECTION = (
    1.0,
    -1.0,
    1.0,
    -1.0,
)

THRUST_MIN = 0.0
THRUST_MAX = 0.20

# Motor torque coefficient
K_TAU = 0.00594

# Motor dynamics
ENABLE_MOTOR_DYNAMICS = True
MOTOR_TIME_CONSTANT = 0.050

# RPM-thrust mapping
OMEGA_REFERENCE = 2900.0

THRUST_POLYNOMIAL = (
    -0.23,
    0.562,
    -0.043,
)

# Reaction torque
ENABLE_REACTION_TORQUE = True
REACTION_TORQUE_MODEL = "legacy_ratio"

# Motor randomization
ENABLE_MOTOR_RANDOMIZATION = False