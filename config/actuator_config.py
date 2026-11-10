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
REACTION_TORQUE_MODEL = "paper_polynomial"

# Paper-based reaction torque model
TORQUE_POLYNOMIAL = (-3.4, 8.7, 2.9)
TORQUE_POLYNOMIAL_SCALE = 1e-4
ROTOR_INERTIA = 0.5e-7  # kg m^2
ENABLE_ROTOR_ACCELERATION_TORQUE = True

# Motor randomization
ENABLE_MOTOR_RANDOMIZATION = False