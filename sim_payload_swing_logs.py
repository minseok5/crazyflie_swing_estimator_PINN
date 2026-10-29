#!/usr/bin/env python3
"""Preview a Crazyflie model with an articulated slung payload in MuJoCo.
Run from repository root: python hover_sim_payload_viewer.py
It generates cf21B_500_payload.xml if absent, then runs a simple hover controller.
"""
from pathlib import Path
import csv
from datetime import datetime
import subprocess
import sys
import time
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

from simulation_core.pid_controller import (
    CascadePID, K_TAU, MOTOR_DIR, THRUST_MIN, THRUST_MAX,
    PHYSICS_HZ, build_allocation_matrix,
)

MODEL_PATH = Path('models/crazyflie/cf21B_500_payload.xml')
BASE_MODEL = Path('models/crazyflie/cf21B_500.xml')
DT = 1.0 / PHYSICS_HZ
SIM_DURATION = 60.0
# Keep the viewer's requested start position. A z offset can be used if the
# ground contact geometry interferes with startup.
INITIAL_POS = np.array([0.0, 0.0, 0.0])

HOVER_POS = np.array([0.0, 0.0, 1.5])
# MOVE_POS = np.array([3.5, 2.5, 1.5])

# Circular trajectory parameters
CIRCLE_RADIUS = 0.5
CIRCLE_PERIOD = 10.0
CIRCLE_CENTER_X = 0.5
CIRCLE_CENTER_Y = 0.0
CIRCLE_ALTITUDE = 0.5

# Hover stabilization conditions
ALTITUDE_TOL = 0.03     # m
VERTICAL_VEL_TOL = 0.05 # m/s
STABLE_DURATION = 1.0   # s

HOVER_DURATION = 5.0

YAW_DES = 0.0
REALTIME_VIEW = True
PAYLOAD_MASS = 0.020

# CSV logging
LOG_DIR = Path("outputs/payload")
GRAPH_DIR = Path("outputs/graph")
LOG_INTERVAL = 0.01 # 10 ms


def get_circular_target(elapsed_time):
        """Retrun the desired position along a circular trajectory"""

        omega = 2.0 *np.pi /CIRCLE_PERIOD
        theta = omega * elapsed_time

        x_des = CIRCLE_CENTER_X - CIRCLE_RADIUS * np.cos(5 * theta)
        y_des = CIRCLE_CENTER_Y + CIRCLE_RADIUS * np.sin(theta)
        z_des = 2.0 + CIRCLE_ALTITUDE # * np.sin(theta / 5)

        return np.array([x_des, y_des, z_des])


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f'MuJoCo object not found: {name}')
    return idx



def plot_simulation_results(csv_path):
    """Plot logged drone position, velocities, attitude, and payload swing."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No simulation data found; skipping plots.")
        return

    def col(name):
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    t = col("time_s")
    pos = np.column_stack([col("drone_x_m"), col("drone_y_m"), col("drone_z_m")])
    pos_des = np.column_stack([col("target_x_m"), col("target_y_m"), col("target_z_m")])
    vel = np.column_stack([col("drone_vx_mps"), col("drone_vy_mps"), col("drone_vz_mps")])
    omega = np.column_stack([
        col("gyro_x_radps"), col("gyro_y_radps"), col("gyro_z_radps")
    ])
    quat = np.column_stack([
        col("quat_w"), col("quat_x"), col("quat_y"), col("quat_z")
    ])
    payload = np.column_stack([
        col("payload_x_m"), col("payload_y_m"), col("payload_z_m")
    ])
    swing_angle = np.column_stack([
        col("swing_angle_x_rad"), col("swing_angle_y_rad")
    ])
    swing_rate = np.column_stack([
        col("swing_rate_x_radps"), col("swing_rate_y_radps")
    ])

    # Convert quaternion (w, x, y, z) to roll, pitch, yaw (radians).
    qw, qx, qy, qz = quat.T
    roll = np.arctan2(2.0 * (qw * qx + qy * qz),
                      1.0 - 2.0 * (qx * qx + qy * qy))
    sin_pitch = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))
    attitude_deg = np.rad2deg(np.column_stack([roll, pitch, yaw]))

    plot_specs = [
        ("Drone position", pos, ["x", "y", "z"], "Position (m)", pos_des, ["x desired", "y desired", "z desired"]),
        ("Linear velocity", vel, ["vx", "vy", "vz"], "Velocity (m/s)", None, None),
        ("Angular velocity", omega, ["wx", "wy", "wz"], "Angular velocity (rad/s)", None, None),
        ("Drone attitude (Euler angles)", attitude_deg,
         ["roll", "pitch", "yaw"], "Angle (deg)", None, None),
        ("Payload swing angle", np.rad2deg(swing_angle),
         ["swing x", "swing y"], "Swing angle (deg)", None, None),
        ("Payload swing angular velocity", swing_rate,
         ["swing rate x", "swing rate y"], "Angular velocity (rad/s)", None, None),
    ]

    # Save all plots in one figure, arranged as a 3x2 grid.
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=True)
    axes = axes.ravel()

    for ax, (title, values, labels, ylabel, desired_values, desired_labels) in zip(axes, plot_specs):
        for i, label in enumerate(labels):
            ax.plot(t, values[:, i], label=f"{label} actual")
        if desired_values is not None:
            for i, label in enumerate(desired_labels):
                ax.plot(t, desired_values[:, i], linestyle="--", label=label)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Crazyflie Simulation State Results", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    csv_stem = Path(csv_path).stem  # e.g. hover_payload_20261027_132200
    timestamp = "_".join(csv_stem.split("_")[-2:])
    graph_path = GRAPH_DIR / f"hover_payload_graph_{timestamp}.png"
    fig.savefig(graph_path, dpi=200, bbox_inches="tight")
    print(f"Graph saved: {graph_path}")

    plt.show()
    plt.close(fig)

def main():
    if not MODEL_PATH.exists():
        generator = Path('create_payload_model.py')
        if not generator.exists():
            raise FileNotFoundError(f'{generator} missing. Place it in the repository root.')
        subprocess.run([sys.executable, str(generator)], check=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    payload_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "payload_body"
    )

 
    free_id = next((i for i in range(model.njnt)
                    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE), None)
    if free_id is None:
        raise RuntimeError('No free joint found in the model.')
    qa = int(model.jnt_qposadr[free_id])
    va = int(model.jnt_dofadr[free_id])
    data.qpos[qa:qa+3] = INITIAL_POS
    data.qpos[qa+3:qa+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[va:va+6] = 0.0

    # Controller feed-forward mass includes drone and payload. This is only a
    # first hover baseline; the payload swing is not actively damped here.
    vehicle = SimpleNamespace(mass=0.04338 + PAYLOAD_MASS, gravity=9.81,
                              inertia_diagonal=(2.3951e-5, 2.3951e-5, 3.2347e-5))
    pid = CascadePID(dt=DT, vehicle=vehicle)
    _, allocation_pinv = build_allocation_matrix()
    force_ids = np.array([object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i}_force') for i in range(4)])
    torque_ids = np.array([object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i}_torque') for i in range(4)])
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
    gyro_adr = int(model.sensor_adr[gyro_id]) if gyro_id >= 0 else None
    gyro_dim = int(model.sensor_dim[gyro_id])

    # Get eccelerometer sensor ID
    acc_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
    acc_adr = int(model.sensor_adr[acc_id])
    acc_dim = int(model.sensor_dim[acc_id])

    # Read the two articulated payload swing joints. Their generalized
    # positions and velocities provide the simulated ground-truth swing state.
    swing_x_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing_joint_x")
    swing_y_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing_joint_y")
    swing_x_qadr = int(model.jnt_qposadr[swing_x_id])
    swing_y_qadr = int(model.jnt_qposadr[swing_y_id])
    swing_x_vadr = int(model.jnt_dofadr[swing_x_id])
    swing_y_vadr = int(model.jnt_dofadr[swing_y_id])

    mujoco.mj_forward(model, data)

    # Create CSV log file
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"hover_payload_{timestamp}.csv"

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "time_s",
        "target_x_m", "target_y_m", "target_z_m",
        "drone_x_m", "drone_y_m", "drone_z_m",
        "drone_vx_mps", "drone_vy_mps", "drone_vz_mps",
        "quat_w", "quat_x", "quat_y", "quat_z",

        # Gyroscope
        "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
        # Acclerometer
        "acc_x_mps2", "acc_y_mps2", "acc_z_mps2",

        "payload_x_m", "payload_y_m", "payload_z_m",
        "swing_angle_x_rad", "swing_angle_y_rad",
        "swing_rate_x_radps", "swing_rate_y_radps",
        "motor0_thrust_N", "motor1_thrust_N",
        "motor2_thrust_N", "motor3_thrust_N",
        "torque_x_Nm", "torque_y_Nm",
        "torque_z_Nm", "collective_force_N"
    ])

    print(f'Model: {MODEL_PATH}; payload mass={PAYLOAD_MASS:.3f} kg; duration={SIM_DURATION}s')
    print(f'CSV log: {csv_path}')

    # Initialize these variables before entering the simulation loop:
    stable_start_time = None
    move_started = False
    move_start_time = None


    with mujoco.viewer.launch_passive(model, data) as viewer:
        n_steps = int(SIM_DURATION / DT)
        for _ in range(n_steps):
            if not viewer.is_running():
                break
            pos = data.qpos[qa:qa+3].copy()
            quat = data.qpos[qa+3:qa+7].copy()
            vel = data.qvel[va:va+3].copy()

            # Gyroscope: angular velocity
            omega = data.sensordata[gyro_adr:gyro_adr + gyro_dim].copy() if gyro_adr is not None else data.qvel[va+3:va+6].copy()

            # Accelerometer: specific force
            acc = data.sensordata[acc_adr:acc_adr + acc_dim].copy()

            # Start horizontal movement only after stable hovering
            if not move_started:
                altitude_error = abs(HOVER_POS[2] - pos[2])
                vertical_speed = abs(vel[2])

                if (
                    altitude_error < ALTITUDE_TOL
                    and vertical_speed < VERTICAL_VEL_TOL
                ):
                    if stable_start_time is None:
                        stable_start_time = data.time

                    elif data.time - stable_start_time >= STABLE_DURATION:
                        move_started = True
                        move_start_time = data.time

                        print(
                            f"Hover stabilized at f={data.time:.2f}s. "
                            "starting circular movement."
                        )
                else:
                    stable_start_time = None

            if move_started:
                #pos_des = MOVE_POS
                elapsed_time = data.time - move_start_time
                pos_des = get_circular_target(elapsed_time)
            else:
                pos_des = HOVER_POS

            wrench = pid(pos_W=pos, quat_wxyz=quat, vel_W=vel, omega_B=omega,
                         pos_des=pos_des, yaw_des=YAW_DES)
            thrust = np.clip(allocation_pinv @ wrench, THRUST_MIN, THRUST_MAX)
            data.ctrl[force_ids] = thrust
            data.ctrl[torque_ids] = MOTOR_DIR * K_TAU * thrust
            mujoco.mj_step(model, data)

            # Record simulation data at LOG_INTERVAL
            if not hasattr(main, "_last_log_time"):
                main._last_log_time = -LOG_INTERVAL

            if data.time - main._last_log_time >= LOG_INTERVAL:
                payload_pos = data.xpos[payload_id].copy()
                swing_angle_x = float(data.qpos[swing_x_qadr])
                swing_angle_y = float(data.qpos[swing_y_qadr])
                swing_rate_x = float(data.qvel[swing_x_vadr])
                swing_rate_y = float(data.qvel[swing_y_vadr])

                csv_writer.writerow([
                    data.time,
                    *pos_des,
                    *data.qpos[qa:qa+3],
                    *data.qvel[va:va+3],
                    *data.qpos[qa+3:qa+7],
                    # Gyroscope
                    *omega,
                    # Accelerometer
                    *acc,
                    # Payload state
                    *payload_pos,
                    swing_angle_x, swing_angle_y,
                    swing_rate_x, swing_rate_y,
                    *thrust,
                    *wrench
                ])

                main._last_log_time = data.time

            # Print drone and payload positions every 0.5 seconds
            if int(data.time / 0.5) != int((data.time - DT) / 0.5):
                drone_pos = data.qpos[qa:qa+3].copy()
                payload_pos = data.xpos[payload_id].copy()

                print(
                    f"t={data.time:6.2f}s | "
                    f"Drone: x={drone_pos[0]: .3f}, "
                    f"y={drone_pos[1]: .3f}, "
                    f"z={drone_pos[2]: .3f} | "
                    f"Payload: {payload_pos[0]: .3f}, "
                    f"y={payload_pos[1]: .3f}, "
                    f"z={payload_pos[2]: .3f}"
                )
            viewer.sync()
            if REALTIME_VIEW:
                time.sleep(DT)
    csv_file.close()
    print(f"CSV saved: {csv_path}")
    print("Viewer closed.")
    plot_simulation_results(csv_path)

if __name__ == '__main__':
    main()
