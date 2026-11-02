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

HOVER_POS = np.array([0.0, 0.0, 0.7])
MOVE_POS = np.array([0.5, 0.0, 0.7])

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
LOG_INTERVAL = 0.01 # 10 ms


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f'MuJoCo object not found: {name}')
    return idx


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
        "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
        "payload_x_m", "payload_y_m", "payload_z_m",
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


    with mujoco.viewer.launch_passive(model, data) as viewer:
        n_steps = int(SIM_DURATION / DT)
        for _ in range(n_steps):
            if not viewer.is_running():
                break
            pos = data.qpos[qa:qa+3].copy()
            quat = data.qpos[qa+3:qa+7].copy()
            vel = data.qvel[va:va+3].copy()
            omega = data.sensordata[gyro_adr:gyro_adr+3].copy() if gyro_adr is not None else data.qvel[va+3:va+6].copy()

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
                        print(
                            f"Hover stabilized at f={data.time:.2f}s. "
                            "starting horizontal movement."
                        )
                else:
                    stable_start_time = None

            if move_started:
                pos_des = MOVE_POS
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

                csv_writer.writerow([
                    data.time,
                    *pos_des,
                    *data.qpos[qa:qa+3],
                    *data.qvel[va:va+3],
                    *data.qpos[qa+3:qa+7],
                    *omega,
                    *payload_pos,
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
    print('Viewer closed.')

if __name__ == '__main__':
    main()
