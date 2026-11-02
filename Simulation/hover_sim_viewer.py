#!/usr/bin/env python3
"""Standalone Crazyflie hover simulation with a live MuJoCo viewer."""
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from simulation_core.pid_controller import (
    CascadePID, K_TAU, MOTOR_DIR, THRUST_MIN, THRUST_MAX,
    PHYSICS_HZ, build_allocation_matrix,
)

MODEL_PATH = Path("models/crazyflie/cf21B_500.xml")
SIM_DURATION = 10.0
DT = 1.0 / PHYSICS_HZ
INITIAL_POS = np.array([0.0, 0.0, 0.0])
POS_DES = np.array([0.0, 0.0, 0.5])
YAW_DES = 0.0
REALTIME_VIEW = True
OUTPUT_DIR = Path("outputs/hover")
CSV_PATH = OUTPUT_DIR / "hover_log.csv"
PLOT_PATH = OUTPUT_DIR / "hover_position.png"


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run this script from the repository root."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    free_id = next(
        (i for i in range(model.njnt)
         if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE),
        None,
    )
    if free_id is None:
        raise RuntimeError("No free joint found in the MuJoCo model.")

    qa = int(model.jnt_qposadr[free_id])
    va = int(model.jnt_dofadr[free_id])
    data.qpos[qa:qa + 3] = INITIAL_POS
    data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[va:va + 6] = 0.0
    mujoco.mj_forward(model, data)

    pid = CascadePID(dt=DT)
    _, allocation_pinv = build_allocation_matrix()
    force_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_force")
        for i in range(4)
    ])
    torque_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_torque")
        for i in range(4)
    ])

    gyro_adr = None
    for sid in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or ""
        if "gyro" in name.lower():
            gyro_adr = int(model.sensor_adr[sid])
            print(f"Using gyro sensor: {name}")
            break

    n_steps = int(round(SIM_DURATION / DT))
    rows = []
    print("\\nStarting standalone Crazyflie hover simulation")
    print(f"Duration: {SIM_DURATION:.2f} s | timestep: {DT:.4f} s")
    print(f"Target position: {POS_DES}")
    print("Motor delay: disabled | Viewer: enabled")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            if not viewer.is_running():
                break

            t = step * DT
            pos_W = data.qpos[qa:qa + 3].copy()
            quat_wxyz = data.qpos[qa + 3:qa + 7].copy()
            vel_W = data.qvel[va:va + 3].copy()
            omega_B = (
                data.sensordata[gyro_adr:gyro_adr + 3].copy()
                if gyro_adr is not None
                else data.qvel[va + 3:va + 6].copy()
            )

            wrench = pid(
                pos_W=pos_W,
                quat_wxyz=quat_wxyz,
                vel_W=vel_W,
                omega_B=omega_B,
                pos_des=POS_DES,
                yaw_des=YAW_DES,
            )
            thrust_cmd = allocation_pinv @ wrench
            thrust = np.clip(thrust_cmd, THRUST_MIN, THRUST_MAX)

            # Ideal instantaneous motors: no first-order lag.
            data.ctrl[force_ids] = thrust
            data.ctrl[torque_ids] = MOTOR_DIR * K_TAU * thrust

            mujoco.mj_step(model, data)
            viewer.sync()

            p = data.qpos[qa:qa + 3].copy()
            v = data.qvel[va:va + 3].copy()
            q = data.qpos[qa + 3:qa + 7].copy()
            rows.append([t, *p, *v, *q, *thrust, *POS_DES])

            if REALTIME_VIEW:
                time.sleep(DT)

    header = [
        "time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "thrust0_N", "thrust1_N", "thrust2_N", "thrust3_N",
        "x_des_m", "y_des_m", "z_des_m",
    ]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    if not rows:
        print("No samples recorded.")
        return

    a = np.asarray(rows, dtype=float)
    ts, pos = a[:, 0], a[:, 1:4]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, label in enumerate(("x (m)", "y (m)", "z (m)")):
        axes[i].plot(ts, pos[:, i], label="Actual")
        axes[i].axhline(POS_DES[i], linestyle="--", label="Target")
        axes[i].set_ylabel(label)
        axes[i].grid(True)
        axes[i].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Crazyflie Hover Position")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print("\\nSimulation finished.")
    print(f"Final position : {pos[-1]}")
    print(f"Target position: {POS_DES}")
    print(f"Position error : {pos[-1] - POS_DES}")
    print(f"CSV log        : {CSV_PATH}")
    print(f"Position plot  : {PLOT_PATH}")


if __name__ == "__main__":
    main()
