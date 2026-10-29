#!/usr/bin/env python3
"""Run one headless Crazyflie + slung-payload episode and save CSV logs.

Run from the repository root:
    python -m dataset_generation.sim_payload_dataset --duration 10
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import mujoco
import numpy as np

from simulation_core.pid_controller import (
    CascadePID, K_TAU, MOTOR_DIR, THRUST_MIN, THRUST_MAX,
    PHYSICS_HZ, build_allocation_matrix,
)
from . import dataset_config as cfg


def ensure_model() -> None:
    """Create the payload XML once if the repository's generator is available."""
    if cfg.MODEL_PATH.exists():
        return
    if not cfg.MODEL_GENERATOR.exists():
        raise FileNotFoundError(
            f"Model not found: {cfg.MODEL_PATH}. "
            f"Generator not found: {cfg.MODEL_GENERATOR}"
        )
    subprocess.run([sys.executable, str(cfg.MODEL_GENERATOR)],
                   cwd=cfg.PROJECT_ROOT, check=True)
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError(f"Generator did not create {cfg.MODEL_PATH}")


def object_id(model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(idx)


def get_target(trajectory: str, elapsed: float, speed_scale: float) -> np.ndarray:
    """Desired position. Circle preserves the current viewer's target equation."""
    z = float(cfg.HOVER_POSITION[2])
    if trajectory == "circle":
        omega = 2.0 * math.pi / cfg.CIRCLE_PERIOD * speed_scale
        theta = omega * elapsed
        # Preserve the original simulation's existing trajectory equation.
        x = cfg.CIRCLE_CENTER_X - cfg.CIRCLE_RADIUS * math.cos(2.0 * theta)
        y = cfg.CIRCLE_CENTER_Y + cfg.CIRCLE_RADIUS * math.sin(theta)
    elif trajectory == "figure8":
        omega = 2.0 * math.pi / cfg.FIGURE8_PERIOD * speed_scale
        theta = omega * elapsed
        x = cfg.CIRCLE_CENTER_X + cfg.FIGURE8_AMPLITUDE_X * math.sin(theta)
        y = cfg.CIRCLE_CENTER_Y + cfg.FIGURE8_AMPLITUDE_Y * math.sin(theta) * math.cos(theta)
    elif trajectory == "straight":
        # Smooth out-and-back path, bounded to STRAIGHT_DISTANCE.
        period = max(2.0 * cfg.STRAIGHT_DISTANCE / cfg.STRAIGHT_SPEED / speed_scale, 1e-6)
        phase = 2.0 * math.pi * elapsed / period
        x = cfg.CIRCLE_CENTER_X + 0.5 * cfg.STRAIGHT_DISTANCE * math.sin(phase)
        y = cfg.CIRCLE_CENTER_Y
    else:
        raise ValueError(f"Unsupported trajectory {trajectory!r}; choose {cfg.TRAJECTORIES}")
    return np.array([x, y, z], dtype=float)


CSV_COLUMNS = [
    "episode_id", "time_s", "trajectory_started",
    "target_x_m", "target_y_m", "target_z_m",
    "drone_x_m", "drone_y_m", "drone_z_m",
    "drone_vx_mps", "drone_vy_mps", "drone_vz_mps",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
    "acc_x_mps2", "acc_y_mps2", "acc_z_mps2",
    "payload_x_m", "payload_y_m", "payload_z_m",
    "swing_angle_x_rad", "swing_angle_y_rad",
    "swing_rate_x_radps", "swing_rate_y_radps",
    "motor0_thrust_N", "motor1_thrust_N", "motor2_thrust_N", "motor3_thrust_N",
    "torque_x_Nm", "torque_y_Nm", "torque_z_Nm", "collective_force_N",
]


def run_episode(
    episode_id: int = 1,
    output_dir: str | Path | None = None,
    duration: float = cfg.DEFAULT_DURATION,
    trajectory: str = cfg.DEFAULT_TRAJECTORY,
    initial_swing_x_rad: float = 0.0,
    initial_swing_y_rad: float = 0.0,
    speed_scale: float = cfg.DEFAULT_SPEED_SCALE,
    seed: int = 0,
    realtime: bool = False,
) -> dict:
    """Run a fresh, independent episode and return its metadata."""
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if speed_scale <= 0:
        raise ValueError("speed_scale must be > 0")
    if trajectory not in cfg.TRAJECTORIES:
        raise ValueError(f"trajectory must be one of {cfg.TRAJECTORIES}")

    ensure_model()
    model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.DT

    payload_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_body")
    free_id = next(
        (i for i in range(model.njnt)
         if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE),
        None,
    )
    if free_id is None:
        raise RuntimeError("No free joint found in the model.")
    qa = int(model.jnt_qposadr[free_id])
    va = int(model.jnt_dofadr[free_id])
    data.qpos[qa:qa + 3] = np.asarray(cfg.INITIAL_DRONE_POSITION, dtype=float)
    data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[va:va + 6] = 0.0

    swing_x_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing_joint_x")
    swing_y_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing_joint_y")
    swing_x_qadr = int(model.jnt_qposadr[swing_x_id])
    swing_y_qadr = int(model.jnt_qposadr[swing_y_id])
    swing_x_vadr = int(model.jnt_dofadr[swing_x_id])
    swing_y_vadr = int(model.jnt_dofadr[swing_y_id])
    data.qpos[swing_x_qadr] = float(initial_swing_x_rad)
    data.qpos[swing_y_qadr] = float(initial_swing_y_rad)
    mujoco.mj_forward(model, data)

    vehicle = SimpleNamespace(
        mass=0.04338 + cfg.PAYLOAD_MASS,
        gravity=9.81,
        inertia_diagonal=(2.3951e-5, 2.3951e-5, 3.2347e-5),
    )
    pid = CascadePID(dt=cfg.DT, vehicle=vehicle)
    _, allocation_pinv = build_allocation_matrix()
    force_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_force")
        for i in range(4)
    ])
    torque_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_torque")
        for i in range(4)
    ])

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    gyro_adr = int(model.sensor_adr[gyro_id]) if gyro_id >= 0 else None
    gyro_dim = int(model.sensor_dim[gyro_id]) if gyro_id >= 0 else 3
    acc_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
    acc_adr = int(model.sensor_adr[acc_id])
    acc_dim = int(model.sensor_dim[acc_id])

    out_dir = Path(output_dir) if output_dir is not None else cfg.OUTPUT_DIR
    if not out_dir.is_absolute():
        out_dir = cfg.PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = out_dir / f"episode_{episode_id:06d}_{stamp}.csv"

    stable_start_time = None
    move_started = False
    move_start_time = None
    last_log_time = -cfg.LOG_INTERVAL
    status = "completed"
    n_steps = int(math.ceil(duration / cfg.DT))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for _ in range(n_steps):
            pos = data.qpos[qa:qa + 3].copy()
            quat = data.qpos[qa + 3:qa + 7].copy()
            vel = data.qvel[va:va + 3].copy()
            if gyro_adr is not None:
                omega = data.sensordata[gyro_adr:gyro_adr + gyro_dim].copy()
            else:
                omega = data.qvel[va + 3:va + 6].copy()
            acc = data.sensordata[acc_adr:acc_adr + acc_dim].copy()

            if not np.all(np.isfinite(np.concatenate([pos, quat, vel, omega, acc]))):
                status = "nonfinite_state"
                break

            if not move_started:
                altitude_error = abs(float(cfg.HOVER_POSITION[2]) - pos[2])
                vertical_speed = abs(vel[2])
                if altitude_error < cfg.ALTITUDE_TOL and vertical_speed < cfg.VERTICAL_VEL_TOL:
                    if stable_start_time is None:
                        stable_start_time = float(data.time)
                    elif data.time - stable_start_time >= cfg.STABLE_DURATION:
                        move_started = True
                        move_start_time = float(data.time)
                else:
                    stable_start_time = None

            if move_started:
                target = get_target(trajectory, data.time - move_start_time, speed_scale)
            else:
                target = np.asarray(cfg.HOVER_POSITION, dtype=float)

            wrench = pid(
                pos_W=pos, quat_wxyz=quat, vel_W=vel, omega_B=omega,
                pos_des=target, yaw_des=cfg.YAW_DES,
            )
            thrust = np.clip(allocation_pinv @ wrench, THRUST_MIN, THRUST_MAX)
            data.ctrl[force_ids] = thrust
            data.ctrl[torque_ids] = MOTOR_DIR * K_TAU * thrust
            mujoco.mj_step(model, data)

            if data.time - last_log_time >= cfg.LOG_INTERVAL:
                # Read all states after mj_step at the same simulation step.
                pos_log = data.qpos[qa:qa + 3].copy()
                quat_log = data.qpos[qa + 3:qa + 7].copy()
                vel_log = data.qvel[va:va + 3].copy()

                if gyro_adr is not None:
                    omega_log = data.sensordata[gyro_adr:gyro_adr + gyro_dim].copy()
                else:
                    omega_log = data.qvel[va + 3:va + 6].copy()

                acc_log = data.sensordata[acc_adr:acc_adr + acc_dim].copy()

                payload_pos = data.xpos[payload_id].copy()

                swing_angle_x = float(data.qpos[swing_x_qadr])
                swing_angle_y = float(data.qpos[swing_y_qadr])
                swing_rate_x = float(data.qvel[swing_x_vadr])
                swing_rate_y = float(data.qvel[swing_y_vadr])

                row = [
                    episode_id, float(data.time), int(move_started), *target,
                    *pos_log, *vel_log, *quat_log, *omega_log, *acc_log, *payload_pos,
                    swing_angle_x, swing_angle_y, swing_rate_x, swing_rate_y,
                    *thrust, *wrench,
                ]
                writer.writerow(row)
                last_log_time = float(data.time)

            if pos[2] < -0.1:
                status = "possible_crash"
                break
            if realtime:
                import time
                time.sleep(cfg.DT)

    return {
        "episode_id": episode_id,
        "csv_path": str(csv_path),
        "status": status,
        "trajectory": trajectory,
        "duration_s": duration,
        "initial_swing_x_rad": initial_swing_x_rad,
        "initial_swing_y_rad": initial_swing_y_rad,
        "speed_scale": speed_scale,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", type=int, default=1)
    parser.add_argument("--duration", type=float, default=cfg.DEFAULT_DURATION)
    parser.add_argument("--trajectory", choices=cfg.TRAJECTORIES, default=cfg.DEFAULT_TRAJECTORY)
    parser.add_argument("--initial-swing-x-deg", type=float, default=0.0)
    parser.add_argument("--initial-swing-y-deg", type=float, default=0.0)
    parser.add_argument("--speed-scale", type=float, default=cfg.DEFAULT_SPEED_SCALE)
    parser.add_argument("--output-dir", type=str, default=str(cfg.OUTPUT_DIR))
    args = parser.parse_args()
    result = run_episode(
        episode_id=args.episode_id,
        output_dir=args.output_dir,
        duration=args.duration,
        trajectory=args.trajectory,
        initial_swing_x_rad=np.deg2rad(args.initial_swing_x_deg),
        initial_swing_y_rad=np.deg2rad(args.initial_swing_y_deg),
        speed_scale=args.speed_scale,
    )
    print(f"Episode {result['episode_id']}: {result['status']}")
    print(f"CSV saved: {result['csv_path']}")


if __name__ == "__main__":
    main()
