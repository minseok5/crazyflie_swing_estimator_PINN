#!/usr/bin/env python3
"""Headless A/B hover comparison: legacy_ratio vs paper_polynomial.

Run from the repository root:
    source .venv/bin/activate
    python compare_reaction_torque_hover.py

Both runs use identical initial conditions, PID gains, motor dynamics,
payload, timestep, and hover target. Only the reaction-torque law changes.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from config.simulation_config import (
    MODEL_PATH,
    DT,
    GRAVITY,
)
from config.vehicle_config import (
    DRONE_MASS,
    DRONE_INERTIA,
    INITIAL_DRONE_POSITION,
    HOVER_POSITION,
)
from config.actuator_config import (
    MOTOR_DIRECTION,
    THRUST_MIN,
    THRUST_MAX,
    K_TAU,
    MOTOR_TIME_CONSTANT,
    OMEGA_REFERENCE,
    THRUST_POLYNOMIAL,
    ENABLE_REACTION_TORQUE,
    TORQUE_POLYNOMIAL,
    TORQUE_POLYNOMIAL_SCALE,
    ROTOR_INERTIA,
    ENABLE_ROTOR_ACCELERATION_TORQUE,
)
from simulation_core.motor_dynamics import Cf21bFirstOrderActuatorModel
from simulation_core.pid_controller import CascadePID, build_allocation_matrix


# Keep consistent with the current payload simulation configuration.
PAYLOAD_MASS_KG = 0.010
DURATION_S = 30.0
LOG_INTERVAL_S = 0.01
REACTION_MODELS = ("legacy_ratio", "paper_polynomial")
OUTPUT_ROOT = Path("outputs/reaction_torque_ab")


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return idx


def quaternion_to_euler_wxyz(q):
    qw, qx, qy, qz = np.asarray(q, dtype=float)
    roll = np.arctan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    sin_pitch = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return np.array([roll, pitch, yaw])


def run_case(reaction_model: str, run_dir: Path):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = float(DT)
    data = mujoco.MjData(model)

    # Match the existing simulation's first-free-joint initialization.
    free_id = next(
        (i for i in range(model.njnt)
         if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE),
        None,
    )
    if free_id is None:
        raise RuntimeError("No free joint found in the MuJoCo model.")
    qadr = int(model.jnt_qposadr[free_id])
    vadr = int(model.jnt_dofadr[free_id])
    data.qpos[qadr:qadr + 3] = np.asarray(INITIAL_DRONE_POSITION, dtype=float)
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[vadr:vadr + 6] = 0.0

    payload_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_body")
    swing_x_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing_joint_x")
    swing_y_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing_joint_y")
    swing_x_qadr = int(model.jnt_qposadr[swing_x_id])
    swing_y_qadr = int(model.jnt_qposadr[swing_y_id])
    swing_x_vadr = int(model.jnt_dofadr[swing_x_id])
    swing_y_vadr = int(model.jnt_dofadr[swing_y_id])

    force_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_force")
        for i in range(4)
    ])
    torque_ids = np.array([
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_torque")
        for i in range(4)
    ])
    gyro_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    gyro_adr = int(model.sensor_adr[gyro_id])
    gyro_dim = int(model.sensor_dim[gyro_id])

    vehicle = SimpleNamespace(
        mass=float(DRONE_MASS) + PAYLOAD_MASS_KG,
        gravity=float(GRAVITY),
        inertia_diagonal=DRONE_INERTIA,
    )
    pid = CascadePID(dt=DT, vehicle=vehicle)
    allocation_matrix, allocation_pinv = build_allocation_matrix()

    actuator = Cf21bFirstOrderActuatorModel(
        dt=DT,
        motor_direction=MOTOR_DIRECTION,
        thrust_min=THRUST_MIN,
        thrust_max=THRUST_MAX,
        legacy_ratio_m=K_TAU,
        time_constant_s=MOTOR_TIME_CONSTANT,
        steady_state_gain_rad_s=OMEGA_REFERENCE,
        thrust_polynomial_coefficients=THRUST_POLYNOMIAL,
        reaction_torque_model=reaction_model,
        torque_polynomial_coefficients=TORQUE_POLYNOMIAL,
        torque_polynomial_scale=TORQUE_POLYNOMIAL_SCALE,
        rotor_inertia_kg_m2=ROTOR_INERTIA,
        include_rotor_acceleration_torque=ENABLE_ROTOR_ACCELERATION_TORQUE,
    )
    actuator.reset(airborne=False)
    mujoco.mj_forward(model, data)

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{reaction_model}.csv"
    fields = [
        "time_s",
        "drone_x_m", "drone_y_m", "drone_z_m",
        "drone_vx_mps", "drone_vy_mps", "drone_vz_mps",
        "roll_rad", "pitch_rad", "yaw_rad",
        "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
        "payload_x_m", "payload_y_m", "payload_z_m",
        "swing_angle_x_rad", "swing_angle_y_rad",
        "swing_rate_x_radps", "swing_rate_y_radps",
        "target_x_m", "target_y_m", "target_z_m",
        "commanded_roll_torque_Nm", "commanded_pitch_torque_Nm",
        "commanded_yaw_torque_Nm", "commanded_collective_N",
        "actual_roll_moment_Nm", "actual_pitch_moment_Nm",
        "actual_net_reaction_torque_Nm", "actual_collective_N",
        "motor0_thrust_N", "motor1_thrust_N",
        "motor2_thrust_N", "motor3_thrust_N",
        "motor0_reaction_torque_Nm", "motor1_reaction_torque_Nm",
        "motor2_reaction_torque_Nm", "motor3_reaction_torque_Nm",
    ]

    rows = []
    next_log_time = 0.0
    n_steps = int(round(DURATION_S / DT))
    target = np.asarray(HOVER_POSITION, dtype=float)

    for step in range(n_steps):
        pos = data.qpos[qadr:qadr + 3].copy()
        quat = data.qpos[qadr + 3:qadr + 7].copy()
        vel = data.qvel[vadr:vadr + 3].copy()
        omega = data.sensordata[gyro_adr:gyro_adr + gyro_dim].copy()

        wrench_cmd = pid(
            pos_W=pos,
            quat_wxyz=quat,
            vel_W=vel,
            omega_B=omega,
            pos_des=target,
            yaw_des=0.0,
        )
        thrust_cmd = np.clip(
            allocation_pinv @ wrench_cmd,
            THRUST_MIN,
            THRUST_MAX,
        )
        actuator_output = actuator.apply(thrust_cmd)
        thrust_actual = actuator_output.f_actual
        torque_actual = actuator_output.q_actual.copy()
        if not ENABLE_REACTION_TORQUE:
            torque_actual[:] = 0.0

        data.ctrl[force_ids] = thrust_actual
        data.ctrl[torque_ids] = torque_actual
        mujoco.mj_step(model, data)

        if data.time + 1e-12 >= next_log_time:
            pos_now = data.qpos[qadr:qadr + 3].copy()
            quat_now = data.qpos[qadr + 3:qadr + 7].copy()
            vel_now = data.qvel[vadr:vadr + 3].copy()
            gyro_now = data.sensordata[gyro_adr:gyro_adr + gyro_dim].copy()
            euler = quaternion_to_euler_wxyz(quat_now)
            payload_pos = data.xpos[payload_id].copy()

            # The first two rows of the allocation matrix give roll/pitch
            # moments from actual thrust. Yaw reaction torque is summed from
            # the actual per-motor torque outputs, not the legacy allocation.
            actual_rp = allocation_matrix[:2, :] @ thrust_actual
            row = {
                "time_s": float(data.time),
                "drone_x_m": float(pos_now[0]),
                "drone_y_m": float(pos_now[1]),
                "drone_z_m": float(pos_now[2]),
                "drone_vx_mps": float(vel_now[0]),
                "drone_vy_mps": float(vel_now[1]),
                "drone_vz_mps": float(vel_now[2]),
                "roll_rad": float(euler[0]),
                "pitch_rad": float(euler[1]),
                "yaw_rad": float(euler[2]),
                "gyro_x_radps": float(gyro_now[0]),
                "gyro_y_radps": float(gyro_now[1]),
                "gyro_z_radps": float(gyro_now[2]),
                "payload_x_m": float(payload_pos[0]),
                "payload_y_m": float(payload_pos[1]),
                "payload_z_m": float(payload_pos[2]),
                "swing_angle_x_rad": float(data.qpos[swing_x_qadr]),
                "swing_angle_y_rad": float(data.qpos[swing_y_qadr]),
                "swing_rate_x_radps": float(data.qvel[swing_x_vadr]),
                "swing_rate_y_radps": float(data.qvel[swing_y_vadr]),
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "commanded_roll_torque_Nm": float(wrench_cmd[0]),
                "commanded_pitch_torque_Nm": float(wrench_cmd[1]),
                "commanded_yaw_torque_Nm": float(wrench_cmd[2]),
                "commanded_collective_N": float(wrench_cmd[3]),
                "actual_roll_moment_Nm": float(actual_rp[0]),
                "actual_pitch_moment_Nm": float(actual_rp[1]),
                "actual_net_reaction_torque_Nm": float(np.sum(torque_actual)),
                "actual_collective_N": float(np.sum(thrust_actual)),
            }
            for i in range(4):
                row[f"motor{i}_thrust_N"] = float(thrust_actual[i])
                row[f"motor{i}_reaction_torque_Nm"] = float(torque_actual[i])
            rows.append(row)
            next_log_time += LOG_INTERVAL_S

        state = np.concatenate([
            data.qpos.copy(), data.qvel.copy(), data.sensordata.copy()
        ])
        if not np.all(np.isfinite(state)):
            print(f"[{reaction_model}] Non-finite state at t={data.time:.3f}s")
            break
        if pos_now_z(data, qadr) < 0.05:
            print(f"[{reaction_model}] Drone below 0.05 m at t={data.time:.3f}s")
            break

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{reaction_model}] CSV saved: {csv_path}")
    print(f"[{reaction_model}] simulated duration: {data.time:.3f} s")
    return rows, csv_path


def pos_now_z(data, qadr):
    return float(data.qpos[qadr + 2])


def summarize(rows, model_name):
    if not rows:
        return {"reaction_model": model_name, "samples": 0}
    t = np.asarray([r["time_s"] for r in rows])
    z = np.asarray([r["drone_z_m"] for r in rows])
    pos_error = np.column_stack([
        np.asarray([r["target_x_m"] - r["drone_x_m"] for r in rows]),
        np.asarray([r["target_y_m"] - r["drone_y_m"] for r in rows]),
        np.asarray([r["target_z_m"] - r["drone_z_m"] for r in rows]),
    ])
    roll = np.asarray([r["roll_rad"] for r in rows])
    pitch = np.asarray([r["pitch_rad"] for r in rows])
    yaw = np.unwrap(np.asarray([r["yaw_rad"] for r in rows]))
    swing_x = np.asarray([r["swing_angle_x_rad"] for r in rows])
    swing_y = np.asarray([r["swing_angle_y_rad"] for r in rows])
    swing_mag = np.sqrt(swing_x**2 + swing_y**2)
    start = t >= max(0.0, t[-1] - 5.0)
    return {
        "reaction_model": model_name,
        "samples": len(rows),
        "simulated_duration_s": float(t[-1]),
        "mean_position_error_last5s_m": float(np.mean(np.linalg.norm(pos_error[start], axis=1))),
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(roll)))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(pitch)))),
        "yaw_change_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
        "payload_swing_peak_deg": float(np.rad2deg(np.max(swing_mag))),
        "payload_swing_rms_deg": float(np.rad2deg(np.sqrt(np.mean(swing_mag**2)))),
        "altitude_min_m": float(np.min(z)),
        "altitude_max_m": float(np.max(z)),
    }


def make_comparison_plot(all_rows, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.ravel()
    specs = [
        ("Drone altitude", "drone_z_m", "Altitude (m)"),
        ("Roll", "roll_rad", "Angle (deg)"),
        ("Pitch", "pitch_rad", "Angle (deg)"),
        ("Yaw rate", "gyro_z_radps", "Angular velocity (rad/s)"),
        ("Payload swing magnitude", None, "Angle (deg)"),
        ("Net reaction torque", "actual_net_reaction_torque_Nm", "Torque (N m)"),
    ]
    for name, rows in all_rows.items():
        t = np.asarray([r["time_s"] for r in rows])
        for ax, (title, key, ylabel) in zip(axes, specs):
            if key is None:
                sx = np.asarray([r["swing_angle_x_rad"] for r in rows])
                sy = np.asarray([r["swing_angle_y_rad"] for r in rows])
                values = np.rad2deg(np.sqrt(sx**2 + sy**2))
            else:
                values = np.asarray([r[key] for r in rows])
                if key in ("roll_rad", "pitch_rad"):
                    values = np.rad2deg(values)
            ax.plot(t, values, label=name)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
    for ax in axes[-2:]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Reaction-torque A/B hover comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"MuJoCo model not found: {MODEL_PATH}. "
            "Generate/check the payload XML before running."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL_PATH}")
    print(f"Payload mass configured in test: {PAYLOAD_MASS_KG:.3f} kg")
    print(f"Rope/model geometry: taken from {MODEL_PATH}")
    print(f"Duration target: {DURATION_S:.1f} s; DT={DT:.6f} s")
    print("Only the reaction-torque law changes between A and B.")
    print("A: legacy_ratio")
    print("B: paper_polynomial")

    all_rows = {}
    summaries = []
    for reaction_model in REACTION_MODELS:
        rows, _ = run_case(reaction_model, run_dir / reaction_model)
        all_rows[reaction_model] = rows
        summaries.append(summarize(rows, reaction_model))

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    plot_path = run_dir / "comparison.png"
    make_comparison_plot(all_rows, plot_path)

    print(f"\nSummary saved: {summary_path}")
    print(f"Comparison graph saved: {plot_path}")
    print("\nA/B summary:")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
