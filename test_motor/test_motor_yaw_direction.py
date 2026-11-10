#!/usr/bin/env python3
"""Isolated MuJoCo yaw-torque sign test for the four Crazyflie motors.

Run from the repository root:
    source .venv/bin/activate
    python test_motor_yaw_direction.py

The test applies balanced collective thrust and a short torque pulse to one
motor at a time. It then checks the sign of the IMU gyro's Z component.
This verifies the sign convention of the MuJoCo actuator chain; it does not
by itself prove the physical propeller spin direction.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_XML = "models/crazyflie/cf21B_500_payload.xml"
MOTOR_DIRECTION = (1.0, -1.0, 1.0, -1.0)
TORQUE_PULSE_NM = 2.0e-4
SETTLE_TIME_S = 0.15
PULSE_TIME_S = 0.15
POST_PULSE_TIME_S = 0.10
HOVER_MARGIN = 1.0


def name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return idx


def run_one_motor(xml_path: Path, motor_index: int, direction: float):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    force_ids = [
        name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_force")
        for i in range(4)
    ]
    torque_ids = [
        name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_torque")
        for i in range(4)
    ]
    gyro_id = name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")

    drone_id = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    total_mass = float(model.body_subtreemass[drone_id])
    collective = total_mass * abs(float(model.opt.gravity[2])) * HOVER_MARGIN
    per_motor_thrust = collective / 4.0

    dt = float(model.opt.timestep)
    n_settle = int(round(SETTLE_TIME_S / dt))
    n_pulse = int(round(PULSE_TIME_S / dt))
    n_post = int(round(POST_PULSE_TIME_S / dt))

    rows = []

    def set_controls(torque_value: float):
        data.ctrl[:] = 0.0
        for aid in force_ids:
            data.ctrl[aid] = per_motor_thrust
        data.ctrl[torque_ids[motor_index]] = torque_value

    def record(phase: str):
        gyro_adr = int(model.sensor_adr[gyro_id])
        gyro = data.sensordata[gyro_adr:gyro_adr + 3].copy()
        rows.append({
            "time_s": float(data.time),
            "phase": phase,
            "gyro_x_rad_s": float(gyro[0]),
            "gyro_y_rad_s": float(gyro[1]),
            "gyro_z_rad_s": float(gyro[2]),
            "selected_motor": motor_index,
            "applied_torque_Nm": direction * TORQUE_PULSE_NM
                if phase == "pulse" else 0.0,
            "collective_thrust_N": collective,
        })

    # Settle with balanced thrust and zero yaw torque.
    for _ in range(n_settle):
        set_controls(0.0)
        mujoco.mj_step(model, data)
        record("settle")

    # Apply only the selected motor's signed reaction torque.
    signed_torque = direction * TORQUE_PULSE_NM
    for _ in range(n_pulse):
        set_controls(signed_torque)
        mujoco.mj_step(model, data)
        record("pulse")

    # Remove torque and observe the resulting angular-rate evolution.
    for _ in range(n_post):
        set_controls(0.0)
        mujoco.mj_step(model, data)
        record("post_pulse")

    pulse_rows = [r for r in rows if r["phase"] == "pulse"]
    # Compare gyro z near the beginning and end of the pulse. The short
    # interval minimizes contamination from position/rope motion.
    z_start = pulse_rows[min(2, len(pulse_rows) - 1)]["gyro_z_rad_s"]
    z_end = pulse_rows[-1]["gyro_z_rad_s"]
    delta_z = z_end - z_start

    return rows, {
        "motor": motor_index,
        "direction": direction,
        "applied_torque_Nm": signed_torque,
        "gyro_z_start_rad_s": z_start,
        "gyro_z_end_rad_s": z_end,
        "delta_gyro_z_rad_s": delta_z,
        "expected_sign": int(np.sign(signed_torque)),
        "observed_sign": int(np.sign(delta_z)),
        "sign_match": bool(np.sign(delta_z) == np.sign(signed_torque)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=DEFAULT_XML, help="Path to MuJoCo XML")
    parser.add_argument("--output", default="outputs/motor_test/yaw_direction")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML not found: {xml_path}. Run from project root or pass --xml."
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for i, direction in enumerate(MOTOR_DIRECTION):
        rows, summary = run_one_motor(xml_path, i, direction)
        all_rows.extend(rows)
        summaries.append(summary)

    csv_path = output_dir / "motor_yaw_direction_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"XML: {xml_path}")
    print(f"CSV saved: {csv_path}")
    print(f"Torque pulse magnitude: {TORQUE_PULSE_NM:.3e} N m")
    print()
    print("Motor | Direction | Applied torque | Delta gyro-z | Sign check")
    for s in summaries:
        print(
            f"{s['motor']:>5} | {s['direction']:>9.1f} | "
            f"{s['applied_torque_Nm']:>14.3e} | "
            f"{s['delta_gyro_z_rad_s']:>12.6e} | "
            f"{'PASS' if s['sign_match'] else 'CHECK'}"
        )

    if not all(s["sign_match"] for s in summaries):
        raise SystemExit(
            "At least one motor's gyro-z response sign did not match the "
            "applied torque sign. Inspect the CSV before changing signs."
        )

    print("\nMuJoCo torque-actuator sign checks: PASS")
    print(
        "Note: PASS confirms actuator/gyro sign consistency only; "
        "physical propeller spin direction still requires authoritative "
        "motor/propeller mapping or hardware evidence."
    )


if __name__ == "__main__":
    main()
