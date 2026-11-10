#!/usr/bin/env python3
"""Reaction-torque verification for the configured Crazyflie actuator model.

Run from the repository root:
    python test_motor_reaction_torque.py

This is a numerical implementation test, not a physical calibration test.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.actuator_config import (
    MOTOR_DIRECTION,
    THRUST_MIN,
    THRUST_MAX,
    K_TAU,
    MOTOR_TIME_CONSTANT,
    OMEGA_REFERENCE,
    THRUST_POLYNOMIAL,
    REACTION_TORQUE_MODEL,
    TORQUE_POLYNOMIAL,
    TORQUE_POLYNOMIAL_SCALE,
    ROTOR_INERTIA,
    ENABLE_ROTOR_ACCELERATION_TORQUE,
)
from config.simulation_config import DT
from simulation_core.motor_dynamics import Cf21bFirstOrderActuatorModel


DURATION_S = 9.0
OUTPUT_DIR = Path("outputs/motor_test")


def commanded_thrust(t: float) -> float:
    if 1.0 <= t < 3.0:
        return 0.05
    if 3.0 <= t < 5.0:
        return 0.10
    if 5.0 <= t < 7.0:
        return 0.03
    return 0.0


def main() -> None:
    direction = np.asarray(MOTOR_DIRECTION, dtype=float)
    actuator = Cf21bFirstOrderActuatorModel(
        dt=DT,
        motor_direction=direction,
        thrust_min=THRUST_MIN,
        thrust_max=THRUST_MAX,
        legacy_ratio_m=K_TAU,
        time_constant_s=MOTOR_TIME_CONSTANT,
        steady_state_gain_rad_s=OMEGA_REFERENCE,
        thrust_polynomial_coefficients=THRUST_POLYNOMIAL,
        reaction_torque_model=REACTION_TORQUE_MODEL,
        torque_polynomial_coefficients=TORQUE_POLYNOMIAL,
        torque_polynomial_scale=TORQUE_POLYNOMIAL_SCALE,
        rotor_inertia_kg_m2=ROTOR_INERTIA,
        include_rotor_acceleration_torque=ENABLE_ROTOR_ACCELERATION_TORQUE,
    )
    actuator.reset(airborne=False)

    n_steps = int(round(DURATION_S / DT))
    rows = []
    previous_omega = np.zeros(4)

    for step in range(n_steps):
        t = step * DT
        f_request = np.full(4, commanded_thrust(t), dtype=float)
        out = actuator.apply(f_request)

        omega_dot = (out.omega - previous_omega) / DT
        ratio = np.clip(out.omega / OMEGA_REFERENCE, 0.0, 1.0)
        a3, a2, a1 = TORQUE_POLYNOMIAL
        aero = TORQUE_POLYNOMIAL_SCALE * (
            a3 * ratio**3 + a2 * ratio**2 + a1 * ratio
        )
        aero = np.maximum(aero, 0.0)
        inertia_component = ROTOR_INERTIA * omega_dot if ENABLE_ROTOR_ACCELERATION_TORQUE else np.zeros(4)
        expected_total = direction * (aero + inertia_component)

        row = {
            "time_s": (step + 1) * DT,
            "thrust_request_N": float(f_request[0]),
        }
        for i in range(4):
            row[f"motor{i}_thrust_N"] = float(out.f_actual[i])
            row[f"motor{i}_omega_rad_s"] = float(out.omega[i])
            row[f"motor{i}_omega_dot_rad_s2"] = float(omega_dot[i])
            row[f"motor{i}_aero_torque_Nm"] = float(direction[i] * aero[i])
            row[f"motor{i}_inertia_torque_Nm"] = float(direction[i] * inertia_component[i])
            row[f"motor{i}_reaction_torque_Nm"] = float(out.q_actual[i])
            row[f"motor{i}_expected_torque_Nm"] = float(expected_total[i])

        rows.append(row)
        previous_omega = out.omega.copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"reaction_torque_test_{stamp}.csv"
    png_path = OUTPUT_DIR / f"reaction_torque_test_{stamp}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    t = np.asarray([r["time_s"] for r in rows])
    omega = np.column_stack([[r[f"motor{i}_omega_rad_s"] for r in rows] for i in range(4)])
    total = np.column_stack([[r[f"motor{i}_reaction_torque_Nm"] for r in rows] for i in range(4)])
    aero = np.column_stack([[r[f"motor{i}_aero_torque_Nm"] for r in rows] for i in range(4)])
    inertia = np.column_stack([[r[f"motor{i}_inertia_torque_Nm"] for r in rows] for i in range(4)])

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    for i in range(4):
        axes[0].plot(t, total[:, i], label=f"motor {i}")
    axes[0].set_title(f"Total reaction torque ({REACTION_TORQUE_MODEL})")
    axes[0].set_ylabel("Torque (N m)")
    axes[0].grid(True)
    axes[0].legend(ncol=4)

    for i in range(4):
        axes[1].plot(t, aero[:, i], label=f"motor {i}")
    axes[1].set_title("Aerodynamic / friction torque component")
    axes[1].set_ylabel("Torque (N m)")
    axes[1].grid(True)
    axes[1].legend(ncol=4)

    for i in range(4):
        axes[2].plot(t, inertia[:, i], label=f"motor {i}")
    axes[2].set_title("Rotor acceleration torque component")
    axes[2].set_ylabel("Torque (N m)")
    axes[2].grid(True)
    axes[2].legend(ncol=4)

    for i in range(4):
        axes[3].plot(t, omega[:, i], label=f"motor {i}")
    axes[3].set_title("Rotor angular velocity")
    axes[3].set_ylabel("rad/s")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True)
    axes[3].legend(ncol=4)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    actual = np.column_stack([[r[f"motor{i}_reaction_torque_Nm"] for r in rows] for i in range(4)])
    expected = np.column_stack([[r[f"motor{i}_expected_torque_Nm"] for r in rows] for i in range(4)])
    max_error = float(np.max(np.abs(actual - expected)))

    print(f"CSV saved: {csv_path}")
    print(f"Graph saved: {png_path}")
    print(f"Reaction torque model: {REACTION_TORQUE_MODEL}")
    print(f"Rotor acceleration torque enabled: {ENABLE_ROTOR_ACCELERATION_TORQUE}")
    print(f"Rotor inertia: {ROTOR_INERTIA:.3e} kg m^2")
    print(f"Maximum implementation consistency error: {max_error:.3e} N m")

    assert np.all(np.isfinite(actual)), "Non-finite reaction torque detected"
    assert max_error < 1e-10, "Reaction torque does not match configured equation"
    assert np.allclose(total[:, 0], -total[:, 1], atol=1e-12), "Motor 0/1 signs are not opposite"
    assert np.allclose(total[:, 0], total[:, 2], atol=1e-12), "Motor 0/2 responses differ"
    assert np.allclose(total[:, 1], total[:, 3], atol=1e-12), "Motor 1/3 responses differ"
    print("Reaction torque equation and motor-direction checks: PASS")


if __name__ == "__main__":
    main()
