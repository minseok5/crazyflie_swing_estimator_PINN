#!/usr/bin/env python3
"""Standalone step-response test for the Crazyflie first-order actuator model.

Run from the repository root:
    python test_motor_dynamics.py

The test does not require MuJoCo. It writes a CSV and PNG under outputs/motor_test/.
It verifies the implemented model's numerical response, not physical calibration.
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
)
from config.simulation_config import DT
from simulation_core.motor_dynamics import Cf21bFirstOrderActuatorModel


DURATION_S = 9.0
OUTPUT_DIR = Path("outputs/motor_test")


def commanded_thrust(t: float) -> float:
    """Piecewise-constant test input: stop, low, high, low, stop."""
    if 1.0 <= t < 3.0:
        return 0.05
    if 3.0 <= t < 5.0:
        return 0.10
    if 5.0 <= t < 7.0:
        return 0.03
    return 0.0


def main() -> None:
    actuator = Cf21bFirstOrderActuatorModel(
        dt=DT,
        motor_direction=MOTOR_DIRECTION,
        thrust_min=THRUST_MIN,
        thrust_max=THRUST_MAX,
        legacy_ratio_m=K_TAU,
        time_constant_s=MOTOR_TIME_CONSTANT,
        steady_state_gain_rad_s=OMEGA_REFERENCE,
        thrust_polynomial_coefficients=THRUST_POLYNOMIAL,
        reaction_torque_model=REACTION_TORQUE_MODEL,
    )
    actuator.reset(airborne=False)

    n_steps = int(round(DURATION_S / DT))
    rows: list[dict[str, float]] = []

    for step in range(n_steps):
        t = step * DT
        f_request = np.full(4, commanded_thrust(t), dtype=float)
        out = actuator.apply(f_request)

        row: dict[str, float] = {
            "time_s": (step + 1) * DT,
            "thrust_request_N": float(f_request[0]),
        }
        for i in range(4):
            row[f"motor{i}_thrust_command_N"] = float(out.f_cmd[i])
            row[f"motor{i}_thrust_actual_N"] = float(out.f_actual[i])
            row[f"motor{i}_omega_rad_s"] = float(out.omega[i])
            row[f"motor{i}_command_normalized"] = float(out.motor_command[i])
            row[f"motor{i}_reaction_torque_Nm"] = float(out.q_actual[i])
        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"motor_step_response_{stamp}.csv"
    png_path = OUTPUT_DIR / f"motor_step_response_{stamp}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    t = np.asarray([r["time_s"] for r in rows])
    request = np.asarray([r["thrust_request_N"] for r in rows])
    thrust = np.column_stack([
        [r[f"motor{i}_thrust_actual_N"] for r in rows] for i in range(4)
    ])
    omega = np.column_stack([
        [r[f"motor{i}_omega_rad_s"] for r in rows] for i in range(4)
    ])
    torque = np.column_stack([
        [r[f"motor{i}_reaction_torque_Nm"] for r in rows] for i in range(4)
    ])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].step(t, request, where="post", linestyle="--", label="requested thrust")
    for i in range(4):
        axes[0].plot(t, thrust[:, i], label=f"motor {i} actual")
    axes[0].set_ylabel("Thrust (N)")
    axes[0].set_title("Motor thrust step response")
    axes[0].grid(True)
    axes[0].legend(ncol=3)

    for i in range(4):
        axes[1].plot(t, omega[:, i], label=f"motor {i}")
    axes[1].set_ylabel("Rotor speed (rad/s)")
    axes[1].set_title("Rotor angular velocity")
    axes[1].grid(True)
    axes[1].legend(ncol=4)

    for i in range(4):
        axes[2].plot(t, torque[:, i], label=f"motor {i}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Reaction torque (N m)")
    axes[2].set_title(f"Reaction torque ({REACTION_TORQUE_MODEL})")
    axes[2].grid(True)
    axes[2].legend(ncol=4)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"CSV saved: {csv_path}")
    print(f"Graph saved: {png_path}")
    print(f"dt={DT:.6f} s, tau={MOTOR_TIME_CONSTANT:.6f} s")
    print(f"thrust limits=[{THRUST_MIN:.4f}, {THRUST_MAX:.4f}] N")
    print(f"reaction torque model={REACTION_TORQUE_MODEL}")

    # Basic implementation checks.
    actual_all = thrust
    omega_all = omega
    assert np.all(np.isfinite(actual_all)), "Non-finite thrust detected"
    assert np.all(np.isfinite(omega_all)), "Non-finite rotor speed detected"
    assert np.all(actual_all >= -1e-12), "Negative actual thrust detected"
    assert np.all(actual_all <= THRUST_MAX + 1e-9), "Thrust exceeds configured maximum"
    assert np.all(omega_all >= -1e-12), "Negative rotor speed detected"
    print("Basic numerical checks: PASS")


if __name__ == "__main__":
    main()
