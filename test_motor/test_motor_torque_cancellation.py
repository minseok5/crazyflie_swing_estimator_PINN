#!/usr/bin/env python3
"""Four-motor reaction-torque cancellation test.
Run from the repository root:
    source .venv/bin/activate
    python test_motor_torque_cancellation.py
"""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.actuator_config import (
    MOTOR_COUNT, MOTOR_DIRECTION, THRUST_MIN, THRUST_MAX, K_TAU,
    MOTOR_TIME_CONSTANT, OMEGA_REFERENCE, THRUST_POLYNOMIAL,
    REACTION_TORQUE_MODEL, TORQUE_POLYNOMIAL, TORQUE_POLYNOMIAL_SCALE,
    ROTOR_INERTIA, ENABLE_ROTOR_ACCELERATION_TORQUE,
)
from config.simulation_config import DT
from simulation_core.motor_dynamics import Cf21bFirstOrderActuatorModel

DURATION_S = 8.0
OUTPUT_DIR = Path("outputs/motor_test/torque_cancellation")


def command(t):
    if 1.0 <= t < 3.0:
        return 0.05
    if 3.0 <= t < 5.0:
        return 0.10
    if 5.0 <= t < 7.0:
        return 0.03
    return 0.0


def main():
    directions = np.asarray(MOTOR_DIRECTION, dtype=float)
    if MOTOR_COUNT != 4 or directions.shape != (4,):
        raise RuntimeError("This test expects exactly four motors.")
    if not np.isclose(directions.sum(), 0.0):
        raise RuntimeError(f"Motor directions do not balance: {directions}")

    actuator = Cf21bFirstOrderActuatorModel(
        dt=DT, motor_direction=directions,
        thrust_min=THRUST_MIN, thrust_max=THRUST_MAX,
        legacy_ratio_m=K_TAU, time_constant_s=MOTOR_TIME_CONSTANT,
        steady_state_gain_rad_s=OMEGA_REFERENCE,
        thrust_polynomial_coefficients=THRUST_POLYNOMIAL,
        reaction_torque_model=REACTION_TORQUE_MODEL,
        torque_polynomial_coefficients=TORQUE_POLYNOMIAL,
        torque_polynomial_scale=TORQUE_POLYNOMIAL_SCALE,
        rotor_inertia_kg_m2=ROTOR_INERTIA,
        include_rotor_acceleration_torque=ENABLE_ROTOR_ACCELERATION_TORQUE,
    )
    actuator.reset(airborne=False)

    rows = []
    max_net = 0.0
    max_pair = 0.0
    for step in range(int(round(DURATION_S / DT))):
        t = step * DT
        f = command(t)
        out = actuator.apply(np.full(4, f))
        net = float(np.sum(out.q_actual))
        max_net = max(max_net, abs(net))
        max_pair = max(
            max_pair,
            abs(float(out.q_actual[0] + out.q_actual[1])),
            abs(float(out.q_actual[2] + out.q_actual[3])),
        )
        row = {
            "time_s": (step + 1) * DT,
            "thrust_command_per_motor_N": f,
            "net_reaction_torque_Nm": net,
        }
        for i in range(4):
            row[f"motor{i}_thrust_N"] = float(out.f_actual[i])
            row[f"motor{i}_omega_rad_s"] = float(out.omega[i])
            row[f"motor{i}_reaction_torque_Nm"] = float(out.q_actual[i])
        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"torque_cancellation_{stamp}.csv"
    png_path = OUTPUT_DIR / f"torque_cancellation_{stamp}.png"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    time = np.array([r["time_s"] for r in rows])
    individual = np.column_stack([
        [r[f"motor{i}_reaction_torque_Nm"] for r in rows]
        for i in range(4)
    ])
    net = np.array([r["net_reaction_torque_Nm"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(4):
        axes[0].plot(time, individual[:, i], label=f"motor {i}")
    axes[0].set_title(f"Individual reaction torques ({REACTION_TORQUE_MODEL})")
    axes[0].set_ylabel("Torque (N m)")
    axes[0].grid(True)
    axes[0].legend(ncol=4)
    axes[1].plot(time, net, label="sum of four motor torques")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_title("Net reaction torque")
    axes[1].set_ylabel("Torque (N m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"CSV saved: {csv_path}")
    print(f"Graph saved: {png_path}")
    print(f"Reaction torque model: {REACTION_TORQUE_MODEL}")
    print(f"Motor directions: {directions.tolist()}")
    print(f"Maximum |sum torque|: {max_net:.3e} N m")
    print(f"Maximum |opposite-pair sum|: {max_pair:.3e} N m")
    if max_net > 1e-10 or max_pair > 1e-10:
        raise SystemExit("Four-motor torque cancellation: FAIL")
    print("Four-motor reaction-torque cancellation: PASS")


if __name__ == "__main__":
    main()
