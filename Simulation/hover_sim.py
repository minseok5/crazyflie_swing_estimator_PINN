import csv
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from simulation_core.pid_controller import (
    CascadePID,
    MASS,
    GRAV,
    K_TAU,
    MOTOR_DIR,
    THRUST_MIN,
    THRUST_MAX,
    PHYSICS_HZ,
    build_allocation_matrix,
    rotmat_from_quat_wxyz,
)


# --------------------------------------------------
# Simulation settings
# --------------------------------------------------
MODEL_PATH = Path("models/crazyflie/cf21B_500.xml")

SIM_DURATION = 10.0
DT = 1.0 / PHYSICS_HZ

# Target hover position [m]
POS_DES = np.array([0.0, 0.0, 0.5])
YAW_DES = 0.0

# Initial drone position [m]
INITIAL_POS = np.array([0.0, 0.0, 0.5])


def get_id(model, object_type, name):
    """Return the MuJoCo ID associated with a named object."""
    object_id = mujoco.mj_name2id(model, object_type, name)

    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")

    return object_id


def main():
    # --------------------------------------------------
    # 1. Load the existing Crazyflie model
    # --------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Keep the model's existing timestep.
    model.opt.timestep = DT

    # Identify the free joint controlling the drone body.
    free_joint_id = -1

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = joint_id
            break

    if free_joint_id < 0:
        raise RuntimeError("No free joint was found in the MuJoCo model.")

    qpos_adr = model.jnt_qposadr[free_joint_id]
    qvel_adr = model.jnt_dofadr[free_joint_id]

    # --------------------------------------------------
    # 2. Initialize the drone at the hover target
    # --------------------------------------------------
    mujoco.mj_resetData(model, data)

    data.qpos[qpos_adr:qpos_adr + 3] = INITIAL_POS

    # MuJoCo free-joint quaternion order: w, x, y, z.
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [
        1.0, 0.0, 0.0, 0.0
    ]

    data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)

    # --------------------------------------------------
    # 3. Initialize the PID controller
    # --------------------------------------------------
    pid = CascadePID(dt=DT)
    pid.reset()

    # Allocation matrix maps:
    # [torque_x, torque_y, torque_z, collective_thrust]
    # to the four individual motor thrusts.
    allocation, allocation_pinv = build_allocation_matrix()

    # Match the motor names and ordering used in the existing project.
    force_ids = [
        get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"motor{i}_force",
        )
        for i in range(4)
    ]

    torque_ids = [
        get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"motor{i}_torque",
        )
        for i in range(4)
    ]

    # Find a gyro sensor if the model provides one.
    gyro_sensor_id = -1

    for sensor_id in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
        )

        if sensor_name and "gyro" in sensor_name.lower():
            gyro_sensor_id = sensor_id
            break

    if gyro_sensor_id >= 0:
        gyro_adr = model.sensor_adr[gyro_sensor_id]
        gyro_dim = model.sensor_dim[gyro_sensor_id]

        if gyro_dim != 3:
            raise RuntimeError("The identified gyro sensor is not 3D.")

        print(
            "Using gyro sensor:",
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_sensor_id
            ),
        )
    else:
        gyro_adr = None
        print("No gyro sensor found; using angular velocity from qvel.")

    # --------------------------------------------------
    # 4. Run the simulation
    # --------------------------------------------------
    n_steps = int(SIM_DURATION / DT)
    log = []

    print("\nStarting standalone Crazyflie hover simulation")
    print(f"Simulation duration : {SIM_DURATION:.2f} s")
    print(f"Physics timestep    : {DT:.4f} s")
    print(f"Target position     : {POS_DES}")
    print("Motor delay         : disabled\n")

    for step in range(n_steps):
        # Read the free-joint state.
        pos_W = data.qpos[qpos_adr:qpos_adr + 3].copy()
        quat_wxyz = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
        vel_W = data.qvel[qvel_adr:qvel_adr + 3].copy()

        rotation = rotmat_from_quat_wxyz(quat_wxyz)

        if gyro_adr is not None:
            # Gyro sensors attached to the body report body-frame
            # angular velocity.
            omega_B = data.sensordata[
                gyro_adr:gyro_adr + 3
            ].copy()
        else:
            # MuJoCo free-joint rotational velocity is expressed
            # in the local frame.
            omega_B = data.qvel[
                qvel_adr + 3:qvel_adr + 6
            ].copy()

        # PID output:
        # [torque_x, torque_y, torque_z, collective_force]
        wrench = pid(
            pos_W=pos_W,
            quat_wxyz=quat_wxyz,
            vel_W=vel_W,
            omega_B=omega_B,
            pos_des=POS_DES,
            yaw_des=YAW_DES,
        )

        # Allocate the desired wrench to individual motor thrusts.
        thrust_cmd = allocation_pinv @ wrench

        # Ideal instantaneous motors:
        # no first-order delay or rotor-speed dynamics.
        thrust_actual = np.clip(
            thrust_cmd,
            THRUST_MIN,
            THRUST_MAX,
        )

        # Reaction torque from the existing allocation convention.
        reaction_torque = MOTOR_DIR * K_TAU * thrust_actual

        # Apply motor outputs to the existing MuJoCo actuators.
        data.ctrl[force_ids] = thrust_actual
        data.ctrl[torque_ids] = reaction_torque

        # Advance the physics by one timestep.
        mujoco.mj_step(model, data)

        # Record the resulting state.
        pos_after = data.qpos[qpos_adr:qpos_adr + 3].copy()
        vel_after = data.qvel[qvel_adr:qvel_adr + 3].copy()
        quat_after = data.qpos[
            qpos_adr + 3:qpos_adr + 7
        ].copy()

        log.append(
            [
                data.time,
                *pos_after,
                *vel_after,
                *quat_after,
                *thrust_actual,
            ]
        )

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------
    output_dir = Path("outputs/hover")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "hover_log.csv"

    header = [
        "time",
        "x", "y", "z",
        "vx", "vy", "vz",
        "qw", "qx", "qy", "qz",
        "thrust_motor0",
        "thrust_motor1",
        "thrust_motor2",
        "thrust_motor3",
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(log)

    result = np.asarray(log)
    time = result[:, 0]
    position = result[:, 1:4]

    # Position plot.
    fig, ax = plt.subplots(figsize=(10, 5))

    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        ax.plot(
            time,
            position[:, axis_index],
            label=f"{axis_name}",
        )

    ax.axhline(
        POS_DES[2],
        linestyle="--",
        label="Target z",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Crazyflie Standalone Hover Simulation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "hover_position.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Final results.
    final_pos = position[-1]
    final_error = final_pos - POS_DES

    print("Simulation finished.")
    print(f"Final position : {final_pos}")
    print(f"Target position: {POS_DES}")
    print(f"Position error : {final_error}")
    print(f"CSV log        : {csv_path}")
    print(f"Position plot  : {plot_path}")


if __name__ == "__main__":
    main()