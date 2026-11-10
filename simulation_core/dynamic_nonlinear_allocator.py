"""Multi-step predictive nonlinear control allocation for CF21B motors.

The allocator predicts the next rotor speed using the actuator's exact
first-order discretisation, then evaluates thrust and reaction torque,
including rotor acceleration torque, before solving for four ESC commands.
"""
from __future__ import annotations

import numpy as np


class DynamicNonlinearAllocator:
    """Bounded multi-step Gauss-Newton allocator for Cf21bFirstOrderActuatorModel."""

    def __init__(self, *, arm_length: float, motor_direction,
                 max_iterations: int = 8, damping: float = 1e-5,
                 prediction_horizon_steps: int = 25,
                 wrench_weights=(50.0, 50.0, 100.0, 1.0)):
        self.arm = float(arm_length)
        self.direction = np.asarray(motor_direction, dtype=float)
        if self.direction.shape != (4,):
            raise ValueError("motor_direction must have shape (4,)")
        if not np.isfinite(self.arm) or self.arm <= 0:
            raise ValueError("arm_length must be positive")
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        self.prediction_horizon_steps = int(prediction_horizon_steps)
        if self.prediction_horizon_steps < 1:
            raise ValueError("prediction_horizon_steps must be >= 1")
        self.weights = np.asarray(wrench_weights, dtype=float)
        if self.weights.shape != (4,) or np.any(self.weights <= 0):
            raise ValueError("wrench_weights must be four positive values")
        self.x = np.array([self.arm, -self.arm, -self.arm, self.arm])
        self.y = np.array([-self.arm, -self.arm, self.arm, self.arm])
        self._u = np.zeros(4)
        self._initialized = False

    def reset(self, motor_command=None):
        self._u = (np.zeros(4) if motor_command is None
                   else np.clip(np.asarray(motor_command, dtype=float), 0.0, 1.0))
        self._initialized = motor_command is not None

    @staticmethod
    def _poly_and_derivative(r, coeff):
        a3, a2, a1 = coeff
        return a3*r**3 + a2*r**2 + a1*r, 3*a3*r**2 + 2*a2*r + a1

    def _predict(self, u, actuator):
        """
        Predict the actuator state after N repeated applications of u.

        The command is held constant over the prediction horizon for the
        optimizer's model. In the simulation, only one command step is
        actually applied before the next allocation call.
        """
        N = self.prediction_horizon_steps
        dt = float(actuator.dt)
        decay = np.asarray(actuator.decay, dtype=float)
        gain = np.asarray(actuator.steady_state_gain_rad_s, dtype=float)
        omega_0 = np.asarray(actuator.omega, dtype=float)

        # Exact N-step solution of the first-order motor model:
        # omega[k+1] = decay*omega[k] + (1-decay)*gain*u
        decay_N = decay ** N
        omega_N = decay_N * omega_0 + (1.0 - decay_N) * gain * u
        omega_Nm1 = (
            (decay ** (N - 1)) * omega_0
            + (1.0 - decay ** (N - 1)) * gain * u
        ) if N > 1 else omega_0

        # Acceleration during the final predicted substep.
        omega_dot_N = (omega_N - omega_Nm1) / dt

        # Use the actuator's own mapping for predicted thrust and torque.
        thrust = actuator.thrust_from_omega(omega_N)
        ratio_raw = omega_N / actuator.omega_reference_rad_s
        ratio = np.clip(ratio_raw, 0.0, actuator.max_ratio)

        # d(omega_N)/du and d(omega_dot_N)/du.
        domega_du = (1.0 - decay_N) * gain
        domega_dot_du = (
            (1.0 - decay) * (decay ** (N - 1)) * gain / dt
        )

        # Thrust derivative, including low-speed and upper-thrust saturation.
        dthrust_dr = actuator._thrust_derivative(ratio)
        thrust_poly = actuator._thrust_polynomial(ratio)
        active_thrust = (
            (ratio_raw > actuator.positive_branch_min_ratio)
            & (ratio_raw < actuator.max_ratio)
            & (thrust_poly > 0.0)
            & (thrust_poly < actuator._mapping_thrust_max)
        )
        dF_du = np.where(
            active_thrust,
            dthrust_dr / actuator.omega_reference_rad_s * domega_du,
            0.0,
        )

        # Reaction torque polynomial follows the actuator implementation:
        # max(polynomial, 0), then add rotor-acceleration torque.
        a3, a2, a1 = actuator.torque_polynomial_coefficients
        trq_poly = a3 * ratio**3 + a2 * ratio**2 + a1 * ratio
        dtrq_dr = 3.0 * a3 * ratio**2 + 2.0 * a2 * ratio + a1
        trq_active = (
            (ratio_raw > 0.0)
            & (ratio_raw < actuator.max_ratio)
            & (trq_poly > 0.0)
        )
        dq_du = actuator.motor_direction * (
            np.where(
                trq_active,
                actuator.torque_polynomial_scale
                * dtrq_dr / actuator.omega_reference_rad_s * domega_du,
                0.0,
            )
        )
        if actuator.include_rotor_acceleration_torque:
            dq_du += (
                actuator.motor_direction
                * actuator.rotor_inertia_kg_m2
                * domega_dot_du
            )

        torque_magnitude = actuator.torque_polynomial_scale * np.maximum(
            trq_poly, 0.0
        )
        if actuator.include_rotor_acceleration_torque:
            torque_magnitude = torque_magnitude + (
                actuator.rotor_inertia_kg_m2 * omega_dot_N
            )
        q = actuator.motor_direction * torque_magnitude

        predicted_wrench = np.array([
            np.dot(self.y, thrust),
            np.dot(-self.x, thrust),
            np.sum(q),
            np.sum(thrust),
        ])

        jac = np.zeros((4, 4))
        jac[0, :] = self.y * dF_du
        jac[1, :] = -self.x * dF_du
        jac[2, :] = dq_du
        jac[3, :] = dF_du

        return predicted_wrench, jac, omega_N, thrust, q

    def _numerical_jacobian(self, u, actuator, base_wrench, eps=1e-3):
        """Finite-difference Jacobian of the exact clipped actuator prediction.

        A numerical, one-sided difference is important at command/thrust
        saturation: the clipped analytic thrust derivative can be zero even
        though reducing an ESC command moves the predicted motor out of
        saturation.
        """
        J = np.zeros((4, 4), dtype=float)
        for i in range(4):
            if u[i] >= 1.0 - eps:
                u_test = u.copy()
                u_test[i] = max(0.0, u[i] - eps)
                delta = u_test[i] - u[i]
                test_wrench, *_ = self._predict(u_test, actuator)
                J[:, i] = (test_wrench - base_wrench) / delta
            elif u[i] <= eps:
                u_test = u.copy()
                u_test[i] = min(1.0, u[i] + eps)
                delta = u_test[i] - u[i]
                test_wrench, *_ = self._predict(u_test, actuator)
                J[:, i] = (test_wrench - base_wrench) / delta
            else:
                u_plus = u.copy()
                u_minus = u.copy()
                u_plus[i] = min(1.0, u[i] + eps)
                u_minus[i] = max(0.0, u[i] - eps)
                delta = u_plus[i] - u_minus[i]
                wrench_plus, *_ = self._predict(u_plus, actuator)
                wrench_minus, *_ = self._predict(u_minus, actuator)
                J[:, i] = (wrench_plus - wrench_minus) / delta
        return J

    def allocate(self, wrench_des, actuator):
        """Return normalized ESC commands in [0,1] for the next physics step."""
        if not getattr(actuator, "include_rotor_acceleration_torque", False):
            raise ValueError("Dynamic allocation requires rotor acceleration torque enabled")
        if actuator.reaction_torque_model != "paper_polynomial":
            raise ValueError("Dynamic allocation requires paper_polynomial reaction torque")
        target = np.asarray(wrench_des, dtype=float)
        if target.shape != (4,) or not np.all(np.isfinite(target)):
            raise ValueError("wrench_des must be a finite [tau_x,tau_y,tau_z,Fz] vector")

        W = self.weights
        if not self._initialized:
            # A hover-like collective initial guess avoids the zero-thrust
            # polynomial branch having a zero local gradient at motor startup.
            per_motor_thrust = max(float(target[3]), 0.0) / 4.0
            omega_guess = actuator.inverse_thrust(
                np.full(4, min(per_motor_thrust, actuator._mapping_thrust_max))
            )
            self._u = np.clip(
                omega_guess / actuator.steady_state_gain_rad_s, 0.0, 1.0
            )
            self._initialized = True
        # Rebuild a collective-thrust seed at every allocation call. A warm
        # start at ESC=1 can otherwise remain trapped on the flat, clipped
        # thrust plateau: tiny local finite differences then report almost no
        # useful gradient even when a lower command would meet the target.
        per_motor_thrust = np.clip(
            max(float(target[3]), 0.0) / 4.0,
            0.0,
            actuator._mapping_thrust_max,
        )
        omega_des = actuator.inverse_thrust(np.full(4, per_motor_thrust))
        N = self.prediction_horizon_steps
        decay_N = np.asarray(actuator.decay, dtype=float) ** N
        gain = np.asarray(actuator.steady_state_gain_rad_s, dtype=float)
        omega_now = np.asarray(actuator.omega, dtype=float)
        denom = (1.0 - decay_N) * gain
        seed_u = np.divide(
            omega_des - decay_N * omega_now,
            denom,
            out=np.zeros(4, dtype=float),
            where=denom > 1e-12,
        )
        seed_u = np.clip(seed_u, 0.0, 1.0)

        # Compare the dynamic inverse seed with the previous solution and
        # start from whichever gives the lower current prediction cost.
        warm_u = np.clip(self._u, 0.0, 1.0)
        warm_pred, *_ = self._predict(warm_u, actuator)
        seed_pred, *_ = self._predict(seed_u, actuator)
        warm_residual = W * (warm_pred - target)
        seed_residual = W * (seed_pred - target)
        if float(seed_residual @ seed_residual) < float(warm_residual @ warm_residual):
            u = seed_u
        else:
            u = warm_u

        pred0, *_ = self._predict(u, actuator)
        residual0 = W * (pred0 - target)
        best_cost = float(residual0 @ residual0)
        best_u = u.copy()
        for _ in range(self.max_iterations):
            pred, _, *_ = self._predict(u, actuator)
            # Use finite differences so the optimizer can leave clipped
            # thrust regions instead of becoming stuck at ESC=1.0.
            J = self._numerical_jacobian(u, actuator, pred)
            residual = W * (pred - target)
            cost = float(residual @ residual)
            if cost < best_cost:
                best_cost = cost
                best_u = u.copy()
            JW = W[:, None] * J
            H = JW.T @ JW + self.damping * np.eye(4)
            g = JW.T @ residual
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            # Limit a single iteration to avoid jumping across the polynomial branch.
            step = np.clip(step, -0.20, 0.20)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                candidate = np.clip(u + scale * step, 0.0, 1.0)
                candidate_pred, *_ = self._predict(candidate, actuator)
                candidate_residual = W * (candidate_pred - target)
                candidate_cost = float(candidate_residual @ candidate_residual)
                if candidate_cost < cost:
                    u = candidate
                    accepted = True
                    break
            if accepted:
                # Keep the best point even when the final allowed iteration
                # accepted an improvement.
                candidate_pred, *_ = self._predict(u, actuator)
                candidate_residual = W * (candidate_pred - target)
                candidate_cost = float(candidate_residual @ candidate_residual)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_u = u.copy()

            if not accepted or np.linalg.norm(scale * step) < 1e-6:
                break

        pred, *_ = self._predict(best_u, actuator)
        self._u = best_u.copy()
        self.last_predicted_wrench = pred.copy()
        self.last_wrench_error = (target - pred).copy()
        return best_u.copy()
