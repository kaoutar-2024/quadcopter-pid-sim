"""Rigid-body dynamics of a quadcopter.

Conventions
-----------
* World frame: x forward, y left, z up (right-handed). Gravity is -z.
* Body frame: x forward, y left, z up, attached to the centre of mass.
* Euler angles: roll (phi, about x), pitch (theta, about y), yaw (psi, about z),
  applied in Z-Y-X order, so R = Rz(psi) Ry(theta) Rx(phi) maps body to world.
* State vector (12): [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
  where (p, q, r) are the body angular rates.

The model is a standard 6-DOF rigid body: thrust acts along body +z, torques
come from the motor mixer, and a small linear drag acts on the velocity.
"""

from __future__ import annotations

import numpy as np

from .params import QuadParams

X, Y, Z, VX, VY, VZ, PHI, THETA, PSI, P, Q, R = range(12)
STATE_SIZE = 12


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body-to-world rotation matrix for Z-Y-X Euler angles."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph],
            [sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph],
            [-sth, cth * sph, cth * cph],
        ]
    )


def derivatives(
    state: np.ndarray,
    thrust: float,
    torque: np.ndarray,
    params: QuadParams,
) -> np.ndarray:
    """Time derivative of the state for a given total thrust and body torque."""
    phi, theta = state[PHI], state[THETA]
    vel = state[VX : VZ + 1]
    omega = state[P : R + 1]

    # Translational dynamics: m a = R [0, 0, T] - m g z_hat - drag * v
    rot = rotation_matrix(phi, theta, state[PSI])
    force = rot @ np.array([0.0, 0.0, thrust])
    accel = force / params.mass - np.array([0.0, 0.0, params.g])
    accel -= params.drag * vel / params.mass

    # Euler-angle kinematics (singular at theta = +/- 90 degrees).
    p, q, r = omega
    sph, cph = np.sin(phi), np.cos(phi)
    tth, cth = np.tan(theta), np.cos(theta)
    phi_dot = p + (q * sph + r * cph) * tth
    theta_dot = q * cph - r * sph
    psi_dot = (q * sph + r * cph) / cth

    # Rotational dynamics: I omega_dot = tau - omega x (I omega)
    inertia = np.array([params.ixx, params.iyy, params.izz])
    omega_dot = (torque - np.cross(omega, inertia * omega)) / inertia

    out = np.empty(STATE_SIZE)
    out[X : Z + 1] = vel
    out[VX : VZ + 1] = accel
    out[PHI], out[THETA], out[PSI] = phi_dot, theta_dot, psi_dot
    out[P : R + 1] = omega_dot
    return out


def rk4_step(
    state: np.ndarray,
    thrust: float,
    torque: np.ndarray,
    params: QuadParams,
    dt: float,
) -> np.ndarray:
    """One fourth-order Runge-Kutta step with thrust and torque held constant."""
    k1 = derivatives(state, thrust, torque, params)
    k2 = derivatives(state + 0.5 * dt * k1, thrust, torque, params)
    k3 = derivatives(state + 0.5 * dt * k2, thrust, torque, params)
    k4 = derivatives(state + dt * k3, thrust, torque, params)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
