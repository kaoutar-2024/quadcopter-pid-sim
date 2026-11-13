"""Cascaded PID flight controller.

Structure (outer to inner):

    position error --PD--> desired horizontal acceleration
                    --small-angle map--> desired roll / pitch
    attitude error --PID--> body torques (roll, pitch, yaw)
    altitude error --PID--> thrust correction on top of hover thrust
                    --mixer--> four motor thrusts

The horizontal position loop is bypassed when the reference gives explicit
roll/pitch angles (used for attitude step-response tests).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dynamics import PHI, PSI, THETA, VX, VY, X, Y, Z, VZ
from .mixer import Mixer
from .params import QuadParams
from .pid import PID


@dataclass
class Reference:
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0
    yaw: float = 0.0
    # If set, these attitude angles (rad) override the position loop.
    roll: float | None = None
    pitch: float | None = None


@dataclass
class Gains:
    # Attitude loops: gains chosen from wn^2 * I and 2 * zeta * wn * I.
    roll_pitch: tuple[float, float, float] = (1.94, 0.10, 0.155)
    yaw: tuple[float, float, float] = (0.88, 0.05, 0.14)
    # Altitude loop (N, N/m, N s/m), scaled by the vehicle mass.
    altitude: tuple[float, float, float] = (11.7, 2.0, 4.2)
    # Horizontal position loop (acceleration per metre, per m/s).
    pos_kp: float = 4.0
    pos_kd: float = 3.6
    max_tilt_deg: float = 30.0
    # Torque limits are chosen so that roll + pitch + yaw demands together
    # stay inside what the four motors can deliver around hover.
    max_torque_rp: float = 0.25   # N m, roll and pitch
    max_torque_yaw: float = 0.04  # N m
    max_thrust_correction: float = 6.0  # N


class QuadController:
    def __init__(self, params: QuadParams, gains: Gains | None = None):
        self.params = params
        self.gains = gains or Gains()
        g = self.gains
        self.mixer = Mixer(params)
        self.max_tilt = math.radians(g.max_tilt_deg)

        rp, yaw = g.max_torque_rp, g.max_torque_yaw
        self.roll_pid = PID(*g.roll_pitch, -rp, rp)
        self.pitch_pid = PID(*g.roll_pitch, -rp, rp)
        self.yaw_pid = PID(*g.yaw, -yaw, yaw)
        tc = g.max_thrust_correction
        self.alt_pid = PID(*g.altitude, -tc, tc)

    def reset(self) -> None:
        for pid in (self.roll_pid, self.pitch_pid, self.yaw_pid, self.alt_pid):
            pid.reset()

    def desired_attitude(self, state: np.ndarray, ref: Reference) -> tuple[float, float]:
        """Desired (roll, pitch) in radians."""
        if ref.roll is not None or ref.pitch is not None:
            return ref.roll or 0.0, ref.pitch or 0.0

        g = self.gains
        ax = g.pos_kp * (ref.x - state[X]) - g.pos_kd * state[VX]
        ay = g.pos_kp * (ref.y - state[Y]) - g.pos_kd * state[VY]

        # Small-angle inversion of the horizontal thrust components, rotated
        # into the current yaw direction.
        psi = state[PSI]
        grav = self.params.g
        pitch = (ax * math.cos(psi) + ay * math.sin(psi)) / grav
        roll = (ax * math.sin(psi) - ay * math.cos(psi)) / grav
        lim = self.max_tilt
        return max(-lim, min(lim, roll)), max(-lim, min(lim, pitch))

    def update(self, state: np.ndarray, ref: Reference, dt: float) -> np.ndarray:
        """Returns the four commanded motor thrusts (N)."""
        roll_ref, pitch_ref = self.desired_attitude(state, ref)

        tau_x = self.roll_pid.update(roll_ref, state[PHI], dt)
        tau_y = self.pitch_pid.update(pitch_ref, state[THETA], dt)
        tau_z = self.yaw_pid.update(ref.yaw, state[PSI], dt)

        # Hover thrust plus PID correction, divided by the tilt so the
        # vertical component of thrust still balances gravity.
        correction = self.alt_pid.update(ref.z, state[Z], dt)
        tilt = math.cos(state[PHI]) * math.cos(state[THETA])
        thrust = (self.params.hover_thrust + correction) / max(tilt, 0.5)

        return self.mixer.allocate(thrust, np.array([tau_x, tau_y, tau_z]))
