"""Closed-loop simulation: dynamics + motor lag + controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .controller import QuadController, Reference
from .dynamics import STATE_SIZE, rk4_step
from .mixer import Mixer
from .params import QuadParams


@dataclass
class Log:
    t: np.ndarray            # (N,)
    state: np.ndarray        # (N, 12)
    motors: np.ndarray       # (N, 4) actual motor thrusts
    ref: np.ndarray          # (N, 4) x, y, z, yaw reference


class Simulator:
    """Fixed-step simulation. The controller runs at every step (500 Hz by
    default) and motor thrust follows the command through a first-order lag."""

    def __init__(self, params: QuadParams, controller: QuadController,
                 dt: float = 0.002, state0: np.ndarray | None = None):
        self.params = params
        self.controller = controller
        self.mixer = Mixer(params)
        self.dt = dt
        self.state0 = np.zeros(STATE_SIZE) if state0 is None else state0.copy()

    def run(
        self,
        duration: float,
        ref_fn: Callable[[float], Reference],
        ext_torque_fn: Callable[[float], np.ndarray] | None = None,
    ) -> Log:
        p, dt = self.params, self.dt
        n = int(round(duration / dt))
        state = self.state0.copy()
        self.controller.reset()
        motors = np.full(4, p.hover_thrust / 4.0)  # motors already at hover

        t_log = np.empty(n)
        s_log = np.empty((n, STATE_SIZE))
        m_log = np.empty((n, 4))
        r_log = np.empty((n, 4))

        alpha = dt / (p.motor_tau + dt)
        for i in range(n):
            t = i * dt
            ref = ref_fn(t)
            command = self.controller.update(state, ref, dt)

            # First-order motor lag.
            motors = motors + alpha * (command - motors)
            thrust, torque = self.mixer.wrench(motors)
            if ext_torque_fn is not None:
                torque = torque + ext_torque_fn(t)

            state = rk4_step(state, thrust, torque, p, dt)

            t_log[i] = t
            s_log[i] = state
            m_log[i] = motors
            r_log[i] = (ref.x, ref.y, ref.z, ref.yaw)

        return Log(t_log, s_log, m_log, r_log)
