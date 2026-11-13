"""Motor mixer for an X-configuration quadcopter.

Motor layout, seen from above (x forward, y left):

        front
    M2 (+d,+d)  M1 (+d,-d)
    M3 (-d,+d)  M4 (-d,-d)
        rear

M1 and M3 spin one way, M2 and M4 the other, so the yaw reaction torques cancel
at hover. Each motor produces an upward force f_i >= 0, and the vehicle
wrench is

    T      =  f1 + f2 + f3 + f4
    tau_x  =  d (-f1 + f2 + f3 - f4)          (roll)
    tau_y  = -d ( f1 + f2 - f3 - f4)          (pitch)
    tau_z  =  c ( f1 - f2 + f3 - f4)          (yaw)

The mixer inverts this 4x4 map, then clips each motor to its physical limits.
"""

from __future__ import annotations

import numpy as np

from .params import QuadParams


def allocation_matrix(params: QuadParams) -> np.ndarray:
    """Matrix A such that [T, tau_x, tau_y, tau_z] = A @ [f1, f2, f3, f4]."""
    d, c = params.d, params.yaw_coeff
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-d, d, d, -d],
            [-d, -d, d, d],
            [c, -c, c, -c],
        ]
    )


class Mixer:
    def __init__(self, params: QuadParams):
        self.params = params
        self._a = allocation_matrix(params)
        self._a_inv = np.linalg.inv(self._a)

    def allocate(self, thrust: float, torque: np.ndarray) -> np.ndarray:
        """Per-motor thrusts for a desired wrench, limited to [0, max].

        If the request is out of range, all four motors are shifted by the
        same amount before clipping. A common shift changes only the total
        thrust, so the roll/pitch/yaw torques are preserved: attitude control
        takes priority over altitude when the motors saturate. Without this,
        clipping one motor creates an unwanted torque and can flip the vehicle.
        """
        wrench = np.array([thrust, torque[0], torque[1], torque[2]])
        forces = self._a_inv @ wrench

        lo, hi = 0.0, self.params.max_motor_thrust
        if forces.max() - forces.min() <= hi - lo:
            if forces.min() < lo:
                forces = forces + (lo - forces.min())
            elif forces.max() > hi:
                forces = forces - (forces.max() - hi)
        return np.clip(forces, lo, hi)

    def wrench(self, forces: np.ndarray) -> tuple[float, np.ndarray]:
        """Total thrust and body torque produced by the given motor thrusts."""
        out = self._a @ forces
        return float(out[0]), out[1:]
