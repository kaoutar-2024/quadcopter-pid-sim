"""Physical parameters of the simulated quadcopter.

The values describe a generic ~0.5 kg X-configuration quadcopter. They are
plausible for a small research quad but do not model any specific product.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QuadParams:
    mass: float = 0.468            # kg
    arm: float = 0.225             # m, centre of the vehicle to each motor
    ixx: float = 4.856e-3          # kg m^2
    iyy: float = 4.856e-3          # kg m^2
    izz: float = 8.801e-3          # kg m^2
    yaw_coeff: float = 0.016       # m, yaw torque produced per N of thrust
    max_motor_thrust: float = 4.0  # N per motor (hover needs ~1.15 N)
    motor_tau: float = 0.02        # s, first-order motor/propeller lag
    drag: float = 0.1              # N s/m, linear translational drag
    g: float = 9.81                # m/s^2

    @property
    def d(self) -> float:
        """Perpendicular distance of each motor from the x and y axes."""
        return self.arm / math.sqrt(2.0)

    @property
    def hover_thrust(self) -> float:
        """Total thrust needed to hover, in newtons."""
        return self.mass * self.g
