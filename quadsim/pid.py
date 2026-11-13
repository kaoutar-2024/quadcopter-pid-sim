"""Discrete PID controller.

* Output is clamped to [out_min, out_max].
* The integral term is clamped to the same range (anti-windup).
* The derivative acts on the measurement, not the error, so a step in the
  setpoint does not cause a derivative kick.
"""

from __future__ import annotations


class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float, out_max: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_meas: float | None = None

    def _clamp(self, value: float) -> float:
        return max(self.out_min, min(self.out_max, value))

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        self.integral = self._clamp(self.integral + self.ki * error * dt)

        derivative = 0.0
        if self._prev_meas is not None and dt > 0.0:
            derivative = -(measurement - self._prev_meas) / dt
        self._prev_meas = measurement

        return self._clamp(self.kp * error + self.integral + self.kd * derivative)
