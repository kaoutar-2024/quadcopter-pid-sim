import pytest

from quadsim.pid import PID


def test_proportional_only():
    pid = PID(2.0, 0.0, 0.0, -100, 100)
    assert pid.update(10.0, 4.0, 0.01) == pytest.approx(12.0)


def test_output_is_clamped():
    pid = PID(100.0, 0.0, 0.0, -1.0, 1.0)
    assert pid.update(10.0, 0.0, 0.01) == 1.0
    assert pid.update(-10.0, 0.0, 0.01) == -1.0


def test_integral_anti_windup_and_recovery():
    pid = PID(0.0, 10.0, 0.0, -1.0, 1.0)
    for _ in range(1000):
        pid.update(100.0, 0.0, 0.01)          # large sustained error
    assert pid.integral <= 1.0
    out = 0.0
    for _ in range(30):
        out = pid.update(-100.0, 0.0, 0.01)   # error flips sign
    assert out < 0.0                          # recovers quickly


def test_derivative_on_measurement_has_no_setpoint_kick():
    pid = PID(0.0, 0.0, 1.0, -100, 100)
    pid.update(0.0, 0.0, 0.01)
    # A big setpoint jump with unchanged measurement must not produce output.
    assert pid.update(50.0, 0.0, 0.01) == pytest.approx(0.0)


def test_derivative_opposes_measurement_motion():
    pid = PID(0.0, 0.0, 1.0, -100, 100)
    pid.update(0.0, 0.0, 0.01)
    assert pid.update(0.0, 0.1, 0.01) == pytest.approx(-10.0)
