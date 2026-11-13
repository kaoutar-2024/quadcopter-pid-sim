import numpy as np
import pytest

from quadsim.mixer import Mixer, allocation_matrix
from quadsim.params import QuadParams

P = QuadParams()
MIX = Mixer(P)


def test_hover_gives_equal_motor_thrust():
    forces = MIX.allocate(P.hover_thrust, np.zeros(3))
    assert forces == pytest.approx(np.full(4, P.hover_thrust / 4.0))


def test_wrench_roundtrip_when_feasible():
    rng = np.random.default_rng(0)
    for _ in range(200):
        thrust = P.hover_thrust * rng.uniform(0.8, 1.2)
        torque = rng.uniform(-0.05, 0.05, 3)
        torque[2] *= 0.1
        forces = MIX.allocate(thrust, torque)
        t_out, tau_out = MIX.wrench(forces)
        assert t_out == pytest.approx(thrust, abs=1e-9)
        assert tau_out == pytest.approx(torque, abs=1e-9)


def test_motor_limits_always_respected():
    rng = np.random.default_rng(1)
    for _ in range(200):
        forces = MIX.allocate(rng.uniform(0, 30), rng.uniform(-2, 2, 3))
        assert forces.min() >= 0.0
        assert forces.max() <= P.max_motor_thrust


def test_desaturation_keeps_torque_and_sacrifices_thrust():
    # Ask for a large roll torque at low thrust: one motor would go negative.
    thrust = 1.0
    torque = np.array([0.25, 0.0, 0.0])
    raw = np.linalg.inv(allocation_matrix(P)) @ np.array([thrust, *torque])
    assert raw.min() < 0.0                      # the naive solution is infeasible

    forces = MIX.allocate(thrust, torque)
    t_out, tau_out = MIX.wrench(forces)
    assert forces.min() >= 0.0
    assert tau_out == pytest.approx(torque, abs=1e-9)   # torque preserved
    assert t_out > thrust                               # thrust raised to stay feasible


def test_positive_roll_torque_raises_left_motors():
    # +y side is the left: M2 (front-left) and M3 (rear-left).
    forces = MIX.allocate(P.hover_thrust, np.array([0.05, 0.0, 0.0]))
    assert forces[1] > forces[0] and forces[2] > forces[3]
