import numpy as np
import pytest

from quadsim.dynamics import (P, PHI, THETA, VX, VZ, X, Z, derivatives,
                              rk4_step, rotation_matrix)
from quadsim.params import QuadParams

PARAMS = QuadParams()


def test_rotation_matrix_is_a_proper_rotation():
    rng = np.random.default_rng(2)
    for _ in range(50):
        r = rotation_matrix(*rng.uniform(-1.2, 1.2, 3))
        assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_hover_is_an_equilibrium():
    state = np.zeros(12)
    d = derivatives(state, PARAMS.hover_thrust, np.zeros(3), PARAMS)
    assert d == pytest.approx(np.zeros(12), abs=1e-12)


def test_free_fall_acceleration_is_g():
    d = derivatives(np.zeros(12), 0.0, np.zeros(3), PARAMS)
    assert d[VZ] == pytest.approx(-PARAMS.g)


def test_roll_torque_gives_roll_acceleration():
    tau = 0.01
    d = derivatives(np.zeros(12), PARAMS.hover_thrust, np.array([tau, 0, 0]), PARAMS)
    assert d[P] == pytest.approx(tau / PARAMS.ixx)


def test_nose_down_pitch_accelerates_forward():
    state = np.zeros(12)
    state[THETA] = np.radians(10)
    d = derivatives(state, PARAMS.hover_thrust, np.zeros(3), PARAMS)
    assert d[VX] > 0.0


def test_tilted_hover_thrust_loses_altitude_without_compensation():
    state = np.zeros(12)
    state[PHI] = np.radians(30)
    d = derivatives(state, PARAMS.hover_thrust, np.zeros(3), PARAMS)
    assert d[VZ] < 0.0


def test_rk4_free_fall_matches_analytic_solution():
    # No drag so the closed-form solution z = -g t^2 / 2 is exact.
    params = QuadParams(drag=0.0)
    state = np.zeros(12)
    dt, steps = 0.001, 1000          # 1 second
    for _ in range(steps):
        state = rk4_step(state, 0.0, np.zeros(3), params, dt)
    assert state[Z] == pytest.approx(-0.5 * params.g, rel=1e-6)
    assert state[X] == 0.0
