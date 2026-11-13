"""End-to-end checks of the cascaded controller in simulation."""

import numpy as np

from quadsim import QuadController, QuadParams, Reference, Simulator
from quadsim.dynamics import PHI, PSI, STATE_SIZE, THETA
from quadsim.scenarios import (WAYPOINTS, attitude_step,
                               disturbance_rejection, waypoint_mission)


def _sim(z0=1.0):
    params = QuadParams()
    state0 = np.zeros(STATE_SIZE)
    state0[2] = z0
    return Simulator(params, QuadController(params), state0=state0)


def test_hover_holds_position():
    log = _sim().run(3.0, lambda t: Reference(0.0, 0.0, 1.0))
    assert np.abs(log.state[:, :3] - np.array([0, 0, 1.0])).max() < 1e-6


def test_attitude_step_response_is_well_damped():
    _, m = attitude_step(angle_deg=10.0)
    assert m["overshoot_pct"] < 10.0
    assert m["settling_time_s"] < 0.6
    assert abs(m["final_error"]) < 0.01 * np.radians(10.0)


def test_disturbance_is_rejected():
    _, m = disturbance_rejection()
    assert m["peak_roll_deg"] < 3.0
    assert m["recovery_time_s"] < 0.5
    assert m["max_lateral_drift_m"] < 0.05


def test_waypoint_mission_reaches_every_waypoint():
    log, m = waypoint_mission()
    assert m["final_position_error_m"] < 0.02
    assert all(t < 4.5 for t in m["time_to_within_5cm_per_leg_s"])
    assert m["max_yaw_deg"] < 1.0


def test_aggressive_diagonal_move_stays_stable():
    # Simultaneous roll + pitch + climb demand pushes motors close to their
    # limits. (The mixer's desaturation itself is covered in test_mixer.py;
    # this checks the whole loop stays bounded and converges.)
    log = _sim().run(8.0, lambda t: Reference(3.0, -2.0, 2.0))
    assert np.isfinite(log.state).all()
    assert np.abs(np.degrees(log.state[:, [PHI, THETA]])).max() < 45.0
    assert np.abs(np.degrees(log.state[:, PSI])).max() < 15.0
    assert np.linalg.norm(log.state[-1, :3] - np.array([3.0, -2.0, 2.0])) < 0.05


def test_yaw_command_is_tracked():
    yaw = np.radians(45.0)
    log = _sim().run(4.0, lambda t: Reference(0.0, 0.0, 1.0, yaw=yaw))
    assert abs(log.state[-1, PSI] - yaw) < np.radians(1.0)
    assert np.abs(log.state[:, :3] - np.array([0, 0, 1.0])).max() < 0.05
