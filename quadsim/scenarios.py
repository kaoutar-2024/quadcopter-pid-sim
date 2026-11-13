"""Ready-made experiments: attitude step, disturbance rejection, waypoint mission.

Run all of them and regenerate the plots in docs/ with:

    python -m quadsim.scenarios
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from .controller import QuadController, Reference
from .dynamics import PHI, PSI, STATE_SIZE, THETA, VX
from .params import QuadParams
from .simulator import Log, Simulator


def _make_sim(z0: float = 1.0) -> Simulator:
    params = QuadParams()
    state0 = np.zeros(STATE_SIZE)
    state0[2] = z0
    return Simulator(params, QuadController(params), state0=state0)


# --------------------------------------------------------------------- metrics

def step_metrics(t: np.ndarray, y: np.ndarray, t_step: float, target: float,
                 band: float = 0.02) -> dict:
    """Overshoot, 10-90 % rise time, settling time and final error of a step
    from 0 to `target` applied at `t_step`."""
    mask = t >= t_step
    ts, ys = t[mask] - t_step, y[mask]
    peak = ys.max() if target > 0 else ys.min()
    overshoot = max(0.0, (peak - target) / target * 100.0)

    def first_crossing(frac: float) -> float:
        idx = np.argmax(ys >= frac * target) if target > 0 else np.argmax(ys <= frac * target)
        return float(ts[idx])

    outside = np.where(np.abs(ys - target) > band * abs(target))[0]
    settling = float(ts[outside[-1]] + (ts[1] - ts[0])) if len(outside) else 0.0
    return {
        "overshoot_pct": float(overshoot),
        "rise_time_s": first_crossing(0.9) - first_crossing(0.1),
        "settling_time_s": settling,
        "final_error": float(ys[-1] - target),
    }


# ------------------------------------------------------------------- scenarios

def attitude_step(angle_deg: float = 10.0, t_step: float = 0.5,
                  duration: float = 3.0) -> tuple[Log, dict]:
    """Hover, then command a roll step; position loop is bypassed."""
    sim = _make_sim()
    target = math.radians(angle_deg)
    log = sim.run(duration, lambda t: Reference(
        0.0, 0.0, 1.0, roll=target if t >= t_step else 0.0, pitch=0.0))
    metrics = step_metrics(log.t, log.state[:, PHI], t_step, target)
    return log, metrics


def disturbance_rejection(torque: float = 0.05, t_on: float = 1.0,
                          width: float = 0.05, duration: float = 4.0
                          ) -> tuple[Log, dict]:
    """Hover and hit the vehicle with a short roll-torque pulse."""
    sim = _make_sim()

    def pulse(t: float) -> np.ndarray:
        on = t_on <= t < t_on + width
        return np.array([torque if on else 0.0, 0.0, 0.0])

    log = sim.run(duration, lambda t: Reference(0.0, 0.0, 1.0), ext_torque_fn=pulse)
    roll_deg = np.degrees(log.state[:, PHI])
    after = log.t >= t_on
    peak = float(np.abs(roll_deg[after]).max())
    outside = np.where(after & (np.abs(roll_deg) > 0.5))[0]
    recovery = float(log.t[outside[-1]] - t_on) if len(outside) else 0.0
    drift = float(np.abs(log.state[after, 1]).max())
    return log, {
        "peak_roll_deg": peak,
        "recovery_time_s": recovery,   # time until roll stays within 0.5 deg
        "max_lateral_drift_m": drift,
    }


WAYPOINTS = [  # (x, y, z, hold_seconds)
    (0.0, 0.0, 1.0, 4.0),
    (2.0, 0.0, 1.0, 5.0),
    (2.0, 2.0, 1.5, 5.0),
    (0.0, 2.0, 1.0, 5.0),
    (0.0, 0.0, 1.0, 5.0),
]


def waypoint_mission() -> tuple[Log, dict]:
    """Take off from the origin and fly a square with a climb on one leg."""
    starts = np.cumsum([0.0] + [w[3] for w in WAYPOINTS])
    duration = float(starts[-1])

    def ref_fn(t: float) -> Reference:
        k = max(0, int(np.searchsorted(starts, t, side="right")) - 1)
        x, y, z, _ = WAYPOINTS[min(k, len(WAYPOINTS) - 1)]
        return Reference(x, y, z)

    sim = _make_sim(z0=0.0)
    log = sim.run(duration, ref_fn)

    legs = []
    for k, (x, y, z, hold) in enumerate(WAYPOINTS):
        sel = (log.t >= starts[k]) & (log.t < starts[k + 1])
        dist = np.linalg.norm(log.state[sel, :3] - np.array([x, y, z]), axis=1)
        outside = np.where(dist > 0.05)[0]
        legs.append(float(log.t[sel][outside[-1]] - starts[k]) if len(outside) else 0.0)
    return log, {
        "time_to_within_5cm_per_leg_s": legs,
        "max_tilt_deg": float(np.degrees(np.abs(log.state[:, PHI:THETA + 1]).max())),
        "max_yaw_deg": float(np.degrees(np.abs(log.state[:, PSI]).max())),
        "max_speed_mps": float(np.linalg.norm(log.state[:, VX:VX + 3], axis=1).max()),
        "final_position_error_m": float(np.linalg.norm(
            log.state[-1, :3] - np.array(WAYPOINTS[-1][:3]))),
    }


# ----------------------------------------------------------------------- plots

def make_plots(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Attitude step
    log, m = attitude_step()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(log.t, np.degrees(log.state[:, PHI]), label="roll")
    ax.plot(log.t, np.where(log.t >= 0.5, 10.0, 0.0), "--", label="command")
    ax.set(xlabel="time (s)", ylabel="roll (deg)",
           title=f"10° roll step: overshoot {m['overshoot_pct']:.1f} %, "
                 f"settling {m['settling_time_s']:.2f} s")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "attitude_step.png", dpi=130); plt.close(fig)

    # 2. Disturbance rejection
    log, m = disturbance_rejection()
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(log.t, np.degrees(log.state[:, PHI]))
    axes[0].set(ylabel="roll (deg)",
                title=f"Roll-torque pulse at t = 1 s: peak {m['peak_roll_deg']:.1f}°, "
                      f"back within 0.5° after {m['recovery_time_s']:.2f} s")
    axes[1].plot(log.t, log.state[:, 1])
    axes[1].set(xlabel="time (s)", ylabel="y position (m)")
    for a in axes: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "disturbance.png", dpi=130); plt.close(fig)

    # 3. Waypoint mission
    log, m = waypoint_mission()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(log.state[:, 0], log.state[:, 1], label="flown path")
    wp = np.array([w[:2] for w in WAYPOINTS])
    axes[0].plot(wp[:, 0], wp[:, 1], "ro--", alpha=0.6, label="waypoints")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Top view", aspect="equal")
    axes[1].plot(log.t, log.state[:, 2], label="z")
    axes[1].plot(log.t, log.ref[:, 2], "--", label="z command")
    axes[1].set(xlabel="time (s)", ylabel="altitude (m)", title="Altitude")
    for i in range(4):
        axes[2].plot(log.t, log.motors[:, i], lw=0.8, label=f"M{i + 1}")
    axes[2].set(xlabel="time (s)", ylabel="thrust (N)", title="Motor thrusts")
    for a in axes: a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "waypoint_mission.png", dpi=130); plt.close(fig)


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "docs"

    _, m = attitude_step()
    print("Attitude step (10 deg roll)")
    for k, v in m.items(): print(f"  {k:24s} {v:.4f}")

    _, m = disturbance_rejection()
    print("Disturbance rejection (0.05 N m pulse, 50 ms)")
    for k, v in m.items(): print(f"  {k:24s} {v:.4f}")

    _, m = waypoint_mission()
    print("Waypoint mission")
    for k, v in m.items():
        print(f"  {k:32s} {np.round(v, 3) if isinstance(v, list) else round(v, 4)}")

    make_plots(out_dir)
    print(f"Plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
