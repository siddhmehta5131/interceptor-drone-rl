"""Trajectory-log schema and adapters (P10.1)."""
import numpy as np
import pytest

from interceptor.guidance.maneuvers import ManeuverType, PlatformType
from interceptor.visualiser import schema


def _log(n=10):
    return schema.TrajectoryLog(
        kind="guidance", dt=0.02, interceptor_pos=np.cumsum(np.ones((n, 3)), 0),
        target_pos=np.cumsum(np.ones((n, 3)), 0) + 5.0, kill_radius=1.0)


def test_validate_rejects_bad_shapes_and_kind():
    with pytest.raises(ValueError):
        schema.TrajectoryLog(kind="x", dt=0.1, interceptor_pos=np.zeros((3, 3)),
                             target_pos=np.zeros((3, 3))).validate()
    with pytest.raises(ValueError):
        schema.TrajectoryLog(kind="fc", dt=0.1, interceptor_pos=np.zeros((3, 3)),
                             target_pos=np.zeros((3, 3)), a_cmd=np.zeros((2, 3))).validate()


def test_save_load_roundtrip(tmp_path):
    lg = _log()
    lg.meta = {"seed": 3}
    lg2 = schema.TrajectoryLog.load(lg.save(tmp_path / "a.npz"))
    assert lg2.kind == "guidance" and lg2.kill_radius == 1.0 and lg2.meta == {"seed": 3}
    np.testing.assert_allclose(lg2.target_pos, lg.target_pos)


def test_to_frames_distance_matches():
    lg = _log()
    frames = lg.to_frames()
    assert len(frames) == len(lg)
    assert frames[0]["distance"] == pytest.approx(5.0 * np.sqrt(3))


def test_guidance_rollout_logs_aligned():
    lg = schema.rollout_guidance(PlatformType.QUADCOPTER, "apn_filtered",
                                 ManeuverType.CONSTANT_TURN
                                 if hasattr(ManeuverType, "CONSTANT_TURN") else list(ManeuverType)[1], 5)
    assert lg.kind == "guidance" and len(lg) > 5
    assert lg.a_apn.shape == lg.interceptor_pos.shape
    assert "killed" in lg.meta


def test_closed_loop_adapter():
    from interceptor.bridge.closed_loop import run_closed_loop
    r = run_closed_loop(max_time=1.0, wind=False)
    lg = schema.from_closed_loop(r)
    assert lg.kind == "closed_loop" and len(lg) == len(r.pos)


def test_fc_adapter():
    from interceptor.fc import setpoints
    from interceptor.fc.pid import RatePID
    from interceptor.fc.rollout import rollout
    lg = schema.from_fc_rollout(rollout(RatePID.tuned(dt=0.002), setpoints.doublet(200, 0.002, 0, 2.0)))
    assert lg.kind == "fc" and lg.omega.shape == (200, 3)
    assert lg.dt == pytest.approx(0.002)
