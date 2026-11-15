"""Ported physics must reproduce the original simulator exactly (plan P1.5).

Fixtures come from ``scripts/make_fixtures.py`` run against the archived
simulator.  Tolerance is 1e-12: the port is a verbatim copy, so any
difference means the model changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interceptor.sim import aero, constants, pipeline

FIX = Path(__file__).resolve().parents[1] / "fixtures"
TOL = 1e-12
STATE_KEYS = ("p", "q", "v", "omega", "Omega")

OPEN_LOOP = ["hover", "climb", "roll_step", "pitch_step", "yaw_step", "mixed"]
CLOSED_LOOP = ["pid_rate_steps", "pid_mixed"]


# The fixtures were recorded from the predecessor simulator with its motor constants.
# The live constants changed on 2026-11-15 (Forster 2015, BUGS #34), so every test in this
# file replays the port with constants.LEGACY_V1: it still proves the equations are a
# verbatim port, independently of the parameter values now in use.
_L = constants.LEGACY_V1
_L_OMEGA_HOVER = float(np.sqrt(constants.PH_M * 9.81 / (4.0 * _L["c_l"])))
_L_C_HOVER = constants.esc_cmd_for_omega(_L_OMEGA_HOVER, bat=_L["bat"])


@pytest.fixture(autouse=True)
def legacy_constants(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class LegacyParams(pipeline.PhysicsParams):
        c_l: float = _L["c_l"]
        c_d: float = _L["c_d"]

    monkeypatch.setattr(pipeline, "PhysicsParams", LegacyParams)
    monkeypatch.setattr(pipeline, "NOM_C_L", _L["c_l"])
    monkeypatch.setattr(pipeline, "NOM_C_D", _L["c_d"])
    monkeypatch.setattr(pipeline, "PH_BAT", _L["bat"])
    monkeypatch.setattr(pipeline, "PH_K_MOT", _L["k_mot"])
    monkeypatch.setattr(constants, "PH_OMEGA_HOVER", _L_OMEGA_HOVER)
    monkeypatch.setattr(constants, "PH_C_HOVER", _L_C_HOVER)


def _init(p0=(0.0, 0.0, 1.0)):
    return (np.array(p0, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3), np.zeros(3), np.full(4, constants.PH_OMEGA_HOVER))


def _assert_states(hist, fx):
    for k in STATE_KEYS:
        got = np.array(hist[k])
        assert got.shape == fx[k].shape
        assert np.max(np.abs(got - fx[k])) <= TOL, k


@pytest.mark.parametrize("name", OPEN_LOOP)
def test_open_loop_rollout(name):
    fx = np.load(FIX / f"sim_{name}.npz")
    params = pipeline.PhysicsParams()
    s = _init()
    hist = {k: [] for k in STATE_KEYS}
    for cmd in fx["cmd"]:
        s = pipeline.rk4(*s, cmd, 4.2, params)
        for k, val in zip(STATE_KEYS, s):
            hist[k].append(val)
    _assert_states(hist, fx)


def test_open_loop_with_wind_and_ground_effect():
    fx = np.load(FIX / "sim_mixed_wind_ge.npz")
    params = pipeline.PhysicsParams()
    s = _init(tuple(fx["p0"]))
    hist = {k: [] for k in STATE_KEYS}
    gains = []
    for cmd in fx["cmd"]:
        ge = aero.ground_effect_gain(s[0][2])
        gains.append(ge)
        s = pipeline.rk4(*s, cmd, 4.2, params, wind_W=fx["wind_W"], ge_gain=ge)
        for k, val in zip(STATE_KEYS, s):
            hist[k].append(val)
    assert np.max(np.abs(np.array(gains) - fx["ge_gain"])) <= TOL
    _assert_states(hist, fx)


@pytest.mark.parametrize("name", CLOSED_LOOP)
def test_closed_loop_pid_mixer(name):
    fx = np.load(FIX / f"sim_{name}.npz")
    params = pipeline.PhysicsParams()
    p, q, v, om, Om = _init()
    om_prev2, integ = np.zeros(3), np.zeros(3)
    hist = {k: [] for k in STATE_KEYS}
    cmds = []
    for a in fx["action"]:
        c_cmd, om_cmd, cut = pipeline.command_from_action(a)
        u, integ = pipeline.pid(om_cmd, om, om_prev2, integ, cut, 0.01)
        cmd = np.zeros(4) if cut else pipeline.mixer(u, c_cmd)
        cmds.append(cmd)
        nxt = pipeline.rk4(p, q, v, om, Om, cmd, 4.2, params)
        om_prev2 = om.copy()
        p, q, v, om, Om = nxt
        for k, val in zip(STATE_KEYS, nxt):
            hist[k].append(val)
    assert np.max(np.abs(np.array(cmds) - fx["cmd"])) <= TOL
    _assert_states(hist, fx)


def test_domain_randomisation_draw_order():
    fx = np.load(FIX / "sim_dr_samples.npz")
    rng = np.random.default_rng(int(fx["seed"]))
    for row in fx["samples"]:
        pp = pipeline.sample_physics_params(rng)
        got = np.concatenate([
            [pp.mass], np.diag(pp.J), [pp.c_l, pp.c_d],
            pp.fx_raw, pp.fy_raw, pp.fz_raw, pp.tx_raw, pp.ty_raw, pp.tz_raw])
        assert np.max(np.abs(got - row)) <= TOL


def test_legacy_constants_match_fixture():
    fx = np.load(FIX / "sim_constants.npz")
    assert _L_C_HOVER == pytest.approx(float(fx["c_hover"]), abs=1e-9)
    assert _L_OMEGA_HOVER == pytest.approx(float(fx["omega_hover"]), abs=1e-9)
    assert constants.PH_M == float(fx["mass"])
    assert np.array_equal(np.asarray(constants.PH_J), fx["J"])
    assert _L["c_l"] == float(fx["c_l"])
    assert _L["c_d"] == float(fx["c_d"])
    assert _L["k_mot"] == float(fx["k_mot"])
    assert _L["omega_max"] == float(fx["omega_max"])
    assert np.array_equal(_L["bat"], fx["bat"])
    for name, key in (("PH_MIX", "mix"), ("PH_R_P", "r_p"),
                      ("PH_KP", "kp"), ("PH_KI", "ki"), ("PH_KD", "kd")):
        assert np.array_equal(np.asarray(getattr(constants, name)), fx[key]), name
