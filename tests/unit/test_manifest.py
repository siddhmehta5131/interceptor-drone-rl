"""P7.5: manifest hashes checkpoints and detects tampering."""

from __future__ import annotations

import json
from pathlib import Path

from training.manifest import build_manifest, sha256, verify_manifest


def _fake_run(root: Path, name: str, payload: bytes = b"model") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "checkpoint.zip").write_bytes(payload)
    (d / "DONE").write_text(json.dumps({
        "run_id": name, "algo": "ppo", "task": "guidance", "seed": 0,
        "num_timesteps": 128}), encoding="utf-8")
    (d / "git.txt").write_text("abc123", encoding="utf-8")
    (d / "config.sha256").write_text("deadbeef", encoding="utf-8")
    return d


def test_manifest_records_hash_and_verifies(tmp_path: Path):
    run = _fake_run(tmp_path, "g3_seed0")
    manifest = build_manifest(tmp_path)
    assert len(manifest["runs"]) == 1
    entry = manifest["runs"][0]
    assert entry["run_id"] == "g3_seed0"
    assert entry["sha256"] == sha256(run / "checkpoint.zip")
    assert entry["commit"] == "abc123"
    assert entry["config_sha256"] == "deadbeef"
    assert verify_manifest(manifest, tmp_path) == []


def test_manifest_detects_tampering_and_missing(tmp_path: Path):
    run = _fake_run(tmp_path, "g3_seed1")
    manifest = build_manifest(tmp_path)
    (run / "checkpoint.zip").write_bytes(b"tampered")
    problems = verify_manifest(manifest, tmp_path)
    assert any("sha256" in p for p in problems)
    (run / "checkpoint.zip").unlink()
    problems = verify_manifest(manifest, tmp_path)
    assert any("missing" in p for p in problems)
