"""Per-stage JSON demo configuration: schema, validation and env wiring.

One JSON file per curriculum stage (``stage_01.json`` ... ``stage_08.json``).
A stage config is the *only* input the stage-mode runtime needs besides the
training package; everything else (env kwargs, observation layout, camera,
video settings, instruction set) is derived from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.envs.stage_config import STAGES
from src.utils.config_loader import apply_stage_overrides
from src.utils.model_loader import PRIVILEGED_FIELDS

SCHEMA_ID = "stage-demo/1"
CONTROL_MODES = ("instructions", "model")
INSTRUCTION_MODES = ("open_loop", "guidance")
VIDEO_FORMATS = ("mp4", "gif", "png")
CAMERA_MODES = ("follow", "fixed", "overview")

#: Rows in one future frame (rel pos, rel vel, visibility) -- obs_builder.
FUTURE_SAMPLE_DIM = 7
#: Base rows in one target frame when history/future are present.
TARGET_FRAME_DIM = 19
#: Stage-1 observation width (no target, no history, no future).
STAGE1_OBS_DIM = 14


class StageDemoError(ValueError):
    """Raised for any malformed stage-demo JSON."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise StageDemoError(msg)


def _get(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    val = mapping.get(key, default)
    return default if val is None else val


def _number(value: Any, name: str, *, lo: Optional[float] = None,
            hi: Optional[float] = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{name} must be a number, got {value!r}")
    out = float(value)
    if lo is not None:
        _require(out >= lo, f"{name} must be >= {lo}, got {out}")
    if hi is not None:
        _require(out <= hi, f"{name} must be <= {hi}, got {out}")
    return out


def _int(value: Any, name: str, *, lo: Optional[int] = None,
         hi: Optional[int] = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{name} must be an int, got {value!r}")
    out = int(value)
    if lo is not None:
        _require(out >= lo, f"{name} must be >= {lo}, got {out}")
    if hi is not None:
        _require(out <= hi, f"{name} must be <= {hi}, got {out}")
    return out


def infer_future_samples(obs_dim: int, history_frames: int, privileged: int) -> int:
    """Recover ``future_samples`` from a checkpoint's observation width.

    ``obs_dim = TARGET_FRAME_DIM * (1 + m) + FUTURE_SAMPLE_DIM * n + privileged``.
    Raises when the width does not match any integer ``n`` -- a mismatch here
    means the demo config and the checkpoint disagree and must not be papered
    over.
    """
    base = TARGET_FRAME_DIM * (1 + int(history_frames)) + int(privileged)
    remainder = int(obs_dim) - base
    _require(remainder >= 0,
             f"obs_dim {obs_dim} is smaller than the base width {base} "
             f"(m={history_frames}, privileged={privileged})")
    _require(remainder % FUTURE_SAMPLE_DIM == 0,
             f"obs_dim {obs_dim} - base {base} = {remainder} is not a multiple "
             f"of {FUTURE_SAMPLE_DIM}; the checkpoint layout does not match "
             f"m={history_frames}, privileged={privileged}")
    return remainder // FUTURE_SAMPLE_DIM


def privileged_width(names: Union[Mapping[str, Any], List[str], Tuple[str, ...]]) -> int:
    """Total width of a privileged block (unknown names -> hard error)."""
    if isinstance(names, Mapping):
        names = tuple(names.keys())
    total = 0
    for name in names:
        _require(name in PRIVILEGED_FIELDS,
                 f"unknown privileged field {name!r}; known: "
                 f"{sorted(PRIVILEGED_FIELDS)}")
        total += int(PRIVILEGED_FIELDS[name])
    return total


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoConfig:
    """Output video settings.  ``format`` defaults to MP4 (imageio-ffmpeg)."""

    name: str = "stage"
    fmt: str = "mp4"
    fps: int = 20
    width_in: float = 12.8
    height_in: float = 7.2
    dpi: int = 100
    bitrate: int = 4000
    trail: int = 220          # trail points kept per panel
    sidecar: bool = True       # also write <name>.json next to the video

    @property
    def pixels(self) -> Tuple[int, int]:
        return (int(round(self.width_in * self.dpi)),
                int(round(self.height_in * self.dpi)))

    @classmethod
    def parse(cls, raw: Optional[Mapping[str, Any]], where: str) -> "VideoConfig":
        raw = raw or {}
        _require(isinstance(raw, Mapping), f"{where} must be a mapping")
        fmt = str(_get(raw, "format", "mp4")).lower()
        _require(fmt in VIDEO_FORMATS,
                 f"{where}.format must be one of {VIDEO_FORMATS}, got {fmt!r}")
        return cls(
            name=str(_get(raw, "name", "stage")),
            fmt=fmt,
            fps=_int(_get(raw, "fps", 20), f"{where}.fps", lo=1, hi=120),
            width_in=_number(_get(raw, "width_in", 12.8), f"{where}.width_in",
                             lo=2.0, hi=64.0),
            height_in=_number(_get(raw, "height_in", 7.2), f"{where}.height_in",
                              lo=2.0, hi=64.0),
            dpi=_int(_get(raw, "dpi", 100), f"{where}.dpi", lo=40, hi=400),
            bitrate=_int(_get(raw, "bitrate", 4000), f"{where}.bitrate", lo=200, hi=60000),
            trail=_int(_get(raw, "trail", 220), f"{where}.trail", lo=0, hi=20000),
            sidecar=bool(_get(raw, "sidecar", True)),
        )


@dataclass(frozen=True)
class CameraConfig:
    """3-D view.  ``follow`` keeps the drone centred; ``fixed`` locks the box."""

    mode: str = "follow"
    distance: float = 14.0
    azimuth_deg: float = 35.0
    elevation_deg: float = 22.0
    zoom: float = 1.0

    @classmethod
    def parse(cls, raw: Optional[Mapping[str, Any]], where: str) -> "CameraConfig":
        raw = raw or {}
        _require(isinstance(raw, Mapping), f"{where} must be a mapping")
        mode = str(_get(raw, "mode", "follow")).lower()
        _require(mode in CAMERA_MODES,
                 f"{where}.mode must be one of {CAMERA_MODES}, got {mode!r}")
        return cls(
            mode=mode,
            distance=_number(_get(raw, "distance", 14.0), f"{where}.distance",
                             lo=1.0, hi=500.0),
            azimuth_deg=_number(_get(raw, "azimuth_deg", 35.0), f"{where}.azimuth_deg",
                                lo=-360.0, hi=360.0),
            elevation_deg=_number(_get(raw, "elevation_deg", 22.0),
                                  f"{where}.elevation_deg", lo=5.0, hi=85.0),
            zoom=float(_get(raw, "zoom", 1.0)),
        )


@dataclass(frozen=True)
class ObservationConfig:
    """Observation layout fed to the env.

    ``auto_fit`` (default) reconciles ``future_samples`` with the checkpoint's
    own ``observation_space`` width, so a demo can never silently feed a policy
    the wrong number of columns.
    """

    history_frames: int = 3
    history_skip: int = 2
    future_samples: int = 3
    future_skip: int = 2
    predictor: str = "const_vel"
    future_source: str = "true"
    privileged_fields: Tuple[str, ...] = ()
    auto_fit: bool = True

    @classmethod
    def parse(cls, raw: Optional[Mapping[str, Any]], where: str) -> "ObservationConfig":
        from src.utils.config_loader import PREDICTORS, FUTURE_SOURCES

        raw = raw or {}
        _require(isinstance(raw, Mapping), f"{where} must be a mapping")
        priv_raw = _get(raw, "privileged_fields", ())
        if isinstance(priv_raw, str):
            priv_raw = [p.strip() for p in priv_raw.split(",") if p.strip()]
        priv = tuple(str(p) for p in priv_raw)
        privileged_width(priv)          # validates the names eagerly
        predictor = str(_get(raw, "predictor", "const_vel"))
        _require(predictor in PREDICTORS,
                 f"{where}.predictor must be one of {PREDICTORS}, got {predictor!r}")
        source = str(_get(raw, "future_source", "true"))
        _require(source in FUTURE_SOURCES,
                 f"{where}.future_source must be one of {FUTURE_SOURCES}, got {source!r}")
        return cls(
            history_frames=_int(_get(raw, "history_frames", 3),
                                f"{where}.history_frames", lo=0, hi=64),
            history_skip=_int(_get(raw, "history_skip", 2), f"{where}.history_skip",
                              lo=1, hi=64),
            future_samples=_int(_get(raw, "future_samples", 3),
                                f"{where}.future_samples", lo=0, hi=64),
            future_skip=_int(_get(raw, "future_skip", 2), f"{where}.future_skip",
                             lo=1, hi=64),
            predictor=predictor,
            future_source=source,
            privileged_fields=priv,
            auto_fit=bool(_get(raw, "auto_fit", True)),
        )

    @property
    def privileged_width(self) -> int:
        return privileged_width(self.privileged_fields)

    def fitted(self, obs_dim: Optional[int]) -> "ObservationConfig":
        """Return a copy whose ``future_samples`` matches ``obs_dim``."""
        if not self.auto_fit or obs_dim is None or self.history_frames == 0:
            return self
        if not self.privileged_fields and not obs_dim:
            return self
        n = infer_future_samples(obs_dim, self.history_frames, self.privileged_width)
        if n == self.future_samples:
            return self
        from dataclasses import replace
        return replace(self, future_samples=n)

    def env_kwargs(self) -> Dict[str, Any]:
        return {
            "history_frames": self.history_frames,
            "history_skip": self.history_skip,
            "future_samples": self.future_samples,
            "future_skip": self.future_skip,
            "predictor": self.predictor,
            "future_source": self.future_source,
            "privileged_fields": tuple(self.privileged_fields),
        }


@dataclass(frozen=True)
class ModelRef:
    """A checkpoint uploaded for this stage."""

    path: Optional[Path] = None
    algo: str = "PPO"
    deterministic: bool = True
    model_id: str = ""

    @property
    def is_set(self) -> bool:
        return self.path is not None

    @classmethod
    def parse(cls, raw: Optional[Mapping[str, Any]], where: str) -> "ModelRef":
        raw = raw or {}
        _require(isinstance(raw, Mapping), f"{where} must be a mapping")
        text = str(_get(raw, "path", "")).strip()
        path = Path(text).expanduser() if text else None
        return cls(
            path=path,
            algo=str(_get(raw, "algo", "PPO")).upper(),
            deterministic=bool(_get(raw, "deterministic", True)),
            model_id=str(_get(raw, "model_id", "")),
        )

    def resolve(self, *, extra_roots: Sequence[Path] = ()) -> Path:
        """Absolute path, optionally looked up under bind-mounted roots."""
        assert self.path is not None
        cand = self.path
        if cand.is_absolute() and cand.exists():
            return cand
        roots = [self.path] + [Path(r) for r in extra_roots]
        for root in roots:
            trial = Path(root) / cand
            if trial.exists():
                return trial
            trial = Path(root) / cand.name
            if trial.is_file():
                return trial
        raise FileNotFoundError(
            f"{where_hint(self.path)} not found; searched {[str(r) for r in roots]}. "
            f"Mount the checkpoint directory (see docker-compose.yml volumes)."
        )


def where_hint(path: Path) -> str:
    return f"model {path}"


@dataclass(frozen=True)
class InstructionConfig:
    """The fixed instruction set used when ``control == "instructions"``."""

    mode: str = "guidance"
    steps: Tuple[Tuple[float, Tuple[float, float, float, float]], ...] = ()
    gains: Mapping[str, float] = field(default_factory=dict)
    max_tilt_deg: float = 25.0
    max_speed: float = 6.0
    max_stick: float = 0.16
    target: str = "waypoint"      # waypoint | target | hover
    lead_time_s: float = 0.0

    @classmethod
    def parse(cls, raw: Optional[Mapping[str, Any]], where: str) -> "InstructionConfig":
        raw = raw or {}
        _require(isinstance(raw, Mapping), f"{where} must be a mapping")
        mode = str(_get(raw, "mode", "guidance")).lower()
        _require(mode in INSTRUCTION_MODES,
                 f"{where}.mode must be one of {INSTRUCTION_MODES}, got {mode!r}")
        steps: List[Tuple[float, Tuple[float, float, float, float]]] = []
        for i, item in enumerate(_get(raw, "steps", ()) or ()):
            where_i = f"{where}.steps[{i}]"
            _require(isinstance(item, Mapping), f"{where_i} must be a mapping")
            action = _get(item, "action", None)
            _require(isinstance(action, (list, tuple)) and len(action) == 4,
                     f"{where_i}.action must be a 4-element list "
                     f"[thrust, roll, pitch, yaw]")
            vec = tuple(_number(a, f"{where_i}.action[{j}]", lo=-1.0, hi=1.0)
                        for j, a in enumerate(action))
            steps.append((_number(_get(item, "t", 0.0), f"{where_i}.t", lo=0.0), vec))
        steps.sort(key=lambda s: s[0])
        for i in range(1, len(steps)):
            _require(steps[i][0] > steps[i - 1][0],
                     f"{where}.steps must have strictly increasing t")
        target = str(_get(raw, "target", "waypoint")).lower()
        _require(target in ("waypoint", "target", "hover"),
                 f"{where}.target must be waypoint|target|hover, got {target!r}")
        if mode == "open_loop":
            _require(bool(steps),
                     f"{where}.steps is required when mode == 'open_loop'")
        gains = {str(k): float(v) for k, v in (_get(raw, "gains", {}) or {}).items()}
        return cls(
            mode=mode,
            steps=tuple(steps),
            gains=gains,
            max_tilt_deg=_number(_get(raw, "max_tilt_deg", 25.0),
                                 f"{where}.max_tilt_deg", lo=1.0, hi=80.0),
            max_speed=_number(_get(raw, "max_speed", 6.0), f"{where}.max_speed",
                              lo=0.1, hi=60.0),
            max_stick=_number(_get(raw, "max_stick", 0.16), f"{where}.max_stick",
                              lo=0.01, hi=1.0),
            target=target,
            lead_time_s=_number(_get(raw, "lead_time_s", 0.0),
                                f"{where}.lead_time_s", lo=0.0, hi=10.0),
        )


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageDemoConfig:
    """One stage's demo/render configuration."""

    stage: int
    path: Path
    title: str = ""
    control: str = "instructions"
    seed: int = 0
    segments: int = 1
    segment_seconds: float = 3.0
    reset_each_segment: bool = True
    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    instructions: InstructionConfig = field(default_factory=InstructionConfig)
    model: ModelRef = field(default_factory=ModelRef)
    target_alt: Union[float, Tuple[float, float]] = 5.0
    stage_overrides: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    # -- derived ---------------------------------------------------------

    @property
    def stage_key(self) -> str:
        return f"stage_{self.stage}"

    @property
    def steps_per_segment(self) -> int:
        from src.physics.constants import PH_DT
        return max(1, int(round(self.segment_seconds / float(PH_DT))))

    @property
    def total_seconds(self) -> float:
        return self.segments * self.segment_seconds

    def stage_config(self):
        """The built-in curriculum stage with ``stage_overrides`` applied."""
        base = STAGES.get(self.stage_key)
        _require(base is not None,
                 f"unknown stage {self.stage_key}; known: {sorted(STAGES)}")
        if not self.stage_overrides:
            return base
        return apply_stage_overrides(
            base, self.stage_overrides, sid=self.stage_key,
            source=f"{self.path.name}: stages.{self.stage_key}",
        )

    def env_kwargs(self, *, obs_dim: Optional[int] = None) -> Dict[str, Any]:
        """Keyword arguments for ``InterceptorBaseEnv``."""
        obs = self.observation.fitted(obs_dim)
        kwargs = dict(obs.env_kwargs())
        kwargs["target_alt"] = self.target_alt
        return kwargs

    def video_path(self, out_dir: Union[str, Path]) -> Path:
        out = Path(out_dir)
        stem = self.video.name if self.video.name != "stage" else self.stage_key
        suffix = "png" if self.video.fmt == "png" else self.video.fmt
        return out / f"{stem}.{suffix}"

    def sidecar_path(self, out_dir: Union[str, Path]) -> Path:
        return Path(out_dir) / f"{self.stage_key}.json"

    # -- parsing ---------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: Path) -> "StageDemoConfig":
        _require(isinstance(raw, Mapping), f"{path}: top level must be a JSON object")
        where = path.name
        schema = str(_get(raw, "schema", SCHEMA_ID))
        _require(schema == SCHEMA_ID,
                 f"{where}: schema must be {SCHEMA_ID!r}, got {schema!r}")
        stage = _int(_get(raw, "stage", 0), f"{where}.stage", lo=1, hi=64)
        control = str(_get(raw, "control", "instructions")).lower()
        _require(control in CONTROL_MODES,
                 f"{where}.control must be one of {CONTROL_MODES}, got {control!r}")
        model = ModelRef.parse(raw.get("model"), f"{where}.model")
        if control == "model":
            _require(model.is_set,
                     f"{where}.control == 'model' requires a model.path")
        alt_raw = _get(raw, "target_alt", 5.0)
        if isinstance(alt_raw, (list, tuple)):
            _require(len(alt_raw) == 2, f"{where}.target_alt must have 2 entries")
            target_alt: Union[float, Tuple[float, float]] = (
                _number(alt_raw[0], f"{where}.target_alt[0]", lo=0.0, hi=200.0),
                _number(alt_raw[1], f"{where}.target_alt[1]", lo=0.0, hi=200.0),
            )
        else:
            target_alt = _number(alt_raw, f"{where}.target_alt", lo=0.0, hi=200.0)
        overrides = _get(raw, "stage_overrides", {}) or {}
        _require(isinstance(overrides, Mapping),
                 f"{where}.stage_overrides must be a mapping")
        cfg = cls(
            stage=stage,
            path=path,
            title=str(_get(raw, "title", f"Stage {stage}")),
            control=control,
            seed=_int(_get(raw, "seed", 1000 + stage), f"{where}.seed",
                      lo=0, hi=2 ** 31 - 1),
            segments=_int(_get(raw, "segments", 1), f"{where}.segments", lo=1, hi=200),
            segment_seconds=_number(_get(raw, "segment_seconds", 3.0),
                                   f"{where}.segment_seconds", lo=0.1, hi=600.0),
            reset_each_segment=bool(_get(raw, "reset_each_segment", True)),
            video=VideoConfig.parse(raw.get("video"), f"{where}.video"),
            camera=CameraConfig.parse(raw.get("camera"), f"{where}.camera"),
            observation=ObservationConfig.parse(raw.get("observation"),
                                                f"{where}.observation"),
            instructions=InstructionConfig.parse(raw.get("instructions"),
                                                 f"{where}.instructions"),
            model=model,
            target_alt=target_alt,
            stage_overrides=dict(overrides),
            notes=str(_get(raw, "notes", "")),
        )
        cfg.stage_config()          # validates stage + overrides eagerly
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema_id,
            "stage": self.stage,
            "title": self.title,
            "control": self.control,
            "seed": self.seed,
            "segments": self.segments,
            "segment_seconds": self.segment_seconds,
            "video": {
                "name": self.video.name,
                "format": self.video.fmt,
                "fps": self.video.fps,
                "width_in": self.video.width_in,
                "height_in": self.video.height_in,
                "dpi": self.video.dpi,
                "bitrate": self.video.bitrate,
            },
            "camera": {
                "mode": self.camera.mode,
                "distance": self.camera.distance,
                "azimuth_deg": self.camera.azimuth_deg,
                "elevation_deg": self.camera.elevation_deg,
            },
            "observation": {
                "history_frames": self.observation.history_frames,
                "history_skip": self.observation.history_skip,
                "future_samples": self.observation.future_samples,
                "future_skip": self.observation.future_skip,
                "predictor": self.observation.predictor,
                "future_source": self.observation.future_source,
                "privileged_fields": list(self.observation.privileged_fields),
                "auto_fit": self.observation.auto_fit,
            },
            "model": {
                "path": str(self.model.path) if self.model.path else None,
                "algo": self.model.algo,
                "model_id": self.model.model_id,
                "deterministic": self.model.deterministic,
            },
            "target_alt": (list(self.target_alt)
                           if isinstance(self.target_alt, tuple) else self.target_alt),
            "notes": self.notes,
        }

    @property
    def schema_id(self) -> str:
        return SCHEMA_ID


def load_stage_config(path: Union[str, Path]) -> StageDemoConfig:
    """Read and validate one stage JSON file."""
    p = Path(path).expanduser()
    _require(p.is_file(), f"stage config not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageDemoError(f"{p.name}: invalid JSON ({exc})") from exc
    return StageDemoConfig.from_dict(raw, path=p)


def discover_stage_configs(path: Union[str, Path]) -> List[Path]:
    """All ``stage_*.json`` files in a directory, ordered by stage number."""
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    _require(p.is_dir(), f"stage config path not found: {p}")
    found: List[Tuple[int, Path]] = []
    for candidate in sorted(p.glob("*.json")):
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            stage = int(raw.get("stage", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if stage > 0:
            found.append((stage, candidate))
    found.sort()
    _require(bool(found), f"no stage_*.json configs in {p}")
    return [c for _, c in found]
