"""I/O helpers: audio load/save, JSON parameters, size/compression reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, *, downmix: bool = True) -> tuple[np.ndarray, int]:
    """Load WAV as float mono (or raise if stereo and downmix=False).

    Returns
    -------
    audio : np.ndarray, shape (n_samples,), float64 in roughly [-1, 1]
    sample_rate : int
    """
    path = Path(path)
    data, sr = sf.read(str(path), always_2d=True, dtype="float64")
    n_ch = data.shape[1]
    if n_ch > 1:
        if not downmix:
            raise ValueError(
                f"Stereo/multi-channel input ({n_ch} channels) rejected. "
                "Pass a mono WAV or enable downmix."
            )
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data, int(sr)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float audio as WAV (PCM_24)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    # Soft clip to avoid hard wrap if synth overshoots slightly
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def save_parameters(path: str | Path, params: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _round(obj: Any) -> Any:
        if isinstance(obj, float):
            # Compact JSON while remaining human-readable
            return float(f"{obj:.6g}")
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(f"{float(obj):.6g}")
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(_round(params), f, indent=2, allow_nan=False)


def load_parameters(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def count_scalars(obj: Any) -> int:
    """Recursively count numeric scalar leaves in a nested structure."""
    if isinstance(obj, (bool, np.bool_)):
        return 0
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return 1
    if isinstance(obj, str):
        return 0
    if isinstance(obj, dict):
        return sum(count_scalars(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_scalars(v) for v in obj)
    return 0


def pcm_stats(audio: np.ndarray, sample_rate: int, *, bytes_per_sample: int = 3) -> dict:
    """Raw PCM size estimates (default 24-bit mono)."""
    n = int(audio.shape[0])
    duration = n / float(sample_rate)
    raw_bytes = n * bytes_per_sample
    return {
        "duration_s": duration,
        "n_samples": n,
        "raw_bytes": raw_bytes,
        "raw_kb": raw_bytes / 1024.0,
    }


def parameter_storage_stats(params: dict, json_path: str | Path | None = None) -> dict:
    """Count scalars and estimate storage (JSON file size or 8 bytes/float)."""
    n_scalars = count_scalars(params)
    float_bytes = n_scalars * 8
    json_bytes = None
    if json_path is not None and Path(json_path).is_file():
        json_bytes = Path(json_path).stat().st_size
    est_bytes = json_bytes if json_bytes is not None else float_bytes
    return {
        "n_scalars": n_scalars,
        "float_bytes": float_bytes,
        "json_bytes": json_bytes,
        "est_bytes": est_bytes,
        "est_kb": est_bytes / 1024.0,
    }


def format_size_report(pcm: dict, param: dict) -> str:
    ratio = pcm["raw_bytes"] / max(param["est_bytes"], 1)
    lines = [
        "Original audio:",
        f"    duration: {pcm['duration_s']:.3f} seconds",
        f"    PCM samples: {pcm['n_samples']:,}",
        f"    raw size: {pcm['raw_kb']:.1f} KB",
        "",
        "Parameter representation:",
        f"    number of scalar parameters: {param['n_scalars']:,}",
        f"    estimated storage: {param['est_kb']:.1f} KB",
        f"    compression ratio: {ratio:.1f}:1",
    ]
    return "\n".join(lines)


def play_audio(audio: np.ndarray, sample_rate: int) -> bool:
    """Play mono audio via sounddevice if available. Returns True on success."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Playback skipped: sounddevice is not installed.")
        return False
    try:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        sd.play(audio, sample_rate, blocking=True)
        return True
    except Exception as exc:  # noqa: BLE001 — platform/device issues are common
        print(f"Playback skipped: {exc}")
        return False
