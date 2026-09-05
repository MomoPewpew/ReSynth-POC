"""Musically structured interpolation between two parameter dicts.

Design (intentionally not a flat lerp of every float):

- Time is normalized to [0, 1] per sound, then trajectories are resampled
  onto a shared grid so different durations/control-point counts align.
- Fundamental frequency blends in log-Hz (cents-like).
- Partials blend as harmonic *ratios* (f_k / f0) plus amplitudes in a
  log1p domain — better for inharmonicity and level differences.
- Phase is taken from the nearer endpoint (not interpolated).
- Envelope, transient, and noise blend on the normalized grid.
- Duration / peak / seed meta are linearly interpolated where meaningful.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _as_f64(x: Any) -> NDArray:
    return np.asarray(x, dtype=np.float64)


def _lerp(a: float, b: float, t: float) -> float:
    return float((1.0 - t) * a + t * b)


def _lerp_arr(a: NDArray, b: NDArray, t: float) -> NDArray:
    return ((1.0 - t) * a + t * b).astype(np.float64)


def _log_lerp(a: float, b: float, t: float, eps: float = 1e-6) -> float:
    a = max(float(a), eps)
    b = max(float(b), eps)
    return float(np.exp((1.0 - t) * np.log(a) + t * np.log(b)))


def _log1p_lerp(a: NDArray, b: NDArray, t: float) -> NDArray:
    a = np.maximum(_as_f64(a), 0.0)
    b = np.maximum(_as_f64(b), 0.0)
    return np.expm1((1.0 - t) * np.log1p(a) + t * np.log1p(b)).astype(np.float64)


def _norm_times(times: Any, duration: float) -> NDArray:
    t = _as_f64(times)
    dur = max(float(duration), 1e-9)
    u = np.clip(t / dur, 0.0, 1.0)
    if len(u) == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    u[0] = 0.0
    u[-1] = 1.0
    return u


def _resample_on_unit(
    times: Any,
    values: Any,
    duration: float,
    unit_grid: NDArray,
) -> NDArray:
    u = _norm_times(times, duration)
    v = _as_f64(values)
    if len(v) == 1:
        return np.full_like(unit_grid, float(v[0]))
    return np.interp(unit_grid, u, v).astype(np.float64)


def _resample_matrix_on_unit(
    times: Any,
    matrix: Any,
    duration: float,
    unit_grid: NDArray,
) -> NDArray:
    """matrix shape (bands, T) → (bands, len(unit_grid))."""
    m = _as_f64(matrix)
    if m.ndim != 2:
        m = np.atleast_2d(m)
    out = np.zeros((m.shape[0], len(unit_grid)), dtype=np.float64)
    for i in range(m.shape[0]):
        out[i] = _resample_on_unit(times, m[i], duration, unit_grid)
    return out


def interpolate_parameters(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
    t: float,
    *,
    n_grid: int | None = None,
) -> dict[str, Any]:
    """Return parameters at morph amount t (0=A, 1=B)."""
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return copy.deepcopy(params_a)
    if t >= 1.0:
        return copy.deepcopy(params_b)

    a = params_a
    b = params_b
    dur_a = float(a["meta"]["duration"])
    dur_b = float(b["meta"]["duration"])
    # Keep A's duration. Stretching/compressing while reusing STFT phases
    # desynchronizes phase from frequency and produces metallic/bell tones.
    dur = dur_a
    sr_a = int(a["meta"]["sample_rate"])
    sr_b = int(b["meta"]["sample_rate"])
    if sr_a != sr_b:
        raise ValueError(f"Sample rates differ: {sr_a} vs {sr_b}")
    sr = sr_a

    cfg_a = a.get("analysis_config") or {}
    cfg_b = b.get("analysis_config") or {}
    n_partials = min(len(a.get("partials") or []), len(b.get("partials") or []))
    n_bands = min(
        len((a.get("noise") or {}).get("band_freqs_hz") or []),
        len((b.get("noise") or {}).get("band_freqs_hz") or []),
    )
    # Keep A's normalized control-point schedule (attack densification).
    # A uniform linspace grid was remapping trajectories and making morphs
    # sound like bright partials/bells even when A≈B.
    times_a = a.get("partial_times") or (
        a["partials"][0].get("times") if a.get("partials") else [0.0, dur_a]
    )
    times_b = b.get("partial_times") or (
        b["partials"][0].get("times") if b.get("partials") else [0.0, dur_b]
    )
    unit = _norm_times(times_a, dur_a)
    if n_grid is not None and n_grid != len(unit):
        unit = np.linspace(0.0, 1.0, int(n_grid))
    n_grid = len(unit)

    # --- fundamental (log-Hz) ---
    f0_a = _resample_on_unit(
        a["fundamental"]["times"], a["fundamental"]["frequency_hz"], dur_a, unit
    )
    f0_b = _resample_on_unit(
        b["fundamental"]["times"], b["fundamental"]["frequency_hz"], dur_b, unit
    )
    f0 = np.exp((1.0 - t) * np.log(np.maximum(f0_a, 1.0)) + t * np.log(np.maximum(f0_b, 1.0)))

    # --- amplitude envelope ---
    env_a = _resample_on_unit(
        a["amplitude_envelope"]["times"], a["amplitude_envelope"]["values"], dur_a, unit
    )
    env_b = _resample_on_unit(
        b["amplitude_envelope"]["times"], b["amplitude_envelope"]["values"], dur_b, unit
    )
    env = _log1p_lerp(env_a, env_b, t)
    env_peak = max(float(np.max(env)), 1e-12)
    env = env / env_peak

    # --- partials: ratios + amps + phase ---
    partials_out: list[dict[str, Any]] = []
    for k in range(n_partials):
        pa = a["partials"][k]
        pb = b["partials"][k]
        freq_a = _resample_on_unit(times_a, pa["frequency_hz"], dur_a, unit)
        freq_b = _resample_on_unit(times_b, pb["frequency_hz"], dur_b, unit)
        ratio_a = freq_a / np.maximum(f0_a, 1.0)
        ratio_b = freq_b / np.maximum(f0_b, 1.0)
        ratio = _lerp_arr(ratio_a, ratio_b, t)
        freq = ratio * f0

        amp_a = _resample_on_unit(times_a, pa["amplitude"], dur_a, unit)
        amp_b = _resample_on_unit(times_b, pb["amplitude"], dur_b, unit)
        amp = _log1p_lerp(amp_a, amp_b, t)

        # Independently recorded notes have unrelated absolute phases.
        # Lerping those phases (or mixing one note's phase with another's
        # frequencies) creates metallic/bell artifacts. Keep A's measured
        # phase for all interior morphs; t=0/t=1 use exact endpoints via
        # early return above.
        phase_raw_a = pa.get("phase") or [0.0]
        if len(phase_raw_a) >= 2:
            phase = _resample_on_unit(
                times_a, np.unwrap(_as_f64(phase_raw_a)), dur_a, unit
            ).tolist()
        else:
            phase = [float(phase_raw_a[0])]

        partials_out.append(
            {
                "id": k + 1,
                "frequency_hz": freq.tolist(),
                "amplitude": amp.tolist(),
                "phase": phase,
            }
        )

    abs_times = (unit * dur).tolist()

    # --- transient ---
    tr_a = a.get("transient") or {}
    tr_b = b.get("transient") or {}
    tr_dur = _lerp(float(tr_a.get("duration_s", 0.05)), float(tr_b.get("duration_s", 0.05)), t)
    tr_start = _lerp(float(tr_a.get("start_s", 0.0)), float(tr_b.get("start_s", 0.0)), t)
    tr_start = float(np.clip(tr_start, 0.0, max(dur - tr_dur, 0.0)))

    def _blend_env_block(block_a: dict, block_b: dict, local_dur: float) -> dict:
        # Resample each side on its own duration, then lerp onto local_dur grid
        n_loc = max(8, min(32, n_grid // 4))
        u_loc = np.linspace(0.0, 1.0, n_loc)
        dur_ea = max(float((block_a.get("times") or [0.0, 1e-3])[-1]), 1e-6)
        dur_eb = max(float((block_b.get("times") or [0.0, 1e-3])[-1]), 1e-6)
        va = _resample_on_unit(block_a.get("times", [0, dur_ea]), block_a.get("values", [0, 0]), dur_ea, u_loc)
        vb = _resample_on_unit(block_b.get("times", [0, dur_eb]), block_b.get("values", [0, 0]), dur_eb, u_loc)
        return {"times": (u_loc * local_dur).tolist(), "values": _log1p_lerp(va, vb, t).tolist()}

    env_a_tr = tr_a.get("amplitude_envelope") or {"times": [0.0, tr_dur], "values": [0.0, 0.0]}
    env_b_tr = tr_b.get("amplitude_envelope") or {"times": [0.0, tr_dur], "values": [0.0, 0.0]}
    tr_env = _blend_env_block(env_a_tr, env_b_tr, tr_dur)

    spec_a = tr_a.get("spectral_envelope") or {}
    spec_b = tr_b.get("spectral_envelope") or {}
    n_tr = max(8, min(32, n_grid // 4))
    u_tr = np.linspace(0.0, 1.0, n_tr)
    dur_sa = max(float((spec_a.get("times") or [0.0, 1e-3])[-1]), 1e-6)
    dur_sb = max(float((spec_b.get("times") or [0.0, 1e-3])[-1]), 1e-6)
    bands_a = _as_f64(spec_a.get("band_freqs_hz") or list(range(n_bands)))
    bands_b = _as_f64(spec_b.get("band_freqs_hz") or list(range(n_bands)))
    n_b_tr = min(len(bands_a), len(bands_b), n_bands if n_bands else len(bands_a))
    mag_a = _resample_matrix_on_unit(
        spec_a.get("times", [0, dur_sa]),
        _as_f64(spec_a.get("band_magnitudes") or np.zeros((n_b_tr, 2)))[:n_b_tr],
        dur_sa,
        u_tr,
    )
    mag_b = _resample_matrix_on_unit(
        spec_b.get("times", [0, dur_sb]),
        _as_f64(spec_b.get("band_magnitudes") or np.zeros((n_b_tr, 2)))[:n_b_tr],
        dur_sb,
        u_tr,
    )
    mag_tr = _log1p_lerp(mag_a, mag_b, t)
    # renormalize columns
    for col in range(mag_tr.shape[1]):
        s = mag_tr[:, col].sum()
        if s > 1e-12:
            mag_tr[:, col] /= s
    band_freqs_tr = _lerp_arr(bands_a[:n_b_tr], bands_b[:n_b_tr], t)

    transient = {
        "start_s": tr_start,
        "duration_s": tr_dur,
        "amplitude_envelope": tr_env,
        "spectral_envelope": {
            "times": (u_tr * tr_dur).tolist(),
            "band_freqs_hz": band_freqs_tr.tolist(),
            "band_magnitudes": mag_tr.tolist(),
        },
    }

    # --- noise ---
    na = a.get("noise") or {}
    nb = b.get("noise") or {}
    amp_na = _resample_on_unit(na.get("times", [0, dur_a]), na.get("amplitude", [0, 0]), dur_a, unit)
    amp_nb = _resample_on_unit(nb.get("times", [0, dur_b]), nb.get("amplitude", [0, 0]), dur_b, unit)
    noise_amp = _log1p_lerp(amp_na, amp_nb, t)

    bf_a = _as_f64(na.get("band_freqs_hz") or [])
    bf_b = _as_f64(nb.get("band_freqs_hz") or [])
    n_b = min(len(bf_a), len(bf_b))
    bf = _lerp_arr(bf_a[:n_b], bf_b[:n_b], t)
    bm_a = _resample_matrix_on_unit(
        na.get("times", [0, dur_a]),
        _as_f64(na.get("band_magnitudes") or np.zeros((n_b, 2)))[:n_b],
        dur_a,
        unit,
    )
    bm_b = _resample_matrix_on_unit(
        nb.get("times", [0, dur_b]),
        _as_f64(nb.get("band_magnitudes") or np.zeros((n_b, 2)))[:n_b],
        dur_b,
        unit,
    )
    bm = _log1p_lerp(bm_a, bm_b, t)
    for col in range(bm.shape[1]):
        s = bm[:, col].sum()
        if s > 1e-12:
            bm[:, col] /= s

    # --- meta / config ---
    peak = _lerp(float(a["meta"]["peak_amplitude"]), float(b["meta"]["peak_amplitude"]), t)
    rms = _lerp(float(a["meta"].get("rms_amplitude", 0.0)), float(b["meta"].get("rms_amplitude", 0.0)), t)
    gain = _lerp(float(a.get("harmonic_gain", 1.0)), float(b.get("harmonic_gain", 1.0)), t)
    n_samples = int(round(dur * sr))
    seed = int(a["meta"].get("seed", 42) if t < 0.5 else b["meta"].get("seed", 42))

    # Prefer denser analysis settings
    cfg = copy.deepcopy(cfg_a if t < 0.5 else cfg_b)
    cfg["partial_control_points"] = n_grid
    cfg["envelope_points"] = n_grid
    cfg["noise_control_points"] = n_grid
    cfg["f0_control_points"] = n_grid
    cfg["max_partials"] = n_partials
    cfg["spectral_bands"] = n_b

    return {
        "version": 1,
        "meta": {
            "sample_rate": sr,
            "duration": dur,
            "channels": 1,
            "original_path": None,
            "peak_amplitude": peak,
            "rms_amplitude": rms,
            "seed": seed,
            "n_samples": n_samples,
            "morph_t": t,
            "morph_from": a.get("meta", {}).get("original_path"),
            "morph_to": b.get("meta", {}).get("original_path"),
        },
        "fundamental": {
            "times": abs_times,
            "frequency_hz": f0.tolist(),
            "frequency_hz_mean": float(np.mean(f0)),
        },
        "harmonic_gain": gain,
        "amplitude_envelope": {"times": abs_times, "values": env.tolist()},
        "partial_times": abs_times,
        "partials": partials_out,
        "transient": transient,
        "noise": {
            "times": abs_times,
            "amplitude": noise_amp.tolist(),
            "band_freqs_hz": bf.tolist(),
            "band_magnitudes": bm.tolist(),
        },
        "analysis_config": cfg,
    }
