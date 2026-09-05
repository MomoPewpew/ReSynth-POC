"""Parametric pitch shifting that keeps phase consistent with frequency."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _as_f64(x: Any) -> NDArray:
    return np.asarray(x, dtype=np.float64)


def _if_phase(freq: NDArray, times: NDArray, phase0: float) -> NDArray:
    """Cumulative instantaneous-frequency phase on a control-point time grid."""
    freq = _as_f64(freq)
    times = _as_f64(times)
    if len(times) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(times) == 1:
        return np.array([phase0], dtype=np.float64)
    dt = np.diff(times, prepend=times[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0
    ip = phase0 + np.cumsum(2.0 * np.pi * freq * dt)
    return ip - ip[0] + phase0


def _formant_amps(
    freqs_old: NDArray,
    amps_old: NDArray,
    freqs_new: NDArray,
) -> NDArray:
    """Lookup original spectral envelope at the new partial frequencies."""
    n_partials, n_times = freqs_old.shape
    out = np.zeros_like(amps_old)
    for ti in range(n_times):
        f = freqs_old[:, ti]
        a = np.maximum(amps_old[:, ti], 0.0)
        order = np.argsort(f)
        f_s = f[order]
        a_s = a[order]
        if len(f_s) >= 2:
            uniq = np.concatenate([[True], np.diff(f_s) > 1e-6])
            f_s = f_s[uniq]
            a_s = a_s[uniq]
        q = freqs_new[:, ti]
        f_pos = np.maximum(f_s, 1.0)
        q_pos = np.maximum(q, 1.0)
        if len(f_s) == 0:
            continue
        if len(f_s) == 1:
            out[:, ti] = a_s[0]
            continue
        out[:, ti] = np.interp(np.log(q_pos), np.log(f_pos), a_s, left=0.0, right=0.0)
    return out


def transpose_parameters(
    params: dict[str, Any],
    semitones: float,
    *,
    keep_above_nyquist: bool = False,
    formant_preserve: float = 0.55,
    noise_pitch_track: float = 0.0,
    register_compensate: bool = True,
) -> dict[str, Any]:
    """Transpose parameters by ``semitones``.

    Upward shifts of piano-like tones go "bell/chime" for three reasons:
    1. Classic stretch keeps a loud 2nd partial (hollow octave).
    2. Full formant preserve leaves a near-sine (weak overtones).
    3. Hammer/noise is under-represented while sustain stays long.

    Defaults therefore *blend* formant remap (~0.55), keep noise bands in
    absolute Hz, and apply register compensation (more hammer, faster decay)
    as pitch rises.
    """
    out = copy.deepcopy(params)
    ratio = float(2.0 ** (semitones / 12.0))
    sr = int(out["meta"]["sample_rate"])
    nyquist = sr * 0.5
    formant_preserve = float(np.clip(formant_preserve, 0.0, 1.0))
    noise_pitch_track = float(np.clip(noise_pitch_track, 0.0, 1.0))
    # 0 at unison → 1 at +1 octave
    up = float(np.clip(semitones / 12.0, 0.0, 2.0))

    # Fundamental
    f0 = _as_f64(out["fundamental"]["frequency_hz"]) * ratio
    out["fundamental"]["frequency_hz"] = f0.tolist()
    out["fundamental"]["frequency_hz_mean"] = float(np.mean(f0))

    times = _as_f64(out.get("partial_times") or out["partials"][0].get("times", [0.0]))
    partials = out.get("partials") or []
    if not partials:
        out["meta"]["transpose_semitones"] = float(semitones)
        out["meta"]["transpose_ratio"] = ratio
        return out

    freq_old = np.stack([_as_f64(p["frequency_hz"]) for p in partials], axis=0)
    amp_old = np.stack([_as_f64(p["amplitude"]) for p in partials], axis=0)
    freq_new = freq_old * ratio

    amp_scaled = amp_old.copy()
    amp_formant = _formant_amps(freq_old, amp_old, freq_new)
    amp_new = (1.0 - formant_preserve) * amp_scaled + formant_preserve * amp_formant

    # Soften upper partials further when rising (reduces chime partial stack)
    if register_compensate and up > 0:
        # Attenuate partials above ~2.5 kHz more as we go up
        bright = np.clip((freq_new - 1800.0) / 4000.0, 0.0, 1.0)
        amp_new = amp_new * (1.0 - 0.65 * up * bright)

    mute = freq_new >= (nyquist * 0.98)
    lost = float(np.mean(amp_old[mute])) if np.any(mute) else 0.0

    if not keep_above_nyquist:
        amp_new = amp_new.copy()
        amp_new[mute] = 0.0
        freq_new = np.minimum(freq_new, nyquist * 0.99)

    for i, partial in enumerate(partials):
        fo = freq_old[i]
        fn = freq_new[i]
        amp = amp_new[i]
        phase_raw = _as_f64(partial.get("phase") or [0.0])

        if len(phase_raw) >= 2 and len(phase_raw) == len(fo):
            ph = np.unwrap(phase_raw)
            if_old = _if_phase(fo, times, float(ph[0]))
            residual = ph - if_old
            if_new = _if_phase(fn, times, float(ph[0]))
            phase_new = if_new + residual
        else:
            phase_new = phase_raw

        partial["frequency_hz"] = fn.tolist()
        partial["amplitude"] = amp.tolist()
        partial["phase"] = _as_f64(phase_new).tolist()

    def _blend_bands(block: dict) -> None:
        if "band_freqs_hz" not in block:
            return
        bf = _as_f64(block["band_freqs_hz"])
        bf_pitched = bf * ratio
        bf_new = (1.0 - noise_pitch_track) * bf + noise_pitch_track * bf_pitched
        block["band_freqs_hz"] = np.minimum(bf_new, nyquist * 0.99).tolist()

    noise = out.get("noise") or {}
    _blend_bands(noise)

    # Register compensation: high piano notes are attack-noisy and short-lived.
    if register_compensate and up > 0:
        if "amplitude" in noise:
            # Strong hammer/air boost with pitch (was barely audible before)
            boost = 1.0 + 8.0 * up + 12.0 * up * up
            if lost > 1e-8:
                boost += 6.0 * up
            noise["amplitude"] = (_as_f64(noise["amplitude"]) * boost).tolist()

        # Faster decay on the global envelope
        env = out.get("amplitude_envelope") or {}
        if "times" in env and "values" in env:
            et = _as_f64(env["times"])
            ev = _as_f64(env["values"])
            # Extra exponential decay: ~2.5x faster at +1 octave
            decay = np.exp(-2.8 * up * et)
            env["values"] = (ev * decay).tolist()

        # Also taper partial amps over time a bit faster
        for partial in partials:
            amp = _as_f64(partial["amplitude"])
            decay = np.exp(-1.8 * up * times)
            partial["amplitude"] = (amp * decay).tolist()

    tr = out.get("transient") or {}
    spec = tr.get("spectral_envelope") or {}
    _blend_bands(spec)
    if register_compensate and up > 0 and "amplitude_envelope" in tr:
        env = tr["amplitude_envelope"]
        vals = _as_f64(env.get("values", []))
        # Hammer click much more present in the upper register
        env["values"] = (vals * (1.0 + 4.0 * up)).tolist()
        # Slightly longer transient window feels more "hit"
        tr["duration_s"] = float(tr.get("duration_s", 0.05)) * (1.0 + 0.5 * up)

    out["meta"]["transpose_semitones"] = float(semitones)
    out["meta"]["transpose_ratio"] = ratio
    out["meta"]["formant_preserve"] = formant_preserve
    out["meta"]["register_compensate"] = register_compensate
    return out


def pitch_shift_audio(
    audio: NDArray,
    sample_rate: int,
    semitones: float,
    *,
    bins_per_octave: int = 12,
) -> NDArray:
    """Phase-vocoder pitch shift of a time-domain signal (librosa)."""
    import librosa

    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if abs(semitones) < 1e-9:
        return y.copy()
    return np.asarray(
        librosa.effects.pitch_shift(
            y,
            sr=sample_rate,
            n_steps=float(semitones),
            bins_per_octave=bins_per_octave,
        ),
        dtype=np.float64,
    )


# Major scale intervals from the tonic (semitones)
MAJOR_SCALE_SEMITONES = [0, 2, 4, 5, 7, 9, 11, 12]

MAJOR_SCALE_DEGREE_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]
