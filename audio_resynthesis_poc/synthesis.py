"""Resynthesis from structured parameters (no original PCM required)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _interp(times: list | NDArray, values: list | NDArray, query: NDArray) -> NDArray:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if len(t) == 0:
        return np.zeros_like(query)
    if len(t) == 1:
        return np.full_like(query, float(v[0]))
    return np.interp(query, t, v).astype(np.float64)


def _mq_cubic_phase_segment(
    phase0: float,
    phase1: float,
    f0: float,
    f1: float,
    n_samples: int,
    sample_rate: int,
) -> NDArray:
    """McAulay–Quatieri cubic phase between two control points."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    T = n_samples / float(sample_rate)
    if T <= 0:
        return np.full(n_samples, phase0, dtype=np.float64)

    w0 = 2.0 * np.pi * f0
    w1 = 2.0 * np.pi * f1
    M = int(np.round(((phase1 - phase0) - 0.5 * (w0 + w1) * T) / (2.0 * np.pi)))
    ph1 = phase1 - 2.0 * np.pi * M

    a0 = phase0
    a1 = w0
    a2 = (3.0 * (ph1 - phase0) - (2.0 * w0 + w1) * T) / (T * T)
    a3 = (2.0 * (phase0 - ph1) + (w0 + w1) * T) / (T * T * T)

    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return a0 + a1 * t + a2 * t**2 + a3 * t**3


def _synthesize_partials_stft(
    params: dict[str, Any],
    n: int,
    sr: int,
) -> NDArray:
    """Rebuild a sparse harmonic STFT from partial control points and ISTFT.

    More robust than time-domain oscillators for real instrument tones.
    Still parametric: only control-point amp/freq/phase are used (no PCM).
    """
    import librosa

    cfg = params.get("analysis_config") or {}
    n_fft = int(cfg.get("stft_n_fft", 4096))
    hop = int(cfg.get("stft_hop", 512))
    duration = n / float(sr)
    shared_times = params.get("partial_times")

    # Empty STFT shaped like a real analysis
    n_frames = 1 + n // hop
    S = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    bin_hz = sr / float(n_fft)
    window = np.hanning(n_fft)
    amp_to_mag = float(np.sum(window)) / 2.0  # inverse of analysis mag_to_amp

    for partial in params.get("partials", []):
        times = np.asarray(
            partial.get("times") or shared_times or [0.0, duration],
            dtype=np.float64,
        )
        freq_cp = np.asarray(partial["frequency_hz"], dtype=np.float64)
        amp_cp = np.asarray(partial["amplitude"], dtype=np.float64)
        phase_cp = np.asarray(partial.get("phase") or [0.0], dtype=np.float64)

        freq = _interp(times, freq_cp, frame_times)
        amp = _interp(times, amp_cp, frame_times)
        if len(phase_cp) >= 2:
            # Interpolate unwrapped phase in a continuity-preserving way
            phase = np.interp(frame_times, times, np.unwrap(phase_cp)).astype(np.float64)
        else:
            # IF from initial phase across frames
            phase0 = float(phase_cp[0])
            dphi = 2.0 * np.pi * freq * hop / sr
            phase = phase0 + np.cumsum(dphi)
            phase = np.concatenate([[phase0], phase[:-1]])

        for ti in range(n_frames):
            f = float(freq[ti])
            a = float(amp[ti])
            if a <= 1e-12 or f <= 0 or f >= sr * 0.5:
                continue
            bin_f = f / bin_hz
            k0 = int(np.floor(bin_f))
            frac = bin_f - k0
            mag = a * amp_to_mag
            c = mag * np.exp(1j * phase[ti])
            # Split across adjacent bins (linear) to reduce scalloping
            if 0 <= k0 < S.shape[0]:
                S[k0, ti] += c * (1.0 - frac)
            if 0 <= k0 + 1 < S.shape[0]:
                S[k0 + 1, ti] += c * frac

    y = librosa.istft(S, hop_length=hop, win_length=n_fft, length=n)
    return np.asarray(y, dtype=np.float64)


def synthesize_harmonics_only(
    params: dict[str, Any],
    *,
    apply_global_envelope: bool = False,
) -> NDArray:
    """Sum harmonic partials from frequency/amplitude/phase trajectories."""
    meta = params["meta"]
    sr = int(meta["sample_rate"])
    duration = float(meta["duration"])
    n = int(meta.get("n_samples") or round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr

    out = _synthesize_partials_stft(params, n, sr)

    if apply_global_envelope and "amplitude_envelope" in params:
        env = params["amplitude_envelope"]
        out *= _interp(env["times"], env["values"], t)

    return out


def _synthesize_banded_noise(
    times: list,
    amplitude: list,
    band_freqs_hz: list,
    band_magnitudes: list,
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    n_fft: int = 2048,
    hop: int = 512,
) -> NDArray:
    """STFT-shaped noise matched to a time-varying RMS envelope."""
    import librosa

    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)

    noise = rng.standard_normal(n_samples).astype(np.float64)
    S = librosa.stft(noise, n_fft=n_fft, hop_length=hop)
    phase = np.exp(1j * np.angle(S))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sample_rate, hop_length=hop)

    band_freqs = np.asarray(band_freqs_hz, dtype=np.float64)
    band_mag = np.asarray(band_magnitudes, dtype=np.float64)  # (bands, T_cp)
    if band_mag.ndim != 2:
        band_mag = np.ones((len(band_freqs), len(frame_times)), dtype=np.float64)
    n_bands = len(band_freqs)

    amp_frames = _interp(times, amplitude, frame_times)
    mag_frames = np.zeros((n_bands, len(frame_times)), dtype=np.float64)
    for b in range(n_bands):
        src = band_mag[b] if b < band_mag.shape[0] else np.zeros(band_mag.shape[1])
        mag_frames[b] = _interp(times, src, frame_times)

    log_c = np.log(np.clip(band_freqs, 1.0, None))
    edges = np.zeros(n_bands + 1)
    edges[0] = max(freqs[1], band_freqs[0] / 1.5) if len(band_freqs) else freqs[1]
    edges[-1] = sample_rate * 0.5
    for i in range(1, n_bands):
        edges[i] = float(np.exp(0.5 * (log_c[i - 1] + log_c[i])))

    # Build relative magnitude shape across frequency
    shape = np.zeros_like(np.abs(S))
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not np.any(mask):
            k = int(np.argmin(np.abs(freqs - band_freqs[b])))
            shape[k, :] = mag_frames[b]
        else:
            shape[mask, :] = mag_frames[b][None, :]

    # Avoid zero shape frames
    col_sum = shape.sum(axis=0, keepdims=True)
    col_sum = np.maximum(col_sum, 1e-12)
    shape = shape / col_sum

    # Unit-shape noise, then scale each frame toward target STFT energy from RMS
    # Approximate: time-domain RMS ≈ sqrt(mean(mag^2)) * scale; use iterative-ish gain
    target_mag = shape * (1.0 + 0.0)  # relative
    # Give all frames equal total magnitude energy then scale by amp via istft post
    S_hat = target_mag * phase
    y = librosa.istft(S_hat, hop_length=hop, length=n_samples)
    y = np.asarray(y, dtype=np.float64)

    # Apply smooth time-domain RMS envelope
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    env = _interp(times, amplitude, t)
    # Normalize current signal to unit RMS in sliding sense, then apply env
    # Simple: global normalize then multiply envelope (envelope already absolute RMS-ish)
    cur = max(float(np.sqrt(np.mean(y**2))), 1e-12)
    y = y / cur
    # Convert RMS envelope to amplitude scale (for Gaussian noise, peak~factor*rms)
    y *= env * np.sqrt(2.0)
    return y


def synthesize_transient(params: dict[str, Any], rng: np.random.Generator) -> NDArray:
    meta = params["meta"]
    sr = int(meta["sample_rate"])
    duration = float(meta["duration"])
    n = int(meta.get("n_samples") or round(duration * sr))
    out = np.zeros(n, dtype=np.float64)

    tr = params.get("transient") or {}
    start = float(tr.get("start_s", 0.0))
    seg_dur = float(tr.get("duration_s", 0.0))
    if seg_dur <= 0:
        return out

    n_seg = max(1, int(round(seg_dur * sr)))
    env = tr.get("amplitude_envelope") or {"times": [0.0, seg_dur], "values": [0.0, 0.0]}
    spec = tr.get("spectral_envelope") or {}
    band_freqs = spec.get("band_freqs_hz") or [1000.0]
    band_mags = spec.get("band_magnitudes") or [[1.0, 1.0]]
    spec_times = spec.get("times") or env["times"]

    # Use relative shape with unit amplitude then apply envelope
    ones_amp = [1.0] * len(spec_times)
    seg = _synthesize_banded_noise(
        times=spec_times,
        amplitude=ones_amp,
        band_freqs_hz=band_freqs,
        band_magnitudes=band_mags,
        n_samples=n_seg,
        sample_rate=sr,
        rng=rng,
        n_fft=min(512, max(128, n_seg)),
        hop=max(32, min(128, n_seg // 4) or 32),
    )
    t_seg = np.arange(n_seg, dtype=np.float64) / sr
    env_v = _interp(env["times"], env["values"], t_seg)
    # Normalize noise then apply envelope absolute values
    seg = seg / max(np.max(np.abs(seg)), 1e-12)
    seg *= env_v

    i0 = int(round(start * sr))
    i1 = min(n, i0 + n_seg)
    out[i0:i1] += seg[: i1 - i0]
    return out


def synthesize(params: dict[str, Any]) -> NDArray:
    """Full resynthesis: harmonics + transient + noise, then level restore."""
    meta = params["meta"]
    sr = int(meta["sample_rate"])
    duration = float(meta["duration"])
    n = int(meta.get("n_samples") or round(duration * sr))
    seed = int(meta.get("seed", 42))
    peak = float(meta.get("peak_amplitude", 1.0))
    rng = np.random.default_rng(seed)

    cfg = params.get("analysis_config") or {}
    n_fft = int(cfg.get("stft_n_fft", 2048))
    hop = int(cfg.get("stft_hop", 512))

    harm = synthesize_harmonics_only(params, apply_global_envelope=False)
    if len(harm) < n:
        harm = np.pad(harm, (0, n - len(harm)))
    else:
        harm = harm[:n]

    transient = synthesize_transient(params, rng)

    noise_p = params.get("noise") or {}
    noise = _synthesize_banded_noise(
        times=noise_p.get("times", [0.0, duration]),
        amplitude=noise_p.get("amplitude", [0.0, 0.0]),
        band_freqs_hz=noise_p.get("band_freqs_hz", [1000.0]),
        band_magnitudes=noise_p.get(
            "band_magnitudes", [[0.0, 0.0]]
        ),
        n_samples=n,
        sample_rate=sr,
        rng=rng,
        n_fft=n_fft,
        hop=hop,
    )

    # Partials were analyzed on peak-normalized audio; scale back to original level.
    out = (harm + transient + noise) * peak

    # Soft safety clip only if numerical overshoot is extreme
    cur_peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if cur_peak > 1.05 * max(peak, 1e-12):
        out *= (1.05 * peak) / cur_peak

    return out
