"""Classical signal analysis → human-inspectable synthesis parameters.

No PCM samples are stored in the parameter dict. Trajectories are downsampled
control points, not full spectrograms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from numpy.typing import NDArray

from config import AnalysisConfig
from synthesis import synthesize_harmonics_only


def _attack_biased_times(
    duration: float,
    n_points: int,
    attack_fraction: float,
    attack_point_fraction: float,
) -> NDArray:
    """More control points early in the sound (attack), rest uniform."""
    n_points = max(2, n_points)
    n_attack = max(1, int(round(n_points * attack_point_fraction)))
    n_rest = n_points - n_attack
    t_attack_end = min(duration * attack_fraction, duration)
    t_attack = np.linspace(0.0, t_attack_end, n_attack, endpoint=False)
    if n_rest > 0:
        t_rest = np.linspace(t_attack_end, duration, n_rest)
        times = np.concatenate([t_attack, t_rest])
    else:
        times = np.linspace(0.0, duration, n_points)
    times = np.unique(np.clip(times, 0.0, duration))
    # Ensure exact endpoint coverage
    if times[0] != 0.0:
        times = np.concatenate([[0.0], times])
    if times[-1] != duration:
        times = np.concatenate([times, [duration]])
    # If uniqueness expanded beyond budget, resample uniformly on the biased grid
    if len(times) > n_points:
        idx = np.linspace(0, len(times) - 1, n_points).astype(int)
        times = times[idx]
        times[0] = 0.0
        times[-1] = duration
    return times.astype(np.float64)


def _interp_series(times: NDArray, values: NDArray, query_times: NDArray) -> NDArray:
    return np.interp(query_times, times, values).astype(np.float64)


def _downsample_traj(
    frame_times: NDArray,
    values: NDArray,
    n_points: int,
    duration: float,
    attack_fraction: float,
    attack_point_fraction: float,
) -> tuple[list[float], list[float]]:
    qt = _attack_biased_times(duration, n_points, attack_fraction, attack_point_fraction)
    qv = _interp_series(frame_times, values, qt)
    return qt.tolist(), qv.tolist()


def _rms_envelope(
    audio: NDArray,
    sample_rate: int,
    frame_length: int,
    hop: int,
) -> tuple[NDArray, NDArray]:
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop)
    return times, rms.astype(np.float64)


def _estimate_f0(
    audio: NDArray,
    sample_rate: int,
    hop: int,
    fmin: float = 65.0,
    fmax: float = 2000.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return frame times, f0 (NaN unvoiced), and voiced boolean."""
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop)
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced_flag, dtype=bool)
    # Fill unvoiced for harmonic tracking with median / forward-back fill
    filled = f0.copy()
    voiced_vals = filled[voiced & np.isfinite(filled)]
    fallback = float(np.median(voiced_vals)) if len(voiced_vals) else 220.0
    # forward fill
    last = fallback
    for i in range(len(filled)):
        if voiced[i] and np.isfinite(filled[i]):
            last = filled[i]
        else:
            filled[i] = last
    # backward fill for leading unvoiced
    first_voiced = np.where(voiced & np.isfinite(f0))[0]
    if len(first_voiced):
        filled[: first_voiced[0]] = f0[first_voiced[0]]
    else:
        filled[:] = fallback
    return times, filled, voiced


def _parabolic_mag_phase(
    mag: NDArray,
    phase: NDArray,
    bin_f: float,
) -> tuple[float, float]:
    """Interpolate magnitude (parabolic) and phase (linear) around bin_f."""
    n_bins = mag.shape[0] - 1
    if bin_f <= 0 or bin_f >= n_bins:
        k = int(np.clip(round(bin_f), 0, n_bins))
        return float(mag[k]), float(phase[k])
    k0 = int(np.floor(bin_f))
    k1 = min(k0 + 1, n_bins)
    frac = bin_f - k0
    # linear mag / phase (stable); parabolic refinement on mag if neighbors exist
    m_lin = (1 - frac) * mag[k0] + frac * mag[k1]
    # unwrap-safe phase interp via complex
    c = (1 - frac) * np.exp(1j * phase[k0]) + frac * np.exp(1j * phase[k1])
    p = float(np.angle(c))
    if 0 < k0 < n_bins:
        alpha = mag[k0 - 1]
        beta = mag[k0]
        gamma = mag[k0 + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            p_offset = 0.5 * (alpha - gamma) / denom
            p_offset = float(np.clip(p_offset, -1.0, 1.0))
            m_peak = beta - 0.25 * (alpha - gamma) * p_offset
            if abs(bin_f - k0) < 1.5:
                m_lin = max(m_peak, 0.0)
    return float(m_lin), p


def _parabolic_bin_refine(mag_col: NDArray, k: int) -> tuple[float, float]:
    """Return refined (bin_float, magnitude) around discrete peak bin k."""
    n_bins = len(mag_col) - 1
    k = int(np.clip(k, 1, n_bins - 1))
    alpha = float(mag_col[k - 1])
    beta = float(mag_col[k])
    gamma = float(mag_col[k + 1])
    denom = alpha - 2 * beta + gamma
    if abs(denom) > 1e-12:
        delta = 0.5 * (alpha - gamma) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        mag_peak = beta - 0.25 * (alpha - gamma) * delta
        return float(k + delta), max(mag_peak, 0.0)
    return float(k), beta


def _extract_partials(
    audio: NDArray,
    sample_rate: int,
    f0_times: NDArray,
    f0_hz: NDArray,
    cfg: AnalysisConfig,
    duration: float,
) -> list[dict[str, Any]]:
    n_fft = cfg.stft_n_fft
    hop = cfg.stft_hop
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)
    frame_times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sample_rate, hop_length=hop)
    # Align f0 to STFT frames
    f0_on_frames = _interp_series(f0_times, f0_hz, frame_times)
    # Hann coherent gain: |X| ≈ A * sum(window) / 2  ⇒  A ≈ 2|X|/sum(window)
    window = np.hanning(n_fft)
    mag_to_amp = 2.0 / float(np.sum(window))
    bin_hz = sample_rate / float(n_fft)
    search_frac = float(cfg.partial_search_fraction)

    partials: list[dict[str, Any]] = []
    # Shared time grid for all partials (compact + interpolable)
    t_cp = _attack_biased_times(
        duration,
        cfg.partial_control_points,
        cfg.attack_fraction,
        cfg.attack_point_fraction,
    ).tolist()

    for k in range(1, cfg.max_partials + 1):
        amps = np.zeros(len(frame_times), dtype=np.float64)
        phs = np.zeros(len(frame_times), dtype=np.float64)
        frqs = np.zeros(len(frame_times), dtype=np.float64)
        last_f = None
        for t_idx, f0 in enumerate(f0_on_frames):
            target = k * float(f0)
            if target >= sample_rate * 0.5 or target <= 0:
                amps[t_idx] = 0.0
                frqs[t_idx] = max(target, 0.0)
                phs[t_idx] = 0.0
                continue

            # Search for a local peak near the expected harmonic (inharmonicity)
            half_w = max(target * search_frac, bin_hz * 1.5)
            lo = max(bin_hz, target - half_w)
            hi = min(sample_rate * 0.5 - bin_hz, target + half_w)
            k_lo = max(1, int(np.floor(lo / bin_hz)))
            k_hi = min(mag.shape[0] - 2, int(np.ceil(hi / bin_hz)))
            if k_hi <= k_lo:
                bin_f = target / bin_hz
                m, p = _parabolic_mag_phase(mag[:, t_idx], phase[:, t_idx], bin_f)
                peak_f = target
            else:
                local = mag[k_lo : k_hi + 1, t_idx]
                peak_rel = int(np.argmax(local))
                peak_bin = k_lo + peak_rel
                bin_f, m = _parabolic_bin_refine(mag[:, t_idx], peak_bin)
                peak_f = bin_f * bin_hz
                # If the peak is weak / ambiguous, stay on the nominal harmonic
                med = float(np.median(local))
                if m < max(med * 1.5, 1e-6):
                    bin_f = target / bin_hz
                    m, p = _parabolic_mag_phase(mag[:, t_idx], phase[:, t_idx], bin_f)
                    peak_f = target
                else:
                    _, p = _parabolic_mag_phase(mag[:, t_idx], phase[:, t_idx], bin_f)

            # Light temporal smoothing of frequency to reduce bin jitter
            if last_f is not None:
                peak_f = 0.7 * peak_f + 0.3 * last_f
            last_f = peak_f

            amps[t_idx] = m * mag_to_amp
            phs[t_idx] = p
            frqs[t_idx] = peak_f

        # Unwrap phase along time for inspection; synth uses IF + phase[0]
        phs = np.unwrap(phs)

        a_cp = _interp_series(frame_times, amps, np.asarray(t_cp)).tolist()
        f_cp = _interp_series(frame_times, frqs, np.asarray(t_cp)).tolist()
        # Initial phase only — synthesis integrates instantaneous frequency.
        # (Full phase trajectories are large and were not improving reconstruction.)
        phase0 = float(phs[0]) if len(phs) else 0.0
        partials.append(
            {
                "id": k,
                "times": t_cp,
                "frequency_hz": f_cp,
                "amplitude": a_cp,
                "phase": [phase0],
            }
        )
    return partials


def _detect_transient_start(audio: NDArray, sample_rate: int) -> float:
    """First strong onset time in seconds (fallback 0)."""
    try:
        onsets = librosa.onset.onset_detect(
            y=audio, sr=sample_rate, units="time", backtrack=True
        )
        if len(onsets):
            return float(onsets[0])
    except Exception:  # noqa: BLE001
        pass
    # Energy rise fallback
    hop = 256
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop)[0]
    if len(rms) < 2:
        return 0.0
    thresh = 0.1 * float(np.max(rms))
    idx = int(np.argmax(rms > thresh))
    return float(librosa.frames_to_time(idx, sr=sample_rate, hop_length=hop))


def _band_freqs(n_bands: int, sample_rate: int) -> NDArray:
    """Log-spaced band center frequencies from ~40 Hz to Nyquist."""
    f_min = 40.0
    f_max = sample_rate * 0.5
    return np.geomspace(f_min, f_max, n_bands).astype(np.float64)


def _analyze_banded_spectrum(
    audio: NDArray,
    sample_rate: int,
    n_fft: int,
    hop: int,
    n_bands: int,
    n_time_points: int,
    duration: float,
    attack_fraction: float,
    attack_point_fraction: float,
) -> dict[str, Any]:
    """Time-varying coarse spectral envelope in log-spaced bands."""
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-12:
        band_freqs = _band_freqs(n_bands, sample_rate)
        times = _attack_biased_times(
            max(duration, 1e-3), n_time_points, attack_fraction, attack_point_fraction
        )
        return {
            "times": times.tolist(),
            "band_freqs_hz": band_freqs.tolist(),
            "band_magnitudes": np.zeros((n_bands, len(times))).tolist(),
            "amplitude": np.zeros(len(times)).tolist(),
        }

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sample_rate, hop_length=hop)
    band_freqs = _band_freqs(n_bands, sample_rate)
    # Band edges midway in log space
    log_c = np.log(band_freqs)
    edges = np.zeros(n_bands + 1)
    edges[0] = max(freqs[1], band_freqs[0] / 1.5)
    edges[-1] = sample_rate * 0.5
    for i in range(1, n_bands):
        edges[i] = float(np.exp(0.5 * (log_c[i - 1] + log_c[i])))

    band_mag = np.zeros((n_bands, S.shape[1]), dtype=np.float64)
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not np.any(mask):
            # nearest bin
            k = int(np.argmin(np.abs(freqs - band_freqs[b])))
            band_mag[b] = S[k]
        else:
            band_mag[b] = S[mask].mean(axis=0)

    # True time-domain RMS per frame (correct absolute noise level)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)[0]
    # Spectral flatness weights: tonal residual (model error) should not become noise
    flat = librosa.feature.spectral_flatness(S=S**2)[0]
    if len(rms) != S.shape[1] or len(flat) != S.shape[1]:
        m = min(len(rms), len(flat), S.shape[1])
        rms = rms[:m]
        flat = flat[:m]
        band_mag = band_mag[:, :m]
        frame_times = frame_times[:m]
    # Emphasize broadband residual; keep a small floor so some texture remains
    noise_amp = rms.astype(np.float64) * np.clip(flat.astype(np.float64), 0.05, 1.0)

    qt = _attack_biased_times(
        duration, n_time_points, attack_fraction, attack_point_fraction
    )
    amp_cp = _interp_series(frame_times, noise_amp, qt)
    mag_cp = np.zeros((n_bands, len(qt)), dtype=np.float64)
    for b in range(n_bands):
        mag_cp[b] = _interp_series(frame_times, band_mag[b], qt)

    # Normalize band magnitudes to relative shape (synth applies amp separately)
    for t in range(mag_cp.shape[1]):
        s = mag_cp[:, t].sum()
        if s > 1e-12:
            mag_cp[:, t] /= s

    return {
        "times": qt.tolist(),
        "band_freqs_hz": band_freqs.tolist(),
        "band_magnitudes": mag_cp.tolist(),
        "amplitude": amp_cp.tolist(),
    }


def analyze_transient(
    audio: NDArray,
    sample_rate: int,
    cfg: AnalysisConfig,
    *,
    start_s: float | None = None,
) -> dict[str, Any]:
    start = float(start_s) if start_s is not None else _detect_transient_start(audio, sample_rate)
    dur = float(cfg.transient_duration_s)
    i0 = int(round(start * sample_rate))
    i1 = min(len(audio), i0 + int(round(dur * sample_rate)))
    if i1 <= i0:
        i0 = 0
        i1 = min(len(audio), int(round(dur * sample_rate)))
        start = 0.0
    segment = audio[i0:i1]
    seg_dur = len(segment) / float(sample_rate)

    # Short amp envelope inside transient
    hop = max(32, cfg.stft_hop // 8)
    n_fft = min(512, max(128, len(segment)))
    if len(segment) < 4:
        env_times = [0.0, max(seg_dur, 1e-4)]
        env_vals = [0.0, 0.0]
        spectral = {
            "times": env_times,
            "band_freqs_hz": _band_freqs(cfg.spectral_bands, sample_rate).tolist(),
            "band_magnitudes": np.zeros((cfg.spectral_bands, 2)).tolist(),
        }
    else:
        t_env, rms = _rms_envelope(segment, sample_rate, frame_length=n_fft, hop=hop)
        if t_env[-1] < seg_dur:
            t_env = np.concatenate([t_env, [seg_dur]])
            rms = np.concatenate([rms, [rms[-1]]])
        n_env = min(16, cfg.envelope_points)
        qt = np.linspace(0.0, seg_dur, n_env)
        env_times = qt.tolist()
        env_vals = _interp_series(t_env, rms, qt).tolist()
        # Peak-normalize envelope for shape; absolute level via max
        peak = max(float(np.max(np.abs(segment))), 1e-12)
        env_vals = [v / max(max(env_vals), 1e-12) * peak for v in env_vals]

        banded = _analyze_banded_spectrum(
            segment,
            sample_rate,
            n_fft=n_fft,
            hop=hop,
            n_bands=cfg.spectral_bands,
            n_time_points=min(16, cfg.noise_control_points),
            duration=seg_dur,
            attack_fraction=0.5,
            attack_point_fraction=0.5,
        )
        spectral = {
            "times": banded["times"],
            "band_freqs_hz": banded["band_freqs_hz"],
            "band_magnitudes": banded["band_magnitudes"],
        }

    return {
        "start_s": start,
        "duration_s": seg_dur,
        "amplitude_envelope": {"times": env_times, "values": env_vals},
        "spectral_envelope": spectral,
    }


def analyze_audio(
    audio: NDArray,
    sample_rate: int,
    cfg: AnalysisConfig | None = None,
    *,
    original_path: str | None = None,
) -> dict[str, Any]:
    """Full analysis → parameter dictionary (no PCM)."""
    cfg = cfg or AnalysisConfig()
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    duration = len(audio) / float(sample_rate)

    # Longer sounds need denser trajectories; keep CLI/config as a floor.
    min_pts = max(cfg.partial_control_points, int(np.ceil(duration * 24)))
    if min_pts > cfg.partial_control_points:
        cfg.partial_control_points = min_pts
        cfg.envelope_points = max(cfg.envelope_points, min_pts)
        cfg.noise_control_points = max(cfg.noise_control_points, min_pts)
        cfg.f0_control_points = max(cfg.f0_control_points, min_pts)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms_amp = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    # Normalize for analysis stability; store peak for later restore
    if peak > 1e-12:
        audio_n = audio / peak
    else:
        audio_n = audio.copy()

    # Global amplitude envelope (on normalized signal)
    t_env, rms = _rms_envelope(
        audio_n, sample_rate, frame_length=cfg.stft_n_fft, hop=cfg.stft_hop
    )
    env_t, env_v = _downsample_traj(
        t_env,
        rms,
        cfg.envelope_points,
        duration,
        cfg.attack_fraction,
        cfg.attack_point_fraction,
    )
    # Peak-normalize envelope shape to ~1
    env_peak = max(max(env_v), 1e-12)
    env_v = [v / env_peak for v in env_v]

    # Fundamental
    f0_times, f0_hz, _voiced = _estimate_f0(audio_n, sample_rate, hop=cfg.stft_hop)
    f0_t, f0_v = _downsample_traj(
        f0_times,
        f0_hz,
        cfg.f0_control_points,
        duration,
        cfg.attack_fraction,
        cfg.attack_point_fraction,
    )

    # Partials
    partials = _extract_partials(audio_n, sample_rate, f0_times, f0_hz, cfg, duration)

    # Harmonic-only residual for noise model
    harm = synthesize_harmonics_only(
        {
            "meta": {
                "sample_rate": sample_rate,
                "duration": duration,
                "peak_amplitude": 1.0,
                "n_samples": len(audio_n),
            },
            "partials": partials,
            "analysis_config": cfg.to_dict(),
            "amplitude_envelope": {"times": env_t, "values": [1.0] * len(env_t)},
        },
        apply_global_envelope=False,
    )
    # Match length
    n = len(audio_n)
    if len(harm) < n:
        harm = np.pad(harm, (0, n - len(harm)))
    else:
        harm = harm[:n]

    # Calibrate global harmonic gain (least-squares) to absorb windowing bias
    denom = float(np.dot(harm, harm))
    if denom > 1e-12:
        harm_gain = float(np.dot(audio_n, harm) / denom)
        harm_gain = float(np.clip(harm_gain, 0.05, 4.0))
    else:
        harm_gain = 1.0
    for p in partials:
        p["amplitude"] = [float(a * harm_gain) for a in p["amplitude"]]
    harm = harm * harm_gain

    residual = audio_n - harm

    # Detect onset on the original (normalized) signal; model residual energy there
    onset = _detect_transient_start(audio_n, sample_rate)
    transient = analyze_transient(residual, sample_rate, cfg, start_s=onset)

    noise_model = _analyze_banded_spectrum(
        residual,
        sample_rate,
        n_fft=cfg.stft_n_fft,
        hop=cfg.stft_hop,
        n_bands=cfg.spectral_bands,
        n_time_points=cfg.noise_control_points,
        duration=duration,
        attack_fraction=cfg.attack_fraction,
        attack_point_fraction=cfg.attack_point_fraction,
    )

    # Shared time grid once (partials reference it) to keep JSON compact
    partial_times = partials[0]["times"] if partials else []
    compact_partials = []
    for p in partials:
        compact_partials.append(
            {
                "id": p["id"],
                "frequency_hz": p["frequency_hz"],
                "amplitude": p["amplitude"],
                "phase": p["phase"],
            }
        )

    params: dict[str, Any] = {
        "version": 1,
        "meta": {
            "sample_rate": int(sample_rate),
            "duration": float(duration),
            "channels": 1,
            "original_path": original_path,
            "peak_amplitude": peak,
            "rms_amplitude": rms_amp,
            "seed": int(cfg.seed),
            "n_samples": int(n),
        },
        "fundamental": {
            "times": f0_t,
            "frequency_hz": f0_v,
            "frequency_hz_mean": float(np.mean(f0_v)),
        },
        "harmonic_gain": harm_gain,
        "amplitude_envelope": {
            "times": env_t,
            "values": env_v,
        },
        "partial_times": partial_times,
        "partials": compact_partials,
        "transient": transient,
        "noise": {
            "times": noise_model["times"],
            "amplitude": noise_model["amplitude"],
            "band_freqs_hz": noise_model["band_freqs_hz"],
            "band_magnitudes": noise_model["band_magnitudes"],
        },
        "analysis_config": cfg.to_dict(),
    }
    return params


def analyze_file(
    path: str | Path,
    cfg: AnalysisConfig | None = None,
    *,
    downmix: bool = True,
) -> dict[str, Any]:
    from io_utils import load_audio

    audio, sr = load_audio(path, downmix=downmix)
    return analyze_audio(audio, sr, cfg, original_path=str(Path(path).resolve()))
