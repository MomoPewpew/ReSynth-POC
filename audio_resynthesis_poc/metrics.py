"""Objective reconstruction metrics.

Definitions (documented for scientific clarity — none prove perceptual identity):

RMS error
    sqrt(mean((x - y)^2))

NMSE (normalized mean squared error)
    mean((x - y)^2) / mean(x^2)   (epsilon floor on denominator)

SNR (dB)
    10 * log10( mean(x^2) / mean((x - y)^2) )

Spectral difference
    mean( |log(|S_x| + eps) - log(|S_y| + eps)| ) over STFT magnitude bins

Spectral convergence
    1 - || |S_x| - |S_y| ||_F / || |S_x| ||_F

Log-mel distance (optional perceptual-ish proxy)
    RMSE between log-mel spectrograms of x and y. Not a perceptual identity proof.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _align(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    n = min(len(x), len(y))
    return np.asarray(x[:n], dtype=np.float64), np.asarray(y[:n], dtype=np.float64)


def rms_error(x: NDArray, y: NDArray) -> float:
    x, y = _align(x, y)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def nmse(x: NDArray, y: NDArray, eps: float = 1e-12) -> float:
    x, y = _align(x, y)
    return float(np.mean((x - y) ** 2) / max(np.mean(x**2), eps))


def snr_db(x: NDArray, y: NDArray, eps: float = 1e-12) -> float:
    x, y = _align(x, y)
    num = np.mean(x**2)
    den = np.mean((x - y) ** 2)
    return float(10.0 * np.log10(max(num, eps) / max(den, eps)))


def stft_mag(
    audio: NDArray,
    n_fft: int = 2048,
    hop: int = 512,
) -> NDArray:
    import librosa

    S = librosa.stft(np.asarray(audio, dtype=np.float64), n_fft=n_fft, hop_length=hop)
    return np.abs(S)


def spectral_difference(
    x: NDArray,
    y: NDArray,
    n_fft: int = 2048,
    hop: int = 512,
    eps: float = 1e-8,
) -> float:
    """Mean absolute log-magnitude STFT difference."""
    x, y = _align(x, y)
    Sx = stft_mag(x, n_fft=n_fft, hop=hop)
    Sy = stft_mag(y, n_fft=n_fft, hop=hop)
    t = min(Sx.shape[1], Sy.shape[1])
    Sx, Sy = Sx[:, :t], Sy[:, :t]
    return float(np.mean(np.abs(np.log(Sx + eps) - np.log(Sy + eps))))


def spectral_convergence(
    x: NDArray,
    y: NDArray,
    n_fft: int = 2048,
    hop: int = 512,
    eps: float = 1e-12,
) -> float:
    """1 - ||Sx - Sy||_F / ||Sx||_F on magnitude STFTs."""
    x, y = _align(x, y)
    Sx = stft_mag(x, n_fft=n_fft, hop=hop)
    Sy = stft_mag(y, n_fft=n_fft, hop=hop)
    t = min(Sx.shape[1], Sy.shape[1])
    Sx, Sy = Sx[:, :t], Sy[:, :t]
    num = np.linalg.norm(Sx - Sy)
    den = np.linalg.norm(Sx)
    return float(1.0 - num / max(den, eps))


def log_mel_distance(
    x: NDArray,
    y: NDArray,
    sample_rate: int,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 64,
    eps: float = 1e-8,
) -> float:
    """RMSE of log-mel spectrograms (proxy only; not perceptual identity)."""
    import librosa

    x, y = _align(x, y)
    Mx = librosa.feature.melspectrogram(
        y=x, sr=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    My = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    t = min(Mx.shape[1], My.shape[1])
    Lx = np.log(Mx[:, :t] + eps)
    Ly = np.log(My[:, :t] + eps)
    return float(np.sqrt(np.mean((Lx - Ly) ** 2)))


def compute_all_metrics(
    original: NDArray,
    resynthesized: NDArray,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict[str, float]:
    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(resynthesized, dtype=np.float64)
    return {
        "duration_original_s": len(x) / float(sample_rate),
        "duration_resynthesized_s": len(y) / float(sample_rate),
        "duration_difference_s": abs(len(x) - len(y)) / float(sample_rate),
        "rms_error": rms_error(x, y),
        "nmse": nmse(x, y),
        "snr_db": snr_db(x, y),
        "spectral_difference": spectral_difference(x, y, n_fft=n_fft, hop=hop),
        "spectral_convergence": spectral_convergence(x, y, n_fft=n_fft, hop=hop),
        "log_mel_distance": log_mel_distance(
            x, y, sample_rate, n_fft=n_fft, hop=hop
        ),
    }


def format_results_block(
    metrics: dict[str, float],
    n_scalars: int,
    param_kb: float,
    pcm_kb: float,
) -> str:
    ratio = (pcm_kb * 1024.0) / max(param_kb * 1024.0, 1.0)
    return "\n".join(
        [
            "=== RESYNTHESIS RESULTS ===",
            "",
            "Duration:",
            f"    Original:      {metrics['duration_original_s']:.3f} s",
            f"    Resynthesized: {metrics['duration_resynthesized_s']:.3f} s",
            f"    Difference:    {metrics['duration_difference_s']:.6f} s",
            "",
            "Parameters:",
            f"    {n_scalars:,} scalar values",
            "",
            "Approx. representation:",
            f"    {param_kb:.1f} KB",
            "",
            "Original PCM:",
            f"    {pcm_kb:.1f} KB",
            "",
            "Compression:",
            f"    {ratio:.1f}x",
            "",
            "SNR:",
            f"    {metrics['snr_db']:.2f} dB",
            "",
            "RMS error:",
            f"    {metrics['rms_error']:.6f}",
            "",
            "NMSE:",
            f"    {metrics['nmse']:.6f}",
            "",
            "Spectral difference (mean |Δlog|S||):",
            f"    {metrics['spectral_difference']:.4f}",
            "",
            "Spectral convergence:",
            f"    {metrics['spectral_convergence']:.4f}",
            "",
            "Log-mel distance (proxy, not perceptual proof):",
            f"    {metrics['log_mel_distance']:.4f}",
        ]
    )
