"""Diagnostic plots for analysis / resynthesis inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def save_diagnostic_plots(
    out_dir: str | Path,
    *,
    original: NDArray | None,
    resynthesized: NDArray | None,
    sample_rate: int,
    params: dict[str, Any] | None = None,
    n_fft: int = 2048,
    hop: int = 512,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _spec(ax, y, title):
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sample_rate, hop_length=hop, x_axis="time", y_axis="hz", ax=ax
        )
        ax.set_title(title)
        return img

    # Waveforms
    if original is not None:
        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(len(original)) / sample_rate
        ax.plot(t, original, lw=0.6)
        ax.set_title("Original waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        p = out_dir / "01_original_waveform.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if resynthesized is not None:
        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(len(resynthesized)) / sample_rate
        ax.plot(t, resynthesized, lw=0.6, color="C1")
        ax.set_title("Resynthesized waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        p = out_dir / "02_resynthesized_waveform.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if original is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        _spec(ax, original, "Original spectrogram")
        fig.tight_layout()
        p = out_dir / "03_original_spectrogram.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if resynthesized is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        _spec(ax, resynthesized, "Resynthesized spectrogram")
        fig.tight_layout()
        p = out_dir / "04_resynthesized_spectrogram.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if original is not None and resynthesized is not None:
        n = min(len(original), len(resynthesized))
        resid = original[:n] - resynthesized[:n]
        fig, ax = plt.subplots(figsize=(10, 4))
        _spec(ax, resid, "Difference / residual spectrogram")
        fig.tight_layout()
        p = out_dir / "05_difference_spectrogram.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if params is not None:
        # Envelope + f0 + partial amplitudes
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        env = params.get("amplitude_envelope", {})
        axes[0].plot(env.get("times", []), env.get("values", []), marker="o", ms=2)
        axes[0].set_ylabel("Envelope")
        axes[0].set_title("Extracted parameter trajectories")

        fund = params.get("fundamental", {})
        axes[1].plot(
            fund.get("times", []), fund.get("frequency_hz", []), marker="o", ms=2, color="C2"
        )
        axes[1].set_ylabel("f0 (Hz)")

        for partial in (params.get("partials") or [])[:8]:
            times = partial.get("times") or params.get("partial_times") or []
            axes[2].plot(
                times,
                partial["amplitude"],
                lw=1,
                label=f"h{partial['id']}",
            )
        axes[2].set_ylabel("Partial amp")
        axes[2].set_xlabel("Time (s)")
        if params.get("partials"):
            axes[2].legend(ncol=4, fontsize=8, loc="upper right")
        fig.tight_layout()
        p = out_dir / "06_parameter_trajectories.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

        noise = params.get("noise") or {}
        bm = np.asarray(noise.get("band_magnitudes", []), dtype=np.float64)
        if bm.ndim == 2 and bm.size:
            fig, ax = plt.subplots(figsize=(10, 4))
            times = noise.get("times", list(range(bm.shape[1])))
            freqs = noise.get("band_freqs_hz", list(range(bm.shape[0])))
            im = ax.imshow(
                bm,
                aspect="auto",
                origin="lower",
                extent=[times[0], times[-1], 0, len(freqs)],
                interpolation="nearest",
            )
            ax.set_title("Noise band magnitudes (relative)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Band index")
            fig.colorbar(im, ax=ax, fraction=0.02)
            fig.tight_layout()
            p = out_dir / "07_noise_bands.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            written.append(p)

    return written
