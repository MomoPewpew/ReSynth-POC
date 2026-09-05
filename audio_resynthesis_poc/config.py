"""Configurable analysis/synthesis budgets for the parametric resynthesis POC.

These knobs deliberately constrain the representation so we can measure the
tradeoff between parameter count and reconstruction quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


# --- Default budgets (easy to change) ---

MAX_PARTIALS = 32
ENVELOPE_POINTS = 64
SPECTRAL_BANDS = 32
TRANSIENT_DURATION_S = 0.05
NOISE_CONTROL_POINTS = 64
PARTIAL_CONTROL_POINTS = 64
F0_CONTROL_POINTS = 64

STFT_N_FFT = 4096
STFT_HOP = 512

RANDOM_SEED = 42

# Fractional search window around k*f0 when locking onto inharmonic partials.
PARTIAL_SEARCH_FRACTION = 0.03

# Attack densification: put this fraction of envelope points in the first
# ATTACK_FRACTION of the duration.
ATTACK_FRACTION = 0.15
ATTACK_POINT_FRACTION = 0.4


@dataclass
class AnalysisConfig:
    """Snapshot of knobs used for one analysis run (stored in the JSON)."""

    max_partials: int = MAX_PARTIALS
    envelope_points: int = ENVELOPE_POINTS
    spectral_bands: int = SPECTRAL_BANDS
    transient_duration_s: float = TRANSIENT_DURATION_S
    noise_control_points: int = NOISE_CONTROL_POINTS
    partial_control_points: int = PARTIAL_CONTROL_POINTS
    f0_control_points: int = F0_CONTROL_POINTS
    stft_n_fft: int = STFT_N_FFT
    stft_hop: int = STFT_HOP
    seed: int = RANDOM_SEED
    attack_fraction: float = ATTACK_FRACTION
    attack_point_fraction: float = ATTACK_POINT_FRACTION
    partial_search_fraction: float = PARTIAL_SEARCH_FRACTION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
