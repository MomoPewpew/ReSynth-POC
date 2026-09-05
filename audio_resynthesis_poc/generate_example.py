#!/usr/bin/env python3
"""Generate a synthetic mono example WAV for the POC smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    sr = 44100
    duration = 1.5
    f0 = 220.0
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Decaying harmonic stack
    amps = [1.0, 0.55, 0.35, 0.22, 0.14, 0.09, 0.06, 0.04]
    env = np.exp(-2.2 * t)
    # Soft attack
    attack = np.minimum(t / 0.015, 1.0)
    tone = np.zeros(n)
    for k, a in enumerate(amps, start=1):
        tone += a * np.sin(2 * np.pi * k * f0 * t)
    tone *= env * attack

    # Broadband transient at onset
    rng = np.random.default_rng(0)
    burst_n = int(0.04 * sr)
    burst = rng.standard_normal(burst_n) * np.linspace(1.0, 0.0, burst_n) ** 2 * 0.35
    noise = np.zeros(n)
    noise[:burst_n] = burst
    # Light sustained breathiness
    noise += 0.02 * rng.standard_normal(n) * env

    audio = tone + noise
    audio /= np.max(np.abs(audio)) * 1.05

    out = Path(__file__).resolve().parent / "input" / "example.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio.astype(np.float64), sr, subtype="PCM_24")
    print(f"Wrote {out} ({duration}s, {sr} Hz, f0={f0} Hz)")


if __name__ == "__main__":
    main()
