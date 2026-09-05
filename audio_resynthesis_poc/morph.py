#!/usr/bin/env python3
"""Interpolate two parameter files and write a stepped morph listening WAV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interpolate import interpolate_parameters
from io_utils import load_parameters, save_audio, save_parameters
from synthesis import synthesize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Morph between two parameter JSONs and write a stepped A→B audio file."
    )
    parser.add_argument("params_a", type=Path, help="Start parameters (t=0), e.g. piano")
    parser.add_argument("params_b", type=Path, help="End parameters (t=1), e.g. quack")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output morph WAV (default: output/morph_ab.wav)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "morph",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=11,
        help="Number of morph steps including endpoints (default 11 → 10%% increments)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.35,
        help="Silence between steps in seconds",
    )
    parser.add_argument(
        "--save-params",
        action="store_true",
        help="Also write parameters_tXX.json for each step",
    )
    args = parser.parse_args()

    if args.steps < 2:
        print("--steps must be >= 2", file=sys.stderr)
        sys.exit(1)

    a = load_parameters(args.params_a)
    b = load_parameters(args.params_b)
    sr = int(a["meta"]["sample_rate"])
    if sr != int(b["meta"]["sample_rate"]):
        print("Sample rates must match", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = args.output or (args.output_dir / "morph_ab.wav")

    gap = np.zeros(int(round(args.gap * sr)), dtype=np.float64)
    chunks: list[np.ndarray] = []
    ts = np.linspace(0.0, 1.0, args.steps)

    print(f"Rendering {args.steps} morph steps (0% → 100%)...\n")
    for i, t in enumerate(ts):
        pct = int(round(100 * t))
        params = interpolate_parameters(a, b, float(t))
        y = synthesize(params)
        # Peak-normalize each step lightly for comparable listening level
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 1e-9:
            y = y * (0.9 / peak)
        chunks.append(y)
        if i < args.steps - 1:
            chunks.append(gap)
        print(
            f"  step {i:2d}: {pct:3d}%  "
            f"dur={params['meta']['duration']:.3f}s  "
            f"f0≈{params['fundamental']['frequency_hz_mean']:.1f} Hz  "
            f"samples={len(y):,}"
        )
        if args.save_params:
            save_parameters(args.output_dir / f"parameters_t{pct:03d}.json", params)

    audio = np.concatenate(chunks) if chunks else np.zeros(0)
    save_audio(out_wav, audio, sr)
    print(f"\nWrote morph listening file: {out_wav}")
    print(f"  total duration: {len(audio) / sr:.2f} s")
    print("  order: 100% A → … → 100% B (resynthesized at each step)")


if __name__ == "__main__":
    main()
