#!/usr/bin/env python3
"""Render a scale from one parameterized note.

Pitch methods
-------------
synthesize_then_shift (default)
    Synthesize once at the recorded pitch, then phase-vocoder shift each degree.
    Keeps the parameterized *timbre* and avoids the additive "bell" artifact of
    scaling partial frequencies.

parametric
    Transpose f0/partial frequencies in parameter space, then synthesize.
    Scientifically interesting, but piano-like tones go chime/bell when raised.

original_shift
    Phase-vocoder shift the original recording (baseline / upper bound).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import load_audio, load_parameters, save_audio
from pitch import (
    MAJOR_SCALE_DEGREE_NAMES,
    MAJOR_SCALE_SEMITONES,
    pitch_shift_audio,
    transpose_parameters,
)
from synthesis import synthesize


def _trim_fade_norm(
    y: np.ndarray,
    trim_n: int,
    fade_n: int,
) -> np.ndarray:
    if len(y) >= trim_n:
        y = y[:trim_n].copy()
    else:
        y = np.pad(y, (0, trim_n - len(y)))
    if fade_n < len(y):
        y[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1e-9:
        y *= 0.9 / peak
    return y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a scale from one parameter file / source note."
    )
    parser.add_argument("parameters", type=Path, help="Source parameters.json")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "scale",
    )
    parser.add_argument("--scale", choices=["major"], default="major")
    parser.add_argument("--trim", type=float, default=1.5)
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--note-fade", type=float, default=0.05)
    parser.add_argument(
        "--method",
        choices=["synthesize_then_shift", "parametric", "original_shift"],
        default="synthesize_then_shift",
        help="Pitch method (default: synthesize_then_shift)",
    )
    parser.add_argument(
        "--formant-preserve",
        type=float,
        default=0.55,
        help="Only for --method parametric",
    )
    parser.add_argument(
        "--no-register-compensate",
        action="store_true",
        help="Only for --method parametric",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also write a 3-way comparison WAV (all methods, same scale)",
    )
    args = parser.parse_args()

    if not args.parameters.is_file():
        print(f"Not found: {args.parameters}", file=sys.stderr)
        sys.exit(1)

    params = load_parameters(args.parameters)
    sr = int(params["meta"]["sample_rate"])
    intervals = MAJOR_SCALE_SEMITONES
    names = MAJOR_SCALE_DEGREE_NAMES

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = args.output or (args.output_dir / f"major_scale_{args.method}.wav")

    trim_n = int(round(args.trim * sr))
    gap = np.zeros(int(round(args.gap * sr)), dtype=np.float64)
    fade_n = max(1, int(round(args.note_fade * sr)))

    # Base signals
    base_resynth = synthesize(params)
    original = None
    orig_path = params.get("meta", {}).get("original_path")
    if orig_path and Path(orig_path).is_file():
        original, orig_sr = load_audio(orig_path)
        if orig_sr != sr:
            print(f"Warning: original SR {orig_sr} != params SR {sr}")
    # Fallback search next to common test location
    if original is None:
        cand = (
            Path(__file__).resolve().parent / "Test Samples" / "060_C4v10.wav"
        )
        if cand.is_file():
            original, _ = load_audio(cand)

    def render_method(method: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        print(f"\n=== method={method} ===")
        for i, (semi, name) in enumerate(zip(intervals, names)):
            if method == "parametric":
                p = transpose_parameters(
                    params,
                    float(semi),
                    formant_preserve=args.formant_preserve,
                    register_compensate=not args.no_register_compensate,
                )
                y = synthesize(p)
                f0 = p["fundamental"]["frequency_hz_mean"]
            elif method == "synthesize_then_shift":
                y = pitch_shift_audio(base_resynth, sr, float(semi))
                f0 = params["fundamental"]["frequency_hz_mean"] * (
                    2.0 ** (semi / 12.0)
                )
            elif method == "original_shift":
                if original is None:
                    raise RuntimeError(
                        "original_shift requires meta.original_path or Test Samples/060_C4v10.wav"
                    )
                y = pitch_shift_audio(original, sr, float(semi))
                f0 = params["fundamental"]["frequency_hz_mean"] * (
                    2.0 ** (semi / 12.0)
                )
            else:
                raise ValueError(method)

            y = _trim_fade_norm(y, trim_n, fade_n)
            chunks.append(y)
            if i < len(intervals) - 1:
                chunks.append(gap)
            print(f"  degree {name}: {semi:+.0f} st  f0≈{f0:.1f} Hz")
        return np.concatenate(chunks)

    audio = render_method(args.method)
    save_audio(out_wav, audio, sr)
    print(f"\nWrote: {out_wav}")
    print(f"  total duration: {len(audio)/sr:.2f} s")

    if args.compare:
        # Long gap between methods so the switch is obvious
        method_gap = np.zeros(int(round(1.0 * sr)), dtype=np.float64)
        parts = []
        for m in ("original_shift", "synthesize_then_shift", "parametric"):
            try:
                parts.append(render_method(m))
                parts.append(method_gap)
            except RuntimeError as exc:
                print(f"Skipping {m}: {exc}")
        if parts:
            # drop trailing gap
            compare = np.concatenate(parts[:-1])
            cpath = args.output_dir / "major_scale_COMPARE_3methods.wav"
            save_audio(cpath, compare, sr)
            print(f"\nWrote comparison (orig PV | resynth PV | parametric): {cpath}")
            print("  1s silence between methods")


if __name__ == "__main__":
    main()
