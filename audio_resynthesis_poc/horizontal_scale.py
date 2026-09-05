#!/usr/bin/env python3
"""Horizontal interpolation across two differently pitched samples.

methods
-------
pv_crossfade (default)
    Resynthesize low & high, phase-vocoder each to the target degree, then
    crossfade by t=semitones/12. Same idea as a multisample keyboard:
    pitch both neighbors toward the note and blend. Avoids inventing a
    fresh additive partial stack (the source of the bell/chime artifact).

parametric
    Morph structured parameters at t, then synthesize. Useful as a stress
    test, but intermediate degrees often sound belley for piano-like tones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interpolate import interpolate_parameters
from io_utils import load_parameters, save_audio, save_parameters
from pitch import MAJOR_SCALE_DEGREE_NAMES, MAJOR_SCALE_SEMITONES, pitch_shift_audio
from synthesis import synthesize


def _snap_to_target_f0(params: dict, target_f0: float) -> dict:
    import copy

    current = float(params["fundamental"]["frequency_hz_mean"])
    if current < 1e-6:
        return params
    ratio = target_f0 / current
    params = copy.deepcopy(params)
    f0 = np.asarray(params["fundamental"]["frequency_hz"], dtype=np.float64) * ratio
    params["fundamental"]["frequency_hz"] = f0.tolist()
    params["fundamental"]["frequency_hz_mean"] = float(np.mean(f0))
    for partial in params.get("partials", []):
        freq = np.asarray(partial["frequency_hz"], dtype=np.float64) * ratio
        partial["frequency_hz"] = freq.tolist()
    return params


def _trim_fade_norm(y: np.ndarray, trim_n: int, fade_n: int) -> np.ndarray:
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


def _semitones_between(f_from: float, f_to: float) -> float:
    return float(12.0 * np.log2(max(f_to, 1e-9) / max(f_from, 1e-9)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Major scale via horizontal interpolation between two pitched samples."
    )
    parser.add_argument("params_low", type=Path)
    parser.add_argument("params_high", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "horizontal_scale",
    )
    parser.add_argument(
        "--method",
        choices=["pv_crossfade", "parametric"],
        default="pv_crossfade",
    )
    parser.add_argument("--trim", type=float, default=1.5)
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--note-fade", type=float, default=0.05)
    parser.add_argument("--save-params", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also write parametric vs pv_crossfade comparison",
    )
    args = parser.parse_args()

    low = load_parameters(args.params_low)
    high = load_parameters(args.params_high)
    sr = int(low["meta"]["sample_rate"])
    if sr != int(high["meta"]["sample_rate"]):
        sys.exit("Sample rates must match")

    f0_low = float(low["fundamental"]["frequency_hz_mean"])
    f0_high = float(high["fundamental"]["frequency_hz_mean"])
    ratio = f0_high / max(f0_low, 1e-9)
    cents = 1200.0 * np.log2(ratio)

    print("Horizontal interpolation scale")
    print(f"  low  f0 ≈ {f0_low:.2f} Hz")
    print(f"  high f0 ≈ {f0_high:.2f} Hz")
    print(f"  span     {cents:.1f} cents (ratio {ratio:.3f})")
    print(f"  method   {args.method}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = args.output or (
        args.output_dir / f"horizontal_major_scale_{args.method}.wav"
    )

    trim_n = int(round(args.trim * sr))
    gap = np.zeros(int(round(args.gap * sr)), dtype=np.float64)
    fade_n = max(1, int(round(args.note_fade * sr)))

    # Pre-render endpoint resyntheses once (for pv_crossfade)
    print("\nSynthesizing endpoints...")
    base_low = synthesize(low)
    base_high = synthesize(high)

    def render(method: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        print(f"\n=== {method} ===")
        for i, (semi, name) in enumerate(
            zip(MAJOR_SCALE_SEMITONES, MAJOR_SCALE_DEGREE_NAMES)
        ):
            t = float(semi) / 12.0
            target_f0 = f0_low * (2.0 ** (semi / 12.0))
            if abs(ratio - 2.0) > 0.05:
                target_f0 = f0_low * (ratio ** (semi / 12.0))

            if method == "parametric":
                params = interpolate_parameters(
                    low,
                    high,
                    t,
                    phase_mode="residual_blend",
                    blend_duration=False,
                )
                params = _snap_to_target_f0(params, target_f0)
                y = synthesize(params)
                if args.save_params:
                    save_parameters(
                        args.output_dir / f"parameters_deg{name}_t{int(t*100):02d}.json",
                        params,
                    )
            else:
                # Pitch both neighbors to the target, then crossfade by keyboard distance
                st_low = _semitones_between(f0_low, target_f0)
                st_high = _semitones_between(f0_high, target_f0)
                y_low = pitch_shift_audio(base_low, sr, st_low)
                y_high = pitch_shift_audio(base_high, sr, st_high)
                n = min(len(y_low), len(y_high))
                y = (1.0 - t) * y_low[:n] + t * y_high[:n]

            y = _trim_fade_norm(y, trim_n, fade_n)
            chunks.append(y)
            if i < len(MAJOR_SCALE_SEMITONES) - 1:
                chunks.append(gap)
            print(
                f"  degree {name}: t={t:.3f}  target≈{target_f0:.1f} Hz  "
                f"dur={len(y)/sr:.2f}s"
            )
        return np.concatenate(chunks)

    audio = render(args.method)
    save_audio(out_wav, audio, sr)
    print(f"\nWrote: {out_wav}")
    print(f"  total duration: {len(audio)/sr:.2f} s")

    if args.compare:
        a = render("pv_crossfade")
        b = render("parametric")
        method_gap = np.zeros(int(round(1.0 * sr)), dtype=np.float64)
        compare = np.concatenate([a, method_gap, b])
        cpath = args.output_dir / "horizontal_COMPARE_pv_vs_parametric.wav"
        save_audio(cpath, compare, sr)
        print(f"\nWrote comparison (pv_crossfade | parametric): {cpath}")


if __name__ == "__main__":
    main()
