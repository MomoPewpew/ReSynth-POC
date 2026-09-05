#!/usr/bin/env python3
"""Resynthesize audio from a parameter JSON and optionally A/B compare."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import (
    load_audio,
    load_parameters,
    parameter_storage_stats,
    pcm_stats,
    play_audio,
    save_audio,
)
from metrics import compute_all_metrics, format_results_block
from synthesis import synthesize


def build_ab_audio(
    resynthesized: np.ndarray,
    original: np.ndarray,
    sample_rate: int,
    *,
    gap_s: float = 0.0,
) -> np.ndarray:
    """Concatenate resynthesized then original — same order as --play hearing."""
    y = np.asarray(resynthesized, dtype=np.float64).reshape(-1)
    x = np.asarray(original, dtype=np.float64).reshape(-1)
    if gap_s > 0:
        gap = np.zeros(int(round(gap_s * sample_rate)), dtype=np.float64)
        return np.concatenate([y, gap, x])
    return np.concatenate([y, x])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resynthesize audio from parametric JSON (no PCM in file)."
    )
    parser.add_argument("parameters", type=Path, help="parameters.json from analyze.py")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output WAV (default: <output-dir>/resynthesized.wav)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help="Original WAV for metrics / A/B (defaults to meta.original_path)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play resynthesized then original (if available)",
    )
    parser.add_argument(
        "--ab",
        action="store_true",
        help=(
            "Also write an A/B WAV: resynthesized, then original "
            "(same order as --play). Default path: <output-dir>/ab_comparison.wav"
        ),
    )
    parser.add_argument(
        "--ab-output",
        type=Path,
        default=None,
        help="Path for the A/B comparison WAV (implies --ab)",
    )
    parser.add_argument(
        "--ab-gap",
        type=float,
        default=0.0,
        help="Silence between resynthesized and original in the A/B file (seconds)",
    )
    parser.add_argument("--plots", action="store_true", help="Write diagnostic plots")
    args = parser.parse_args()

    if not args.parameters.is_file():
        print(f"Parameters not found: {args.parameters}", file=sys.stderr)
        sys.exit(1)

    want_ab = args.ab or args.ab_output is not None

    params = load_parameters(args.parameters)
    sr = int(params["meta"]["sample_rate"])
    y = synthesize(params)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = args.output or (args.output_dir / "resynthesized.wav")
    save_audio(out_wav, y, sr)
    print(f"Wrote resynthesized audio: {out_wav}")

    original_path = args.original
    if original_path is None:
        cand = params.get("meta", {}).get("original_path")
        if cand and Path(cand).is_file():
            original_path = Path(cand)

    original = None
    if original_path is not None and Path(original_path).is_file():
        original, orig_sr = load_audio(original_path)
        if orig_sr != sr:
            print(
                f"Warning: original SR {orig_sr} != params SR {sr}; "
                "metrics use params SR length alignment only."
            )
        cfg = params.get("analysis_config") or {}
        metrics = compute_all_metrics(
            original,
            y,
            sr,
            n_fft=int(cfg.get("stft_n_fft", 2048)),
            hop=int(cfg.get("stft_hop", 512)),
        )
        pstats = parameter_storage_stats(params, args.parameters)
        pcm = pcm_stats(original, sr)
        print()
        print(
            format_results_block(
                metrics,
                n_scalars=pstats["n_scalars"],
                param_kb=pstats["est_kb"],
                pcm_kb=pcm["raw_kb"],
            )
        )
    else:
        print("No original audio available for metrics/A-B (pass --original).")

    if want_ab:
        if original is None:
            print(
                "Cannot write A/B file: original audio not found (pass --original).",
                file=sys.stderr,
            )
            sys.exit(1)
        ab_path = args.ab_output or (args.output_dir / "ab_comparison.wav")
        ab = build_ab_audio(y, original, sr, gap_s=args.ab_gap)
        save_audio(ab_path, ab, sr)
        print(f"Wrote A/B comparison audio: {ab_path}")
        print("  (order: resynthesized → original, same as --play)")

    if args.plots:
        from plots import save_diagnostic_plots

        plots_dir = args.output_dir / "plots"
        cfg = params.get("analysis_config") or {}
        written = save_diagnostic_plots(
            plots_dir,
            original=original,
            resynthesized=y,
            sample_rate=sr,
            params=params,
            n_fft=int(cfg.get("stft_n_fft", 2048)),
            hop=int(cfg.get("stft_hop", 512)),
        )
        print(f"Wrote {len(written)} plots to {plots_dir}")

    if args.play:
        print("\nPlaying resynthesized...")
        play_audio(y, sr)
        if original is not None:
            print("Playing original...")
            play_audio(original, sr)


if __name__ == "__main__":
    main()
