#!/usr/bin/env python3
"""Analyze a mono WAV into a human-inspectable parameter JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import analyze_file
from config import AnalysisConfig
from io_utils import (
    format_size_report,
    load_audio,
    parameter_storage_stats,
    pcm_stats,
    save_parameters,
)
from synthesis import synthesize


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        max_partials=args.max_partials,
        envelope_points=args.envelope_points,
        spectral_bands=args.spectral_bands,
        transient_duration_s=args.transient_duration,
        noise_control_points=args.noise_control_points,
        partial_control_points=args.partial_control_points,
        seed=args.seed,
    )


def run_once(
    input_path: Path,
    output_json: Path,
    cfg: AnalysisConfig,
    *,
    plots: bool,
    plots_dir: Path,
    reject_stereo: bool,
    quiet: bool = False,
) -> dict:
    params = analyze_file(input_path, cfg, downmix=not reject_stereo)
    save_parameters(output_json, params)

    audio, sr = load_audio(input_path, downmix=not reject_stereo)
    pcm = pcm_stats(audio, sr)
    pstats = parameter_storage_stats(params, output_json)
    if not quiet:
        print(format_size_report(pcm, pstats))
        print(f"\nWrote parameters: {output_json}")

    if plots:
        from plots import save_diagnostic_plots

        # Quick resynth for comparison plots during analysis
        y = synthesize(params)
        written = save_diagnostic_plots(
            plots_dir,
            original=audio,
            resynthesized=y,
            sample_rate=sr,
            params=params,
            n_fft=cfg.stft_n_fft,
            hop=cfg.stft_hop,
        )
        if not quiet:
            print(f"Wrote {len(written)} plots to {plots_dir}")

    return {"params": params, "pcm": pcm, "pstats": pstats}


def run_ablation(input_path: Path, args: argparse.Namespace) -> None:
    from metrics import compute_all_metrics

    audio, sr = load_audio(input_path, downmix=not args.reject_stereo)
    partials_list = [8, 16, 32, 64]
    print("\n=== ABLATION: max_partials sweep ===\n")
    print(f"{'partials':>10} {'scalars':>10} {'SNR_dB':>10} {'spec_conv':>10} {'est_KB':>10}")
    out_root = Path(args.output_dir)
    for mp in partials_list:
        cfg = build_config(args)
        cfg.max_partials = mp
        out_json = out_root / f"parameters_partials_{mp}.json"
        result = run_once(
            input_path,
            out_json,
            cfg,
            plots=False,
            plots_dir=out_root / "plots",
            reject_stereo=args.reject_stereo,
            quiet=True,
        )
        y = synthesize(result["params"])
        m = compute_all_metrics(audio, y, sr, n_fft=cfg.stft_n_fft, hop=cfg.stft_hop)
        print(
            f"{mp:10d} {result['pstats']['n_scalars']:10d} "
            f"{m['snr_db']:10.2f} {m['spectral_convergence']:10.4f} "
            f"{result['pstats']['est_kb']:10.1f}"
        )
    # Also write default parameters.json as the last default budget run
    cfg = build_config(args)
    run_once(
        input_path,
        out_root / "parameters.json",
        cfg,
        plots=args.plots,
        plots_dir=out_root / "plots",
        reject_stereo=args.reject_stereo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze audio into compact, interpretable synthesis parameters."
    )
    parser.add_argument("input", type=Path, help="Input WAV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output parameters JSON (default: <output-dir>/parameters.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory",
    )
    parser.add_argument("--max-partials", type=int, default=AnalysisConfig.max_partials)
    parser.add_argument("--envelope-points", type=int, default=AnalysisConfig.envelope_points)
    parser.add_argument("--spectral-bands", type=int, default=AnalysisConfig.spectral_bands)
    parser.add_argument(
        "--transient-duration", type=float, default=AnalysisConfig.transient_duration_s
    )
    parser.add_argument(
        "--noise-control-points", type=int, default=AnalysisConfig.noise_control_points
    )
    parser.add_argument(
        "--partial-control-points", type=int, default=AnalysisConfig.partial_control_points
    )
    parser.add_argument("--seed", type=int, default=AnalysisConfig.seed)
    parser.add_argument("--plots", action="store_true", help="Write diagnostic plots")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Sweep max-partials 8/16/32/64 and print quality table",
    )
    parser.add_argument(
        "--reject-stereo",
        action="store_true",
        help="Reject stereo instead of downmixing to mono",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    if args.ablation:
        run_ablation(args.input, args)
        return

    out_json = args.output or (args.output_dir / "parameters.json")
    cfg = build_config(args)
    run_once(
        args.input,
        out_json,
        cfg,
        plots=args.plots,
        plots_dir=args.output_dir / "plots",
        reject_stereo=args.reject_stereo,
    )


if __name__ == "__main__":
    main()
