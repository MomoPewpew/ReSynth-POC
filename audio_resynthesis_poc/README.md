# Parametric Audio Resynthesis POC

A small classical-signal-processing experiment:

> Can an arbitrary monophonic recording be analyzed into a **finite, human-inspectable** set of synthesis parameters, then reconstructed so playback is perceptually close to the original?

This is **not** a sampler product, VST, plugin, or ML autoencoder. It is a diagnostic analyze → parameters → resynthesize loop.

## Setup

```bash
cd audio_resynthesis_poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: regenerate the bundled synthetic example
python generate_example.py
```

## Quick start

```bash
python analyze.py input/example.wav --plots
python resynthesize.py output/parameters.json --plots
# Optional A/B playback:
python resynthesize.py output/parameters.json --play
```

Outputs:

- `output/parameters.json` — structured parameters (no PCM)
- `output/resynthesized.wav` — reconstruction
- `output/plots/` — waveform / spectrogram / trajectory diagnostics

Ablation (partial count vs quality):

```bash
python analyze.py input/example.wav --ablation
```

## Architecture

```text
output = harmonic_partials + transient_component + stochastic_noise
```

| Component | What is stored | How it is made |
|---|---|---|
| Global meta | sample rate, duration, peak/RMS, seed | from the WAV header + levels |
| Amplitude envelope | ~64 attack-biased control points | frame RMS → downsample |
| Fundamental | f0 trajectory | `librosa.pyin` |
| Harmonic partials | up to `MAX_PARTIALS` freq/amp tracks + initial phase | STFT samples at `k·f0` |
| Transient | short amp + band spectral envelope | onset window on residual |
| Noise | time-varying RMS × log bands | residual after harmonics, weighted by spectral flatness |

Budgets live in [`config.py`](config.py) and as CLI flags (`--max-partials`, `--envelope-points`, …). Raising them without measuring the quality tradeoff defeats the experiment.

## Parameter file

See [`examples/parameters_example.json`](examples/parameters_example.json). It is self-contained: resynthesis does not need the original WAV (except for metrics / A/B). The `seed` field makes stochastic noise deterministic.

The JSON intentionally stores **control points**, not a spectrogram and not PCM samples.

## Metrics

Printed after resynthesis when the original is available. Definitions are in [`metrics.py`](metrics.py):

- **RMS / NMSE / SNR** — time-domain error (sensitive to phase)
- **Spectral difference** — mean `|Δ log |S||`
- **Spectral convergence** — `1 − ‖|Sₓ|−|Sᵧ|‖ / ‖|Sₓ|‖`
- **Log-mel distance** — optional perceptual-*ish* proxy (not a proof of perceptual identity)

## Scientific caveat: exact vs parametric reconstruction

**Exact waveform reconstruction** means every PCM sample matches. Codecs and wavetable dumps aim at this.

**Parametric / perceptual reconstruction** means a compact generative description that *sounds* like the source. This POC targets the second.

If the waveform SNR is modest but spectrograms and listening are close, that is still a useful result: it shows which information a structured synthesizer captures, and which it loses (phase, micro-modulation, inharmonicity, etc.).

## Example results (synthetic `input/example.wav`)

On the bundled 1.5 s decaying harmonic tone + attack noise burst (defaults: 32 partials):

| | |
|---|---|
| PCM samples | 66,150 (~194 KB at 24-bit) |
| Scalar parameters | ~7,300 (~139 KB as indented JSON) |
| SNR | ~9.7 dB |
| Spectral convergence | ~0.88 |

Ablation on this file (8 harmonics in the source) plateaus early: more partials add parameters without improving SNR. That is expected and desirable — it means we are not “winning” by copying the waveform.

JSON text is verbose; packing the same floats as binary would be smaller (~7k × 8 bytes ≈ 57 KB). Longer real recordings usually improve the PCM∶parameter ratio more than this short clip does.

## What this experiment tells us about the feasibility of a parametric sampler

1. **A hybrid harmonic + transient + noise parameterization is workable** with classical analysis alone — no neural latent space required for a first cut.
2. **Most energy of stable pitched sounds lives in a small number of evolving partials.** Beyond that, extra oscillators buy little on highly harmonic material.
3. **Waveform SNR understates listening quality** when residual energy is phase error rather than missing timbre; spectral metrics and A/B listening matter more for “sampler-like” judgment.
4. **Transients and true noise need their own models.** Folding everything into harmonics either fails the attack or forces an explosion of parameters.
5. **Feasibility of a future parametric sampler is plausible** if the representation stays semantically structured (so sounds can later be interpolated / morphed). This repo deliberately stops before interpolation.

## Project layout

```text
audio_resynthesis_poc/
├── analyze.py          # CLI: audio → parameters.json
├── resynthesize.py     # CLI: parameters → WAV (+ metrics / --play)
├── analysis.py         # analysis algorithms
├── synthesis.py        # deterministic resynthesis
├── metrics.py          # objective scores
├── plots.py            # diagnostic figures
├── config.py           # budget knobs
├── io_utils.py         # WAV / JSON / size reporting
├── generate_example.py
├── requirements.txt
├── input/example.wav
├── output/
└── examples/parameters_example.json
```

## Future (not implemented)

Structured parameters are meant to support later experiments such as interpolating `parameters_A` and `parameters_B`. Do not add that here until the single-sound loop is understood.
