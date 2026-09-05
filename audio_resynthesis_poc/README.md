# Parametric Audio Resynthesis POC

A classical-signal-processing experiment:

> Can a monophonic recording be analyzed into a **finite, human-inspectable** set of synthesis parameters, then reconstructed so playback is perceptually close to the original?

This is **not** a sampler product, VST, plugin, or ML autoencoder. It is a lab for testing whether a *structured* parameterization can support resynthesis, morphing, and transposition.

---

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
python resynthesize.py output/parameters.json --play          # A/B playback
python resynthesize.py output/parameters.json --ab            # A/B WAV: resynth then original
```

Related experiments:

```bash
# Vertical morph (same pitch, e.g. two velocities)
python morph.py output/A/parameters.json output/B/parameters.json --trim 3

# Pitch a single parameterized note through a major scale
python scale.py output/A/parameters.json --method synthesize_then_shift

# Horizontal walk between two differently pitched notes
python horizontal_scale.py output/LOW/parameters.json output/HIGH/parameters.json
```

---

## Architecture

```text
output = harmonic_partials + transient_component + stochastic_noise
```

| Component | What is stored | How it is made |
|---|---|---|
| Global meta | sample rate, duration, peak/RMS, seed | WAV + level stats |
| Amplitude envelope | attack-biased control points | frame RMS → downsample |
| Fundamental | f0 trajectory | `librosa.pyin` |
| Harmonic partials | freq / amp / unwrapped phase tracks near `k·f0` | STFT peak tracking (inharmonicity-aware) |
| Transient | short amp + band spectral envelope | onset window on residual |
| Noise | time-varying RMS × log bands | residual, flatness-weighted |

Budgets live in [`config.py`](config.py). Control-point density is auto-scaled toward the STFT hop rate so phase/amp tracks stay usable on real instruments.

The parameter JSON contains **no PCM** and is not a disguised spectrogram dump.

---

## Lessons from the experiments

These are the durable patterns. They matter more than any single SNR number.

### 1. Split “capture” from “playback transpose”

| Stage | What worked |
|---|---|
| **Capture** | Analyze → structured params → resynthesize at *native* pitch |
| **Transpose** | Phase-vocoder (or similar) on the resynthesized audio |
| **Did not work** | Scaling partial frequencies in parameter space (“stretch the harmonic template”) |

Naive parametric transpose of a piano-like tone becomes a **bell / chime**: a clean additive stack with the wrong brightness and missing hammer character. Formant remaps and noise boosts help only marginally.

**Proven pattern for up-pitching one sample:**

```text
audio → analyze → parameters → resynthesize @ native pitch
                                      ↓
                         phase-vocoder pitch shift
                                      ↓
                                 played note
```

Among three scale methods tested on a C4 piano sample, **resynth-then-PV** was the most realistic for nearly an octave of upward shift (better than parametric freq scaling; competitive with PV on the original WAV).

### 2. Name two different morphs differently

| Name | Meaning | What worked |
|---|---|---|
| **Vertical** | Same pitch, different takes (velocity, round-robin) | Parameter morph of amps / ratios / envelopes; **keep one side’s measured phase** |
| **Horizontal** | Different pitches on the keyboard | **Not** inventing mid-pitch partial stacks. Instead: PV-shift both neighbors to the target pitch, then crossfade by keyboard distance `t` |

```text
Vertical (e.g. v10 ↔ v13 at C4):
  params_A ──morph(t)──► synthesize

Horizontal (e.g. C4 ↔ C5):
  resynth_A ──PV to target──┐
                             ├─ crossfade(t) ──► note
  resynth_B ──PV to target──┘
```

Pure parameter morph across pitch (**horizontal parametric**) recreates the same additive/bell failure as parametric transpose: intermediates are unrecorded pitches synthesized as fresh harmonic stacks.

### 3. Phase is not a free morph knob

- Measured STFT phase is essential for piano-like reconstruction quality.
- Absolute phases of two separately recorded notes are **unrelated**.
- Lerping those phases, or pairing one note’s phase with another’s frequencies, produces **metallic / bell** artifacts.
- **Same-pitch morphs:** keep endpoint A’s phase; morph amplitudes and spectral shape.
- **Pitch changes:** rebuild phase as `IF(new_freq) + residual`, or avoid param-space pitch change entirely (prefer PV on audio).

### 4. Trust acoustic f0, not filenames

Sample libraries disagree about which octave is “C3” vs “C4”. Always print and check `fundamental.frequency_hz_mean`.

In this POC, files labeled `MF C3` / `MF C4` analyzed near **~263 Hz and ~526 Hz** (about an octave). Horizontal tests used that measured span.

Also: missing-fundamental / strong-even-harmonic piano notes can fool pitch trackers into the wrong octave. Inspect spectra when results look off by 12 semitones.

### 5. Inharmonicity and control-point density matter

- Exact `k·f0` bin sampling misses piano partials (sharp of harmonic). Track local peaks near each harmonic.
- Aggressively downsampling trajectories (tens of points on a long note) destroys phase/amp detail. Density near the STFT hop is much safer for real instruments.
- Harmonic synthesis via **sparse STFT peak placement + ISTFT** outperformed naive time-domain oscillators for these tones.

### 6. Metrics vs ears

- SNR alone does not prove perceptual identity (and can look poor when residual is mostly phase error).
- Spectral convergence + A/B listening (`--ab`, `--play`) are the practical judges for sampler-like goals.
- Ablations (e.g. `--ablation` on partial count) reveal whether quality gains are real or just “more parameters ≈ more PCM”.

### 7. What the parameterization is good for

| Use | Verdict in this POC |
|---|---|
| Lossy but intelligible capture of a monophonic note | Promising |
| Same-pitch morph (velocity / RR-style) | Promising if phase is handled carefully |
| Inspectable / editable representation | Yes — by design |
| Parametric transpose across large intervals | Poor for piano-like sounds (bells) |
| Parametric mid-pitch invention (horizontal) | Poor for piano-like sounds (bells) |
| Native resynth + PV transpose | Good |
| Neighbor resynth + PV + crossfade (horizontal) | Good |

### 8. Working architecture sketch for a future parametric sampler

```text
                 ┌── vertical morph (params) ──┐
 record note ──► analyze ──► parameters ───────┼──► resynthesize @ native
                 │                              │
                 │                              ▼
                 │                     playback pitch / key distance
                 │                              │
                 │              ┌───────────────┼───────────────┐
                 │              ▼               ▼               ▼
                 │         PV shift        PV+crossfade     (avoid raw
                 │         (one sample)    (two neighbors)   param stretch)
                 └─────────────────────────────────────────────┘
```

**Store and edit in parameter space. Transpose and fill keyboard gaps in the audio domain after resynthesis.**

That split is the main product-shaped lesson of the POC.

---

## Scientific caveat: exact vs parametric reconstruction

**Exact waveform reconstruction** reproduces every PCM sample. Codecs aim at this.

**Parametric / perceptual reconstruction** uses a compact generative description that *sounds* like the source. This POC targets the second.

Failure modes (bell on transpose, metallic morphs) are scientifically useful: they show where a harmonic+noise model stops being a piano and becomes an additive synthesizer.

---

## Metrics

Defined in [`metrics.py`](metrics.py):

- **RMS / NMSE / SNR** — time-domain error (phase-sensitive)
- **Spectral difference** — mean `|Δ log |S||`
- **Spectral convergence** — `1 − ‖|Sₓ|−|Sᵧ|‖ / ‖|Sₓ|‖`
- **Log-mel distance** — perceptual-*ish* proxy only

---

## Tools

| Script | Role |
|---|---|
| `analyze.py` | Audio → `parameters.json` (+ optional plots / ablation) |
| `resynthesize.py` | Params → WAV; `--play`, `--ab`, metrics |
| `morph.py` | Vertical (or general) param morph listening strip |
| `scale.py` | Major scale from one note (`synthesize_then_shift` / `parametric` / `original_shift`) |
| `horizontal_scale.py` | Scale between two pitches (`pv_crossfade` / `parametric`) |
| `pitch.py` | Transpose helpers + audio PV shift |
| `interpolate.py` | Structured parameter interpolation |

---

## Project layout

```text
audio_resynthesis_poc/
├── analyze.py / resynthesize.py
├── analysis.py / synthesis.py / interpolate.py / pitch.py
├── morph.py / scale.py / horizontal_scale.py
├── metrics.py / plots.py / config.py / io_utils.py
├── requirements.txt
├── Test Samples/          # drop real notes here
├── input/example.wav      # synthetic smoke-test tone
├── output/                # runs, plots, listening WAVs
└── examples/parameters_example.json
```

---

## What this experiment tells us about feasibility

1. A **hybrid harmonic + transient + noise** parameterization can capture monophonic notes well enough that downstream PV transposition still sounds like the instrument.
2. The representation is **semantically structured** enough for same-pitch morphs (vertical).
3. **Do not** equate “parametric” with “transpose by scaling frequencies.” That path fails for inharmonic / hammered instruments in this setup.
4. A feasible sampler architecture is: **params for identity and morphing; resynth + PV (+ neighbor crossfade) for pitch and keyboard filling.**
5. Listening comparisons beat metric-only iteration when the failure mode is “sounds like a bell.”
