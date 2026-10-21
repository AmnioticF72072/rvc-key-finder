# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Dual-engine key-finder: `key_finder.py` combines source-audio key-finding
  (engine 1) with model-range-from-weights (engine 2) and cross-validates.
- `tools/read_pitch_range.py`: read a voice model's vocal range by diffing its
  `enc_p.emb_pitch` (256-slot Mel-quantized pitch embedding) against the base
  model.

### Fixed
- Spectral-energy arbitration in `analyze_pitch.py`: the correct octave-error
  test compares energy at **f0 vs 2×f0** (not f0 vs f0/2). Verified on a real
  case — G.E.M. "Where Did U Go" (175 Hz was a subharmonic; 350 Hz was the true
  fundamental, 3.85 M × energy).
- Shift-suggestion sign: `need = -12·log2(p95/material_max)` (negative = lower
  the key).

## [0.1.0] - 2026-10-22

### Added
- `analyze_pitch.py`: dual-range autocorrelation pitch detection with
  spectral-energy arbitration against octave/harmonic errors.
- `pipeline/finish.py`: zero-original-vocal finishing with smoothstep fades
  (never hard-cuts).
- `pipeline/mix_promax.py`: Pro Max vocal chain — reverb high-pass return,
  parallel compression, linear (measure-then-shift) mastering.
- `tools/clean_vocal.py`: harmonic-gate de-bleeding.
- `tools/find_glitch.py`: hallucination / jump / overshoot scan.
- `tools/fir_match.py`: firequalizer-based spectral-envelope matching.
- `tools/decrypt_kgm.py`: self-contained KuGou KGM/KGMA/VPR decryptor.
- `docs/PITFALLS.md`, `docs/RECIPES.md`: pitfall encyclopedia + parameter recipes.
