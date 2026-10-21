# rvc-key-finder

[中文](README.md) · English

**Automatic key detection for RVC song covers — with octave-error protection.**

A dual-engine tool that finds the correct pitch-shift (in semitones) for an RVC
(Retrieval-based Voice Conversion) song cover. It combines two independent
approaches and cross-validates them — a combination that, to the best of our
research, does not exist anywhere else in the open-source world.

---

## The problem it solves

The very first step of any RVC cover is deciding **how many semitones to shift**
the source vocal so it lands inside your voice model's comfortable range.
**Get this wrong and everything downstream is wasted.**

Yet almost every tutorial and every existing cover project (AICoverGen,
xinxin-song, RVC-dstts, Ultimate RVC…) does the same thing:

> Run f0 through RMVPE/crepe → **eyeball the median → manually guess "-5 or -8"**.

Two real problems hide here that almost nobody addresses:

### Problem 1: Autocorrelation picks the wrong octave

When autocorrelation searches for a peak across 60–1200 Hz, it can lock onto:

- a male voice's **2nd harmonic (2f0)** — often *more* periodic than the
  fundamental — misclassifying a male voice as a female voice one octave up
- a female voice's **subharmonic (f0/2)** — misclassifying a female voice as male

The result: the shift is off by a full octave. A song that needs "-8" ends up
shifted "-12" or "0", sounding either piercing or muddy.

**This bit us for real.** A documented case:

> G.E.M. "Where Did U Go": autocorrelation found 175 Hz in the low range,
> suggesting a male voice. But spectral-energy arbitration found **350 Hz
> carrying 3.85 million × the energy of 175 Hz** — 175 Hz was only a subharmonic;
> 350 Hz was the true fundamental (female voice). The correct answer was "-8".

### Problem 2: Model range is guessed, not read

Almost everyone estimates their model's vocal range by "listening to the
training material and guessing an upper limit". But the answer is literally
**stored in the model weights**: RVC's pitch embedding records which pitches the
model ever saw during training.

---

## The two engines (the part nobody else does)

| Existing cover projects | This project |
|---|---|
| Read f0 median directly | **Dual-range cross-check**: find peaks in 60–250 Hz *and* 60–1200 Hz separately |
| Ignore octave errors | **Spectral-energy arbitration**: compare energy at f0 vs 2×f0 — whichever is stronger is the fundamental |
| Guess the shift by hand | **Automatic shift suggestion**: fit the 95th percentile to your material's ceiling, rounded to even semitones |
| Estimate model range by ear | **Read the weights**: diff the pitch embedding against the base model |

To the best of our research, the first three do not exist anywhere in the
RVC-cover niche (generic pitch-detection research has octave-correction, but
nobody applied it to RVC cover key-finding). And "reading the range off the
weights" has only a single closed-source precedent (Pop Maker Studio) — this
project is an independent open implementation that additionally *combines* it
with source-audio key-finding.

---

## Quick start

```bash
pip install numpy soundfile soxr

# Dual-engine: source-audio key-finding + model-range from weights
python key_finder.py --vocal vocals.wav \
    --model my_model.pth --base base_model.pth --material-max 375

# Engine 1 only (source audio, no torch needed)
python analyze_pitch.py --vocal vocals.wav --material-max 375

# Engine 2 only (read model range from weights)
python tools/read_pitch_range.py --model my_model.pth --base base_model.pth
```

- `--material-max`: the highest *comfortable* pitch (Hz) in your model's
  training material (this project measures ≈ 375 Hz for its own voice model).
- `--model` + `--base`: enables engine 2 (requires torch).

Sample dual-engine output:

```
【Engine 1 · source-audio key-finding】
Spectral-energy arbitration: 2×low-range / low-range = 3561629.2 ×
  -> 350 Hz is the true fundamental (low range was a subharmonic)
★ recommended shift: -8 semitones

【Engine 2 · model range from weights】
trained slots: 127 / 256 -> range 53 ~ 940 Hz

Cross-check: source fundamental 350 Hz falls inside the model range — consistent.
```

---

## How it works (30 seconds)

### Engine 1 — source audio (octave-error protection)

1. FFT-accelerated autocorrelation per frame (O(n log n) instead of O(n²))
2. Find the peak in **two ranges** separately: 60–250 Hz and 60–1200 Hz
3. When they differ by ≈ 2×, run **spectral-energy arbitration**:
   - 2×f0 has significantly more energy → f0 was a subharmonic (female voice)
   - f0 has significantly more energy → f0 is the true fundamental (male voice)
4. Take the 95th percentile of the true fundamental → derive the shift in
   semitones, rounded to an even number

Pure numpy, no RMVPE/Applio required.

### Engine 2 — model range (read the weights, no audio)

RVC encodes pitch with a 256-slot **pitch embedding** (`enc_p.emb_pitch`,
256×hidden), quantized on the Mel scale (50–1100 Hz). During fine-tuning, only
the slots whose pitch actually appears in the training audio move relative to
the base model. Diffing the trained model's embedding against the base model's
reveals which pitches were seen = the model's range.

The two engines are independent and complementary: engine 1 can octave-error,
engine 2 doesn't measure audio at all; engine 2 reports "which pitches were
seen" (not "which are comfortable"), which engine 1's material ceiling narrows
down to a usable range.

---

## Bundled cover utilities

`pipeline/` and `tools/` also contain other tools hardened during real cover
production (zero-original-vocal finishing, harmonic-gate de-bleeding, a
Pro Max mixing chain, glitch detection, etc.). They are not the focus of this
repository but are released together as battle-tested artifacts. See
[docs/PITFALLS.md](docs/PITFALLS.md) (Chinese).

## Documentation

- [docs/PITFALLS.md](docs/PITFALLS.md) — pitfall encyclopedia with measured data (Chinese)
- [docs/RECIPES.md](docs/RECIPES.md) — parameter recipes (Chinese)
- [NOTICE](NOTICE) — AI-assistance disclosure + third-party attribution

## License

MIT. Code written by an AI coding assistant driven by a human's debugging and
measurements (see [NOTICE](NOTICE)).

## Disclaimer

This project ships no songs, voice models, or copyrighted material. Use it only
on audio you have the rights to.
