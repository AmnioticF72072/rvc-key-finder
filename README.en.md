# rvc-key-finder

[中文](README.md) · English

A small tool for figuring out the pitch shift when making RVC song covers.

The first step of any cover is deciding how many semitones to shift the source
vocal so it lands in your model's comfortable range. Get this wrong and nothing
else matters. This tool helps me (and you) get that step right.

---

## What problem it solves

Most people key an RVC cover like this: run f0 through RMVPE or crepe, look at
the median, then fill in a "-5" or "-8" by feel. Two things can go wrong there.

### Problem 1: autocorrelation picks the wrong octave

When autocorrelation searches for a peak across 60–1200 Hz, it can lock onto:

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

### Problem 2: model range is guessed, not read

Most people estimate their model's range by listening to the training material
and guessing an upper limit. But the answer is actually in the weights: RVC's
pitch embedding records which pitches the model saw during training.

---

## What this does

| Common approach | Here |
|---|---|
| Read the f0 median directly | Find peaks in 60–250 Hz *and* 60–1200 Hz separately |
| Octave errors ignored | Compare energy at f0 vs 2×f0 — whichever is stronger is the fundamental |
| Shift guessed by hand | Automatic suggestion, rounded to an even number of semitones |
| Model range estimated by ear | Diff the pitch embedding against the base model |

I didn't find the first three implemented anywhere in the RVC-cover space.
Generic pitch detection has octave correction, but I haven't seen it applied to
RVC cover key-finding. The "read the range off the weights" idea has a
closed-source precedent (Pop Maker Studio); this is an open implementation that
also combines it with source-audio key-finding.

---

## Usage

```bash
pip install numpy soundfile soxr

# both: source-audio key-finding + model range from weights
python key_finder.py --vocal vocals.wav \
    --model my_model.pth --base base_model.pth --material-max 375

# source audio only (no torch needed)
python analyze_pitch.py --vocal vocals.wav --material-max 375

# model range only
python tools/read_pitch_range.py --model my_model.pth --base base_model.pth
```

- `--material-max`: the highest *comfortable* pitch (Hz) in your model's
  training material (mine is about 375 Hz).
- `--model` + `--base`: enables the weights engine (requires torch).

Sample output:

```
【source-audio key-finding】
spectral-energy arbitration: 2×low-range / low-range = 3561629.2 ×
  -> 350 Hz is the true fundamental (low range was a subharmonic)
recommended shift: -8 semitones

【model range from weights】
trained slots: 127 / 256 -> 53 ~ 940 Hz

cross-check: 350 Hz falls inside the model range — consistent.
```

---

## How it works

### Source audio

1. FFT-accelerated autocorrelation per frame
2. Find the peak in two ranges separately: 60–250 Hz and 60–1200 Hz
3. When they differ by ≈ 2×, compare energy at f0 vs 2×f0:
   - 2×f0 has significantly more energy → f0 was a subharmonic (female voice)
   - f0 has significantly more energy → f0 is the fundamental (male voice)
4. Take the 95th percentile of the true fundamental, derive the shift in
   semitones, rounded to an even number

Pure numpy, no RMVPE/Applio needed.

### Model range (read the weights, no audio)

RVC encodes pitch with a 256-slot pitch embedding (`enc_p.emb_pitch`,
256×hidden), quantized on the Mel scale (50–1100 Hz). During fine-tuning, only
the slots whose pitch actually appears in the training audio move relative to
the base model. Diffing the two embeddings shows which pitches were seen.

The two approaches are independent and complementary: the source-audio one can
pick the wrong octave, the weights one doesn't touch audio at all; the weights
one only says "which pitches were seen" (not "which are comfortable"), which
the material ceiling narrows down.

---

## Other tools

`pipeline/` and `tools/` also have some things I wrote while making covers
(zero-original-vocal finishing, de-bleeding, a mixing chain, glitch checking,
etc.). They're not the focus here, but I left them in. See
[docs/PITFALLS.md](docs/PITFALLS.md) (Chinese).

## Docs

- [docs/PITFALLS.md](docs/PITFALLS.md) — pitfall notes with measurements (Chinese)
- [docs/RECIPES.md](docs/RECIPES.md) — parameter recipes (Chinese)
- [NOTICE](NOTICE) — what was AI-assisted, and third-party attribution

## License

MIT. Most of the code was written by an AI assistant; I did the debugging,
measurements and direction (see [NOTICE](NOTICE)).

## Disclaimer

There are no songs, voice models or copyrighted material here, just tools and
methods. Only use it on audio you have the rights to.
