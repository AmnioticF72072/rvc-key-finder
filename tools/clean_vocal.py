#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""谐波门控清理 —— 压掉非谐波（伴奏残渣/和声叠层），只留最强基频及其谐波。

目的：让 RVC 的 f0 不在八度间乱跳。
原理：对每一帧估计 f0，构建以 f0 各次谐波为中心的高斯掩码，把掩码外的
能量压到 FLOOR（默认 0.32），从而剥掉与当前基频无关的声部（和声/伴奏残渣）。

用法：
  python tools/clean_vocal.py --src vocals.wav --out vocals_clean.wav \
      [--sigma 42] [--floor 0.32]
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

N, HOP = 2048, 512


def est_f0(s, sr):
    """自相关找基频（FFT 加速 + 抛物线插值）。"""
    s0 = s - s.mean()
    if float((s0 ** 2).sum()) < 1e-7:
        return 0.0, 0.0
    n = len(s0)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    S = np.fft.rfft(s0, nfft)
    ac = np.fft.irfft(S * np.conj(S), nfft)[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(sr / 1100), min(int(sr / 60), len(ac) - 2)
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi]
    i = int(np.argmax(seg))
    conf = float(seg[i])
    if conf < 0.26:
        return 0.0, conf
    off = 0.0
    if 0 < i < len(seg) - 1:
        a, b, c = seg[i - 1], seg[i], seg[i + 1]
        dd = a - 2 * b + c
        if abs(dd) > 1e-12:
            off = 0.5 * (a - c) / dd
    return float(sr / (lo + i + off)), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=42.0)
    ap.add_argument("--floor", type=float, default=0.32)
    args = ap.parse_args()

    x, sr = sf.read(args.src, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = np.ascontiguousarray(x, np.float32)

    win = np.hanning(N).astype(np.float32)
    freqs = np.fft.rfftfreq(N, 1 / sr)
    nfr = 1 + (len(x) - N) // HOP
    pad = np.pad(x, (N // 2, N // 2 + HOP))
    frames = np.lib.stride_tricks.sliding_window_view(pad, N)[::HOP][:nfr]

    # 逐帧估 f0
    f0 = np.zeros(nfr, np.float32)
    for k in range(nfr):
        f0[k] = est_f0(frames[k], sr)[0]

    # 中值滤波去八度跳变（7 帧）
    fp = np.pad(f0, 3, mode="edge")
    f0m = np.array([np.median(fp[i:i + 7]) for i in range(nfr)], np.float32)
    for i in range(nfr):
        if f0m[i] > 0 and f0[i] > 0:
            r = f0[i] / f0m[i]
            if 1.85 < r < 2.2 or 0.46 < r < 0.55:
                f0[i] = f0m[i]
    f0m[(f0m < 70) | (f0m > 1100)] = 0
    f0s = np.convolve(np.pad(f0m, 4, mode="edge"), np.ones(9, np.float32) / 9, mode="valid")

    STFT = np.fft.rfft(frames * win, axis=1)
    mag = np.abs(STFT)
    phase = np.angle(STFT)
    mask = np.full(mag.shape, args.floor, np.float32)
    for k in range(nfr):
        f = f0s[k]
        if f <= 0:
            mask[k, :] = 0.5
            continue
        for h in range(1, 13):
            fh = h * f
            if fh > sr / 2 - 200:
                break
            mask[k] = np.maximum(mask[k], np.exp(-0.5 * ((freqs - fh) / args.sigma) ** 2))
    mask = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, mask)
    mask = np.clip(mask, 0.30, 1.0)

    OUT = np.fft.irfft(mag * mask * np.exp(1j * phase), n=N, axis=1).astype(np.float32) * win
    y = np.zeros(len(pad) + N, np.float32)
    norm = np.zeros(len(pad) + N, np.float32)
    for k in range(nfr):
        i = k * HOP
        y[i:i + N] += OUT[k]
        norm[i:i + N] += win ** 2
    norm[norm < 1e-6] = 1.0
    y = (y / norm)[N // 2: N // 2 + len(x)]
    r0 = float(np.sqrt((x ** 2).mean()))
    r1 = float(np.sqrt((y ** 2).mean()) + 1e-12)
    y *= r0 / r1
    np.clip(y, -0.999, 0.999, out=y)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sf.write(args.out, y, sr, subtype="FLOAT")
    print("已写 %s | 有基频帧 %.1f%%" % (args.out, 100 * (f0s > 0).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
