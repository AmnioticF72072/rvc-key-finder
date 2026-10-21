# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""
频谱匹配（FIR 版）——用 ffmpeg 的 firequalizer 一次性精确匹配频谱包络。

为什么不用级联 equalizer
------------------------
ffmpeg 的 `equalizer` 是双二阶滤波器，相邻频段（1/3 倍频程间隔）的响应会严重
互相叠加，导致实际响应偏离设计值：实测里想削 6 dB 的频段最后被削了 9 dB，
而中间频段又相对被抬亮。用 30 段 equalizer 串联必然出现这种"过冲"。

`firequalizer` 是单条 FIR（默认 4096 抽头），在指定的 gain_entry 之间插值，
各点互不干扰，能精确实现任意平滑曲线。

做法
----
1. 算目标（真人录音）与输入（模型输出）的 1/6 倍频程功率谱
2. 用 500~1600 Hz 的平均能量把两者对齐（消除整体电平差）
3. 增益 = 目标 - 输入，乘 strength，限幅，在对数频率轴上平滑
4. 输出 firequalizer 的 gain_entry 串（20 Hz ~ 20 kHz 共 60+ 点）

用法
----
  python C2_fir_match.py --target <真人> --input <模型输出> \
      --strength 0.8 --out-eq eq_fir.txt
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa

SR = 44100
FMIN, FMAX = 22.0, 19500.0
NPOINT = 72          # gain_entry 点数（1/6 倍频程密度）
ANCHOR = (500.0, 1600.0)   # 电平对齐锚点频段


def load_mono(path, sr=SR):
    x, s = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if s != sr:
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    return x


def power_spectrum(path, sr=SR, n_fft=8192):
    """返回 (频率轴, 有声帧平均功率谱)"""
    x = load_mono(path, sr)
    r = librosa.feature.rms(y=x, frame_length=2048, hop_length=512)[0]
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=512))
    n = min(S.shape[1], r.size)
    if n < 8:
        raise RuntimeError("音频太短: %s" % path)
    keep = r[:n] > np.percentile(r[:n], 55)
    if keep.sum() < 4:
        keep = np.ones(n, dtype=bool)
    P = (S[:, :n][:, keep] ** 2).mean(axis=1)
    fr = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return fr, P


def log_grid():
    """对数等距频率点"""
    return np.geomspace(FMIN, FMAX, NPOINT)


def band_db(fr, P, grid, frac=1 / 6):
    out = np.zeros(len(grid))
    for i, f in enumerate(grid):
        lo, hi = f / 2 ** frac, f * 2 ** frac
        k = (fr >= lo) & (fr < hi)
        out[i] = 10 * np.log10(max(P[k].sum() if k.any() else 1e-16, 1e-16))
    return out


def norm_curve(fr, P, grid):
    """按锚点频段（500~1600Hz）归一化的频谱曲线

    用锚点归一化的目的：消除两段音频之间的整体电平差，
    只比较"相对频谱形状"——这才是说话人的音色特征。
    """
    b = band_db(fr, P, grid)
    idx = (grid >= ANCHOR[0]) & (grid < ANCHOR[1])
    if idx.sum() == 0:
        idx = np.ones(len(grid), dtype=bool)
    return b - float(b[idx].mean())


def build(target, inp, strength=0.8, limit=9.0, smooth=1.0):
    grid = log_grid()
    fr_t, Pt = power_spectrum(target)
    fr_i, Pi = power_spectrum(inp)
    t = norm_curve(fr_t, Pt, grid)
    i = norm_curve(fr_i, Pi, grid)
    gain = (t - i) * strength
    gain = np.clip(gain, -limit, limit)
    # 在对数频率轴上平滑（点数域平滑 ≈ 对数频率域平滑）
    k = max(1, int(round(smooth * 6)))
    ker = np.hanning(2 * k + 1)
    ker /= ker.sum()
    gain = np.convolve(np.pad(gain, k, mode="edge"), ker, mode="valid")
    gain = np.clip(gain, -limit, limit)
    return grid, t, i, gain


def make_chain(grid, gain, hp=85, lp=None):
    entries = []
    # 20Hz 起始用高通处理，不放进 gain_entry
    for f, g in zip(grid, gain):
        if f < hp * 1.15:
            continue
        entries.append("entry(%g,%.2f)" % (f, g))
    parts = []
    if hp:
        parts.append("highpass=f=%d:poles=2" % hp)
    parts.append("firequalizer=gain_entry='%s'" % ";".join(entries))
    if lp:
        parts.append("lowpass=f=%d:poles=2" % lp)
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--limit", type=float, default=9.0)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--hp", type=int, default=85)
    ap.add_argument("--out-eq", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    grid, t, i, g = build(args.target, args.input, args.strength,
                          args.limit, args.smooth)
    chain = make_chain(grid, g, args.hp)

    print("=" * 76)
    print(" FIR 频谱匹配：%s -> %s" % (os.path.basename(args.input),
                                       os.path.basename(args.target)))
    print("=" * 76)
    print(" %-9s %10s %10s %10s" % ("频率Hz", "目标dB", "输入dB", "修正dB"))
    print("-" * 76)
    for f, a, b, c in zip(grid, t, i, g):
        if f < 40:
            continue
        step = 1 if f < 200 else (2 if f < 1000 else 4)
        bar = ("+" if c > 0 else "-") * min(int(abs(c) * 1.4), 24)
        print(" %-9.0f %10.1f %10.1f %+10.2f  %s" % (f, a, b, c, bar))
    print("-" * 76)
    print(" 修正前总偏差 %.1f dB -> 理论残留 %.1f dB" % (
        np.abs(t - i).sum(), np.abs(t - i - g).sum()))
    if args.out_eq:
        open(args.out_eq, "w", encoding="utf-8").write(chain)
        print(" 已写入 %s" % args.out_eq)
    if args.json:
        json.dump({"freq": grid.tolist(), "target": t.tolist(),
                   "input": i.tolist(), "gain": g.tolist(), "chain": chain},
                  open(args.json, "w", encoding="utf-8"), indent=2)
    print()
    print(" ffmpeg -af：")
    print(" " + chain[:400] + (" ..." if len(chain) > 400 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
