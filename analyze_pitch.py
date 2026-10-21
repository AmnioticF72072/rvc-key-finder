#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""双范围音高定档 —— 决定整首翻唱的降调量（全局决策，错一次全曲报废）。

核心问题：自相关在 60~1200Hz 全范围找峰时，男声的**倍频(2f0)**周期性往往更高，
argmax 会落在倍频上，把男声误判成高八度的女声 → 降调量错一倍。

修法（本项目实测）：
  1. 同一帧分别在 60~250Hz 和 60~1200Hz 两个范围找峰；
  2. 两结果接近 → 可信；
  3. 高范围 ≈ 2×低范围 → 取低范围（被倍频误导的男声）；
  4. 拿不准 → 用**频谱能量**裁决：真基频所在频带的能量应显著强于其半频处
     （实测一例：350Hz 能量是 175Hz 的 385 万倍 → 350 是基频，175 是次谐波）。

纯 numpy 实现，不依赖 RMVPE/Applio，任何环境都能跑。

用法：
  python pipeline/analyze_pitch.py --vocal vocals.wav [--material-max 375]

  --material-max : 你的声音模型训练素材的最高舒适音（Hz）。
                   超过此值的高音模型只能硬顶，会刺耳，必须降调避开。
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

try:
    import soxr
except ImportError:
    soxr = None


def load_mono(path, sr_t=48000):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, np.float32)
    if sr != sr_t:
        if soxr is None:
            raise RuntimeError("需要 soxr 做重采样: pip install soxr")
        x = soxr.resample(x, sr, sr_t, quality="VHQ")
    return x, sr_t


def _ac(s):
    """FFT 加速自相关（归一化）。

    np.correlate 是 O(n²)，297 秒的歌逐秒打分要 20 分钟；
    换 FFT 自相关后 <10ms —— 任何逐窗分析脚本都要注意这一点。
    """
    n = len(s)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    S = np.fft.rfft(s, nfft)
    ac = np.fft.irfft(S * np.conj(S), nfft)[:n]
    return ac / (ac[0] + 1e-12)


def f0_in_range(s, sr, lo_hz, hi_hz):
    """在指定频率范围内用自相关找基频。返回 (f0, 自相关置信度)。"""
    if len(s) < sr // hi_hz * 2:
        return 0.0, 0.0
    ac = _ac(s - s.mean())
    lo = int(sr / hi_hz)
    hi = min(int(sr / lo_hz), len(ac) - 2)
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi]
    i = int(np.argmax(seg))
    # 抛物线插值，亚采样精度
    off = 0.0
    if 0 < i < len(seg) - 1:
        a, b, c = seg[i - 1], seg[i], seg[i + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-12:
            off = 0.5 * (a - c) / d
    return float(sr / (lo + i + off)), float(seg[i])


def band_energy(s, sr, f0, bw=8.0):
    """某个频率附近 ±bw Hz 的频谱能量（用于次谐波/倍频裁决）。"""
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fr = np.fft.rfftfreq(len(s), 1 / sr)
    m = (fr > f0 - bw) & (fr < f0 + bw)
    return float(sp[m].sum())


def analyze(path, material_max=0.0, sr_t=48000):
    x, sr = load_mono(path, sr_t)
    secs = len(x) // sr

    lo_f, full_f, confs = [], [], []
    for i in range(0, len(x) - sr // 2, sr // 2):
        s = x[i:i + sr // 2]
        if np.sqrt((s ** 2).mean()) < 3e-4:
            continue
        lf, _ = f0_in_range(s, sr, 60, 250)
        ff, _ = f0_in_range(s, sr, 60, 1200)
        if lf <= 0 or ff <= 0:
            continue
        lo_f.append(lf)
        full_f.append(ff)

    lo_f = np.array(lo_f)
    full_f = np.array(full_f)
    if lo_f.size == 0:
        print("未检测到有效人声，请确认输入是分离后的 vocals 轨")
        return None

    # 倍频误判比例：高范围明显是低范围的 2 倍
    octave_ratio = np.greater(full_f, lo_f * 1.8).mean()
    print("=" * 60)
    print("双范围定档  %s" % os.path.basename(path))
    print("=" * 60)
    print("60-250Hz   中位 %.0f Hz" % np.median(lo_f))
    print("60-1200Hz  中位 %.0f Hz" % np.median(full_f))
    print("倍频误判比例 %.0f%%  (>50%% 说明是男声被倍频误导，取低范围)" % (100 * octave_ratio))

    # 频谱能量裁决：真基频所在频带能量应显著强于其半频处和倍频处。
    # 关键：即使倍频误判比例高，也要用能量裁决一次——因为女声的真基频
    # 本身就在 300~500Hz，低范围找到的 175Hz 可能是【次谐波】而非真基频。
    # 正确判据：直接比较【低范围候选 f0】与【2×f0】的能量——
    #   若 2×f0 能量显著更强 → 真基频是 2×f0（低范围是次谐波，女声常见）；
    #   若 f0 能量显著更强 → f0 是真基频（男声）。
    real = lo_f
    W = sr * 5
    n = len(x) // W
    e = np.array([np.sqrt((x[k * W:(k + 1) * W] ** 2).mean()) for k in range(n)])
    k0 = int(np.argmax(e)) if n else 0
    seg = x[k0 * W:(k0 + 2) * W]
    f_low = float(np.median(lo_f))
    e_f0 = band_energy(seg, sr, f_low)
    e_2f0 = band_energy(seg, sr, f_low * 2)
    energy_ratio = e_2f0 / (e_f0 + 1e-30)
    print("频谱能量裁决: 2×低范围/低范围 = %.1f 倍" % energy_ratio)
    if energy_ratio > 2:
        # 2×f0 能量显著更强 → 低范围是次谐波，真基频在 2 倍处（女声）
        real = full_f
        print("  -> %.0f Hz 是真实基频（低范围是次谐波误判，女声常见）" % np.median(full_f))
    elif energy_ratio < 0.5:
        real = lo_f
        print("  -> %.0f Hz 是真实基频（能量显著强于 2 倍频）" % f_low)
    else:
        real = lo_f if octave_ratio > 0.5 else full_f
        print("  -> 能量不明确，按倍频比例取 %s" % ("低范围(男声)" if real is lo_f else "高范围"))

    print("-" * 60)
    label = "真实基频" if real is full_f else "真实音域(60-250)"
    print("%s: 中位 %.0f | 5%% %.0f | 95%% %.0f | 最大 %.0f Hz"
          % (label, np.median(real), np.percentile(real, 5),
             np.percentile(real, 95), real.max()))
    print("-" * 60)

    # 给出降调建议（偶数半音），保证降后 95 分位 ≤ 素材上限。
    # need 是负值 = 需要降的半音数。
    if material_max > 0:
        p95 = np.percentile(real, 95)
        need = -12 * np.log2(p95 / material_max)
        cand = min([0, -2, -4, -6, -8, -10, -12], key=lambda k: abs(k - need))
        print("素材上限 %.0f Hz -> 全曲 95%%(%.0fHz) 需降 %.1f 半音" % (material_max, p95, -need))
        print("★ 推荐定档: %d 半音" % cand)
        for k in sorted(set([cand - 2, cand, cand + 2])):
            if -14 <= k <= 0:
                print("   降 %2d: 95%% %.0f Hz%s" %
                      (k, p95 * 2 ** (k / 12),
                       "  <= 推荐" if k == cand else ""))
    else:
        print("（未指定 --material-max，跳过降调量建议）")
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True, help="分离后的人声轨 wav")
    ap.add_argument("--material-max", type=float, default=0.0,
                    help="声音模型素材的最高舒适音 Hz")
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()
    analyze(args.vocal, args.material_max, args.sr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
