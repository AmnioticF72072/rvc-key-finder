#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""异常体检 —— 找三类问题：
  1 幻觉：源几乎无声，输出却明显有声（= 突然"啊"一下）
  2 突变：输出电平在 100ms 内跳变过大（= 咔哒/杂音）
  3 突响：输出比源响很多（模型放大）

用法：
  python tools/find_glitch.py --src 源人声.wav --voc 成品人声.wav
"""
import argparse
import sys

import numpy as np
import soundfile as sf

try:
    import soxr
except ImportError:
    soxr = None

SR = 48000


def load(path, sr_t=SR):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = np.ascontiguousarray(x, np.float32)
    if sr != sr_t:
        if soxr is None:
            raise RuntimeError("需要 soxr: pip install soxr")
        x = soxr.resample(x, sr, sr_t, quality="VHQ")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源人声 wav")
    ap.add_argument("--voc", required=True, help="成品人声 wav")
    args = ap.parse_args()

    src = load(args.src)
    voc = load(args.voc)
    n = min(len(src), len(voc))
    src, voc = src[:n], voc[:n]

    W = int(0.05 * SR)  # 50ms 窗
    nw = n // W

    def env(x):
        return 20 * np.log10(np.sqrt((x[:nw * W].reshape(nw, W) ** 2).mean(1)) + 1e-12)

    es, ev = env(src), env(voc)
    print("分析 %.1fs，50ms 窗 %d 个" % (n / SR, nw))

    # 1 幻觉
    hall = np.where((es < -48) & (ev > -34))[0]
    runs = []
    if hall.size:
        st = prev = hall[0]
        for i in hall[1:]:
            if i - prev <= 3:
                prev = i
            else:
                runs.append((st, prev))
                st = prev = i
        runs.append((st, prev))
    runs = [(a, b) for a, b in runs if b - a >= 2]
    print("\n=== 1. 幻觉段（源<-48dB 但输出>-34dB，>=100ms）: %d 段 ===" % len(runs))
    for a, b in runs[:30]:
        print("   %6.2fs ~ %6.2fs (%.2fs)  源 %.0f dB / 输出 %.0f dB"
              % (a * W / SR, (b + 1) * W / SR, (b - a + 1) * W / SR,
                 es[a:b + 1].mean(), ev[a:b + 1].mean()))

    # 2 突变
    d = np.abs(np.diff(ev))
    jump = np.where(d > 8.0)[0]
    print("\n=== 2. 电平突变（相邻 50ms 差 >8 dB）: %d 处 ===" % jump.size)
    for i in jump[:25]:
        print("   %6.2fs  %+.1f dB  (%.0f -> %.0f)"
              % (i * W / SR, ev[i + 1] - ev[i], ev[i], ev[i + 1]))

    # 3 突响
    loud = np.where((es > -45) & (ev > es + 12))[0]
    print("\n=== 3. 输出比源突响 >12dB: %d 窗 ===" % loud.size)
    segs = []
    if loud.size:
        st = prev = loud[0]
        for i in loud[1:]:
            if i - prev <= 3:
                prev = i
            else:
                segs.append((st, prev))
                st = prev = i
        segs.append((st, prev))
    for a, b in segs[:15]:
        print("   %6.2fs ~ %6.2fs  +%.0f dB"
              % (a * W / SR, (b + 1) * W / SR, (ev[a:b + 1] - es[a:b + 1]).mean()))

    print("\n=== 汇总 ===")
    print("幻觉 %.2fs | 突变 %d 处 | 突响 %.2fs"
          % (sum((b - a + 1) for a, b in runs) * W / SR, jump.size,
             sum((b - a + 1) for a, b in segs) * W / SR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
