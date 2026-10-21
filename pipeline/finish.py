#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""零原唱收尾 —— 全曲只用模型输出，绝不混入任何原唱。

原则与做法：
  1) 两套参数(A/B)各自转换，逐秒按【周期性*0.75 + 谐波度*0.25】打分择优；
  2) 连续同选段拼接，段间 30ms 交叉淡化（同音色，不会突变）；
  3) 仍不达标的段：按严重程度【衰减音量】让伴奏盖住，而不是换原唱；
  4) 源静音而模型造声的段：平滑沉到 -30dB（那是模型幻觉，不是人声）。

★ 绝不硬切到 0：硬边界会产生咔哒/宽带噪声。所有衰减/退让边界一律 smoothstep。

用法：
  # 双套参数择优（A/B 各自用不同参数转换）
  python tools/finish.py --vocal-src vocals.wav \
      --a outA.wav --b outB.wav --out 成品人声.wav

  # 单套参数（音色全程一致，杜绝接缝电音 —— 推荐）
  python tools/finish.py --vocal-src vocals.wav \
      --a out.wav --single --out 成品人声.wav
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import soundfile as sf

try:
    import soxr
except ImportError:
    soxr = None

SR = 48000
GOOD = 0.46      # 达标线：只有音高确实不明确才衰减
SILENT = -50.0   # 源静音阈值


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


def _ac(s):
    n = len(s)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    S = np.fft.rfft(s, nfft)
    ac = np.fft.irfft(S * np.conj(S), nfft)[:n]
    return ac / (ac[0] + 1e-12)


def harm_conf(s, sr=SR):
    """谐波度：8 次谐波能量占总能量比（衡量音高是否明确）。"""
    if len(s) < 512:
        return 0.0, 0.0
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fr = np.fft.rfftfreq(len(s), 1 / sr)
    tot = sp[(fr > 80) & (fr < 7900)].sum() + 1e-12
    ac = _ac(s - s.mean())
    lo, hi = int(sr / 1200), min(int(sr / 60), len(ac) - 1)
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi]
    f0 = sr / (lo + int(np.argmax(seg)))
    h = 0.0
    for k in range(1, 9):
        fk = k * f0
        if fk > 7900:
            break
        m = (fr > fk * 0.965) & (fr < fk * 1.035)
        h += sp[m].sum()
    return min(float(h / tot), 1.0), float(seg.max())


def f0_of(s, sr=SR):
    if len(s) < 1024:
        return 0.0, 0.0
    ac = _ac(s - s.mean())
    lo, hi = int(sr / 1200), min(int(sr / 60), len(ac) - 1)
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi]
    return float(sr / (lo + int(np.argmax(seg)))), float(seg.max())


def voiced_rms(x):
    seg = int(0.5 * SR)
    es = [np.sqrt((x[a:a + seg] ** 2).mean()) for a in range(0, len(x) - seg, seg)
          if np.sqrt((x[a:a + seg] ** 2).mean()) > 1e-4]
    return float(np.median(es)) if es else 1e-9


def smoothstep(n):
    u = np.linspace(0, 1, max(n, 2), dtype=np.float32)
    return u * u * (3 - 2 * u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal-src", required=True, help="清理后的源人声 wav")
    ap.add_argument("--a", required=True, help="RVC 输出 A")
    ap.add_argument("--b", default=None, help="RVC 输出 B（双套参数择优时用）")
    ap.add_argument("--out", required=True, help="成品人声输出路径")
    ap.add_argument("--single", action="store_true",
                    help="单套参数：整轨直接用 A，不做切分（音色统一，杜绝接缝电音）")
    ap.add_argument("--shift", type=int, default=0, help="降调半音数（用于超界判断）")
    ap.add_argument("--overhead-hz", type=float, default=0.0,
                    help="源f0×降调倍率超过此频率的秒强制衰减（素材上限）")
    ap.add_argument("--quiet-ranges", default="",
                    help="手动退让区间，如 '240-252,300-305'（源多声部混乱段）")
    ap.add_argument("--quiet-db", type=float, default=-8.0)
    args = ap.parse_args()

    src = load(args.vocal_src)
    A = load(args.a)
    if args.b:
        B = load(args.b)
    else:
        B = A.copy()
    n = min(len(src), len(A), len(B))
    src, A, B = src[:n], A[:n], B[:n]
    secs = n // SR

    # 电平对齐
    for tag, arr in (("A", A), ("B", B)):
        g = voiced_rms(src) / voiced_rms(arr)
        print("%s 组电平对齐 %+.1f dB" % (tag, 20 * np.log10(g)))
        if tag == "A":
            A = A * g
        else:
            B = B * g

    # 逐秒打分择优
    pick, scores = [], []
    for t in range(secs):
        a, b = t * SR, (t + 1) * SR
        dbs = 20 * np.log10(np.sqrt((src[a:b] ** 2).mean()) + 1e-12)
        best = None
        for tag, arr in (("A", A), ("B", B)):
            dbo = 20 * np.log10(np.sqrt((arr[a:b] ** 2).mean()) + 1e-12)
            h, c = harm_conf(arr[a:b])
            pen = 0.0
            if dbs > -38 and dbo < dbs - 12:  # 漏唱惩罚
                pen = min(0.35, (dbs - 12 - dbo) * 0.02)
            if args.overhead_hz > 0:  # 超界惩罚
                sf0, sfconf = f0_of(src[a:b])
                if sfconf > 0.35 and sf0 > 0:
                    target = sf0 * 2 ** (-args.shift / 12)
                    if target > args.overhead_hz:
                        pen = max(pen, min(0.40, 0.18 + (target / args.overhead_hz - 1) * 0.35))
            sc = 0.75 * c + 0.25 * h - pen
            if best is None or sc > best[0]:
                best = (sc, tag)
        scores.append(best[0] if dbs > -48 else 0.0)
        pick.append(best[1] if dbs > -48 else "-")

    print("择优分布:", dict(Counter(pick)))

    y = np.zeros(n, np.float32)
    FADE = int(0.03 * SR)
    if args.single:
        y = A.copy()
        print("★ 单套参数模式：整轨直接使用 A，无拼接")
    else:
        i, segs = 0, []
        while i < secs:
            j = i
            while j < secs and pick[j] == pick[i]:
                j += 1
            segs.append((i, j, pick[i]))
            i = j
        print("段数 %d" % len(segs))
        for k, (a, b, tag) in enumerate(segs):
            i0, i1 = a * SR, min(b * SR, n)
            piece = (A if tag == "A" else B)[i0:i1].copy()
            if tag == "-":
                piece[:] = 0.0
            L = min(FADE, len(piece) // 2)
            if L > 0 and k > 0:
                piece[:L] *= np.linspace(0, 1, L, dtype=np.float32)
            y[i0:i1] = piece

    # 单套模式下源静音段平滑压到 -30dB（模型幻觉）
    if args.single:
        FLOOR = 10 ** (-30.0 / 20)
        SMOOTH = int(0.50 * SR)
        for t in range(secs):
            if pick[t] != "-":
                continue
            i0, i1 = t * SR, min((t + 1) * SR, n)
            sl = i1 - i0
            if sl < 8:
                continue
            L = min(SMOOTH, sl // 2)
            ss = smoothstep(L)
            env = np.ones(sl, np.float32)
            env[:len(ss)] = 1 - (1 - FLOOR) * ss
            env[-len(ss):] = (1 - (1 - FLOOR) * ss)[::-1]
            if sl < 2 * len(ss):
                env[:] = np.minimum(env, FLOOR * 2)
            y[i0:i1] *= env

    # 不达标段衰减（不换原唱）
    quiet = [(t, -min(9.0, 2.0 + (GOOD - scores[t]) / GOOD * 9.0))
             for t in range(secs) if 0 < scores[t] < GOOD]
    runs = []
    if quiet:
        st = prev = quiet[0][0]
        for t, _ in quiet[1:]:
            if t - prev <= 2:
                prev = t
            else:
                runs.append((st, prev + 1))
                st = prev = t
        runs.append((st, prev + 1))
    print("衰减段 %d 个 / %.1fs" % (len(runs), sum(b - a for a, b in runs)))
    for a, b in runs:
        avg = float(np.mean([g for t, g in quiet if a <= t < b]))
        i0, i1 = a * SR, min(b * SR, n)
        L = min(int(0.40 * SR), (i1 - i0) // 2)
        g = 10 ** (avg / 20)
        env = np.full(i1 - i0, g, np.float32)
        if L > 2:
            ss = smoothstep(L)
            env[:L] = 1 - (1 - g) * ss
            env[-L:] = (1 - (1 - g) * ss)[::-1]
        y[i0:i1] *= env
        print("   %6.1f~%6.1fs (%.0fs) 衰减 %.1f dB" % (a, b, b - a, avg))

    # 手动退让区间
    if args.quiet_ranges:
        gq = 10 ** (args.quiet_db / 20)
        for spec in args.quiet_ranges.split(","):
            spec = spec.strip()
            if "-" not in spec:
                continue
            a_s, b_s = spec.split("-", 1)
            t0, t1 = float(a_s), float(b_s)
            i0, i1 = int(t0 * SR), min(int(t1 * SR), n)
            if i1 - i0 < SR // 4:
                continue
            L = min(int(0.40 * SR), (i1 - i0) // 2)
            ss = smoothstep(L)
            env = np.full(i1 - i0, gq, np.float32)
            env[:len(ss)] = 1 - (1 - gq) * ss
            env[-len(ss):] = (1 - (1 - gq) * ss)[::-1]
            y[i0:i1] *= env
            print("手动退让 %.1f~%.1fs 衰减 %.1f dB" % (t0, t1, args.quiet_db))

    np.clip(y, -0.99, 0.99, out=y)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, y, SR, subtype="FLOAT")
    print("人声已写 %s（零原唱）| 峰值 %.3f" % (args.out, float(np.abs(y).max())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
