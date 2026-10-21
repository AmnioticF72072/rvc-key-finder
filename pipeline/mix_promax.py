#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""Pro Max 人声后期链 —— 目标是"咬字清楚 + 有空间感"。

每条都对应社区专业做法 + 实测结论：
  1. 混响返回先高通 300Hz —— 低频混响是"糊"的头号主因
  2. 预延迟 24ms → 35ms —— 让清辅音先出来再进混响
  3. 湿量 16% → 11% —— 干信号占主导（干 70-80%）
  4. 并行压缩：干净支 + 8:1 压扁支按 28% 混合，提密度不糊辅音
  5. 串行压缩：第一级 3.2:1 起音 12ms（让辅音瞬态通过）+ 第二级 2:1 慢速胶水
  6. de-esser 放压缩之后（压缩后齿音更明显，此时处理更透明）
  7. 母带用【测量后线性平移】而不是 loudnorm 动态模式

用法：
  python tools/mix_promax.py --vocal 成品人声.wav --acc 伴奏.wav \
      --name "歌名 翻唱" --vocal-db 3.5 --wet 0.11 --outdir ./成品
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

try:
    import soxr
except ImportError:
    soxr = None

SR = 48000


def find_ffmpeg():
    p = os.environ.get("FFMPEG")
    if p and os.path.exists(p):
        return p
    for cand in ("ffmpeg", "ffmpeg.exe"):
        import shutil
        if shutil.which(cand):
            return cand
    raise RuntimeError("未找到 ffmpeg，请安装并加入 PATH，或设置环境变量 FFMPEG")


FF = find_ffmpeg()


def load(path, sr_t=SR, mono=True):
    x, sr = sf.read(path, dtype="float32")
    if mono and x.ndim > 1:
        x = x.mean(1)
    x = np.ascontiguousarray(x, np.float32)
    if sr != sr_t:
        if soxr is None:
            raise RuntimeError("需要 soxr: pip install soxr")
        x = soxr.resample(x, sr, sr_t, quality="VHQ")
    return x


def run(cmd, want_stderr=False):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("  !! ", (r.stderr or "")[-300:])
    return (r.stderr or "") if want_stderr else (r.stdout or "")


def stats(path):
    pk, ss, nn = 0.0, 0.0, 0
    with sf.SoundFile(path) as f:
        for blk in f.blocks(blocksize=1 << 16, dtype="float32", always_2d=True):
            pk = max(pk, float(np.max(np.abs(blk))))
            ss += float(np.sum(blk ** 2))
            nn += blk.size
    return pk, np.sqrt(ss / max(nn, 1))


def make_ir(dur, sr=SR, seed=11):
    """合成板式混响冲激响应（随机噪声 + 稀疏早期反射 + 一阶反馈延迟网络）。"""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / dur)
    for dly, g in [(9, 0.5), (15, 0.38), (22, 0.3), (31, 0.22), (41, 0.15)]:
        i = int(dly / 1000 * sr)
        if i < n:
            ir[i] += g
    a = 0.70
    for i in range(1, n):
        ir[i] = a * ir[i] + (1 - a) * ir[i - 1]
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    return ir


def conv(x, h):
    nf = 1 << int(np.ceil(np.log2(len(x) + len(h) - 1)))
    y = np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(h, nf), nf)
    return y[:len(x)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True, help="零原唱收尾后的人声 wav")
    ap.add_argument("--acc", required=True, help="伴奏 wav（降同调后）")
    ap.add_argument("--name", required=True, help="导出文件名（不含扩展名）")
    ap.add_argument("--outdir", default="./成品", help="导出目录")
    ap.add_argument("--vocal-db", type=float, default=3.5)
    ap.add_argument("--wet", type=float, default=0.11)
    ap.add_argument("--rev-time", type=float, default=1.5)
    ap.add_argument("--pre-delay", type=float, default=0.035)
    ap.add_argument("--rev-hpf", type=float, default=300.0, help="混响返回高通 Hz")
    ap.add_argument("--parallel", type=float, default=0.28, help="并行压缩混入比例")
    ap.add_argument("--eq", default="", help="可选：外部 EQ 链文件（firequalizer 串）")
    ap.add_argument("--keep-temp", action="store_true", help="保留中间 wav 文件")
    args = ap.parse_args()

    TMP = os.path.join(os.path.dirname(os.path.abspath(args.outdir)), "_pm_tmp")
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    V = args.vocal
    eqpart = ""
    if args.eq and os.path.exists(args.eq):
        eqpart = open(args.eq, encoding="utf-8").read().strip() + ","

    correct = (eqpart +
               "equalizer=f=280:width_type=o:width=1.0:g=-2.4,"   # 去 mud
               "equalizer=f=620:width_type=o:width=1.3:g=-1.5,"   # 去 boxy/nasal
               "equalizer=f=4200:width_type=o:width=1.1:g=1.6,"   # 咬字核心区
               "equalizer=f=9500:width_type=o:width=1.2:g=1.0")    # air

    clean = os.path.join(TMP, "clean.wav")
    run([FF, "-y", "-v", "error", "-i", V, "-af", correct,
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", clean])

    comp = os.path.join(TMP, "comp.wav")
    run([FF, "-y", "-v", "error", "-i", clean,
         "-af", ("acompressor=threshold=-20dB:ratio=3.2:attack=12:release=140:makeup=1.15:knee=2.5:detection=rms:link=average,"
                 "acompressor=threshold=-16dB:ratio=2.0:attack=45:release=320:makeup=1.0:knee=4"),
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", comp])

    ds = os.path.join(TMP, "ds.wav")
    run([FF, "-y", "-v", "error", "-i", comp,
         "-af", "equalizer=f=7000:width_type=o:width=2.6:g=-1.8",
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", ds])

    par = os.path.join(TMP, "par.wav")
    run([FF, "-y", "-v", "error", "-i", clean,
         "-af", "acompressor=threshold=-30dB:ratio=8:attack=2:release=90:makeup=2.2:knee=2",
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", par])

    main = load(ds)
    pcomp = load(par)
    n = min(len(main), len(pcomp))
    main, pcomp = main[:n], pcomp[:n]
    r1 = np.sqrt((main ** 2).mean()) + 1e-9
    r2 = np.sqrt((pcomp ** 2).mean()) + 1e-9
    pcomp = pcomp * (r1 / r2)
    base = (1 - args.parallel) * main + args.parallel * pcomp
    print("并行压缩：干净支 %.0f%% + 压扁支 %.0f%%" % ((1 - args.parallel) * 100, args.parallel * 100))

    # 板式混响（返回高通）
    ir = make_ir(args.rev_time)
    pre = int(args.pre_delay * SR)
    wet = np.zeros(len(base), np.float32)
    if pre < len(base):
        wet[pre:] = conv(base[:len(base) - pre], ir)
    wet /= (np.max(np.abs(wet)) + 1e-9)

    wn = os.path.join(TMP, "wet.wav")
    sf.write(wn, wet, SR, subtype="FLOAT")
    weth = os.path.join(TMP, "wet_hp.wav")
    run([FF, "-y", "-v", "error", "-i", wn, "-af", "highpass=f=%d:poles=2" % int(args.rev_hpf),
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", weth])
    wet = load(weth)[:len(base)]
    wet *= (np.max(np.abs(base)) + 1e-9) / (np.max(np.abs(wet)) + 1e-9)

    st = np.stack([base, base], 1)
    wetst = np.stack([np.roll(wet, int(0.006 * SR)) * 0.8, wet], 1)
    voc = (1 - args.wet) * st + args.wet * wetst
    voc -= voc.mean(0, keepdims=True)
    pk = float(np.max(np.abs(voc)))
    if pk > 0.985:
        voc *= 0.985 / pk
    vpath = os.path.join(TMP, "vocal.wav")
    sf.write(vpath, voc, SR, subtype="FLOAT")
    print("混音：湿量 %.0f%% / 预延迟 %.0fms / 混响返回高通 %.0fHz / 衰减 %.1fs"
          % (args.wet * 100, args.pre_delay * 1000, args.rev_hpf, args.rev_time))

    # 合流
    ACC = args.acc
    AAF = ("afftdn=nf=-42:nr=12:tn=1,"
           "equalizer=f=2600:width_type=o:width=1.5:g=-2.2,"   # 给咬字让出频段
           "lowpass=f=16500:poles=2")

    def br(s, af, tag):
        t = os.path.join(TMP, "b_%s.wav" % tag)
        run([FF, "-y", "-v", "error", "-i", s, "-af", af, "-ar", "48000", "-ac", "2",
             "-c:a", "pcm_f32le", t])
        r = stats(t)[1] if os.path.exists(t) else 1e-9
        if os.path.exists(t):
            os.remove(t)
        return r

    rv, ra = br(vpath, "anull", "v"), br(ACC, AAF, "a")
    gv = 10 ** (args.vocal_db / 20) * ra / max(rv, 1e-9)
    print("滤波后 人声 %.4f / 伴奏 %.4f -> x%.3f (+%.1f dB)" % (rv, ra, gv, args.vocal_db))

    mix = os.path.join(TMP, "mix.wav")
    af = ("[0:a]volume=%.5f[v];[1:a]%s[ia];"
          "[v][ia]amix=inputs=2:duration=shortest:dropout_transition=0,"
          "lowshelf=f=110:g=1.0,highshelf=f=10800:g=0.6[o]") % (gv, AAF)
    run([FF, "-y", "-v", "error", "-i", vpath, "-i", ACC,
         "-filter_complex", af, "-map", "[o]", "-ar", "48000", "-ac", "2",
         "-c:a", "pcm_f32le", mix])

    # 母带：轻压缩 → 测 LUFS → 线性平移
    if os.path.exists(mix):
        compm = os.path.join(TMP, "master.wav")
        run([FF, "-y", "-v", "error", "-i", mix,
             "-af", "acompressor=threshold=-15dB:ratio=1.5:attack=35:release=300:makeup=1.0:knee=5",
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", compm])
        info = run([FF, "-hide_banner", "-nostats", "-i", compm,
                    "-af", "loudnorm=I=-11:TP=-1.0:print_format=json", "-f", "null", "-"],
                   want_stderr=True)
        m = re.search(r'"input_i"\s*:\s*"(-?[\d.]+)"', info or "")
        cur = float(m.group(1)) if m else None
        if cur is None:
            _, r0 = stats(compm)
            cur = 20 * np.log10(r0 + 1e-9) - 2.0
            print("  (LUFS 测量失败，用 RMS 估算 %.1f)" % cur)
        gain = max(-18.0, min(18.0, -10.8 - cur))
        print("母带：测得 %.2f LUFS -> 线性 %+.1f dB" % (cur, gain))
        master = "volume=%.3fdB,alimiter=limit=0.891:attack=4:release=80:level=disabled" % gain
        for ext, enc in [("mp3", ["-c:a", "libmp3lame", "-b:a", "320k"]),
                         ("flac", ["-c:a", "flac", "-compression_level", "8"])]:
            dst = os.path.join(args.outdir, "%s.%s" % (args.name, ext))
            run([FF, "-y", "-v", "error", "-i", compm, "-af", master,
                 "-ar", "48000", "-ac", "2"] + enc + [dst])
            if os.path.exists(dst):
                print("导出 %s (%.1f MB)" % (dst, os.path.getsize(dst) / 1048576))

    if not args.keep_temp:
        import shutil
        shutil.rmtree(TMP, ignore_errors=True)
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
