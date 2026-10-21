#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""Pitch Embedding Diff —— 从 RVC 模型权重直接读出「音域」。

原理（参考 RVC pitch embedding 的公开结构）：
  RVC 用一张 256 槽的 pitch embedding 表（emb_pitch，形状 256×hidden）来编码音高。
  每个槽对应一个 Mel 频率区间（f0 50~1100Hz 量化到 256 槽）。
  训练时只有「训练音频真的唱到了」的槽位，其向量才会相对底模(base model)发生移动。

  因此：把「训练好的模型」和「底模」的 emb_pitch 逐槽做差，看哪些槽动过，
  就是模型「训练时见过哪些音高」= 它的音域。

用法：
  python tools/read_pitch_range.py --model 模型.pth --base 底模.pth
"""
import argparse
import os
import sys

import numpy as np

F0_MIN, F0_MAX, F0_BIN = 50.0, 1100.0, 256


def _mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


MEL_MIN, MEL_MAX = _mel(F0_MIN), _mel(F0_MAX)


def bin2hz(b):
    """槽位号 -> 频率 Hz（RVC coarse_f0 的反函数）。"""
    b = np.asarray(b, dtype=float)
    m = (b - 1) * (MEL_MAX - MEL_MIN) / (F0_BIN - 2) + MEL_MIN
    return 700.0 * (np.exp(m / 1127.0) - 1.0)


def hz2bin(f):
    """频率 Hz -> 槽位号。"""
    f = np.asarray(f, dtype=float)
    m = _mel(f)
    return np.clip((m - MEL_MIN) * (F0_BIN - 2) / (MEL_MAX - MEL_MIN) + 1, 1, F0_BIN - 1)


def _load_pitch(path):
    """加载模型的 emb_pitch 权重，兼容两种 key 格式（点号平铺 / 嵌套 dict）。"""
    try:
        import torch
    except ImportError:
        raise RuntimeError("读取模型权重需要 torch：pip install torch")
    d = torch.load(path, map_location="cpu")
    w = d.get("weight", d) if isinstance(d, dict) else d
    if not isinstance(w, dict):
        raise ValueError("%s 不是有效的 RVC 权重文件" % path)
    # 格式 1：点号平铺（底模常见）
    if "enc_p.emb_pitch.weight" in w:
        return w["enc_p.emb_pitch.weight"].numpy()
    # 格式 2：嵌套 dict（inferred 模型常见）
    if "enc_p" in w and "emb_pitch" in w["enc_p"]:
        return w["enc_p"]["emb_pitch"]["weight"].numpy()
    raise KeyError("在 %s 中找不到 emb_pitch 层" % path)


def read_range(model_path, base_path, thr_frac=0.30):
    """返回 (训练槽 idx 数组, 每槽移动量数组, 阈值)。"""
    mp = _load_pitch(model_path)
    bp = _load_pitch(base_path)
    if mp.shape != bp.shape:
        raise ValueError("模型与底模的 emb_pitch 形状不一致：%s vs %s" % (mp.shape, bp.shape))

    # 每槽移动量 = 差向量的 L2 范数
    diff = np.linalg.norm(mp - bp, axis=1)
    thr = thr_frac * float(diff.max())
    trained = diff > thr
    idx = trained.nonzero()[0]
    return idx, diff, thr


def describe(idx, diff, thr):
    if idx.size == 0:
        print("未检测到任何训练过的音高槽（模型与底模几乎无差异？）")
        return None

    lo_hz = bin2hz(idx.min())
    hi_hz = bin2hz(idx.max())
    p05, p95 = bin2hz(np.percentile(idx, 5)), bin2hz(np.percentile(idx, 95))

    print("=" * 60)
    print("模型音域（Pitch Embedding Diff）")
    print("=" * 60)
    print("训练槽: %d / %d 个（阈值 = 最大移动量的 %.0f%%）" % (idx.size, F0_BIN, thr / diff.max() * 100))
    print("槽位 idx: %d ~ %d" % (idx.min(), idx.max()))
    print("对应频率: %.0f ~ %.0f Hz" % (lo_hz, hi_hz))
    print("5%%~95%% 音区: %.0f ~ %.0f Hz" % (p05, p95))
    print("中位音高: %.0f Hz" % bin2hz(np.median(idx)))
    print("-" * 60)
    print("注：这里读出的是「训练数据出现过哪些音高」，")
    print("    不等于「哪个音区唱得舒服」。用于和源音频定档交叉验证。")
    return {"idx": idx, "lo": lo_hz, "hi": hi_hz, "p05": p05, "p95": p95,
            "median": bin2hz(np.median(idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="训练好的 RVC 模型 .pth")
    ap.add_argument("--base", required=True, help="底模 .pth（fine-tune 的起点）")
    ap.add_argument("--thr", type=float, default=0.30, help="阈值 = 最大移动量的比例")
    args = ap.parse_args()

    for p in (args.model, args.base):
        if not os.path.exists(p):
            print("文件不存在: %s" % p)
            return 1

    idx, diff, thr = read_range(args.model, args.base, args.thr)
    describe(idx, diff, thr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
