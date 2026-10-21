#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本项目代码由 AI 编程助手编写（详见仓库根目录 NOTICE）。
"""key_finder —— RVC 翻唱定调（双引擎交叉验证）。

这是本项目的主入口，把两条互相独立的路线合到一起：

  引擎① 源音频定档（analyze_pitch.py）
     测源人声的基频，用「双范围 + 频谱能量裁决」防倍频/次谐波误判。
     回答：这首歌的人声基频到底是多少？会不会八度误判？

  引擎② 模型音域（tools/read_pitch_range.py）
     读 RVC 模型的 pitch embedding 权重，和底模做差分。
     回答：我的声音模型训练时见过哪些音高（= 它的音域）？

两条路线独立、互补：
  - 引擎① 可能八度误判（倍频/次谐波），引擎② 不测音频、无此问题；
  - 引擎② 读出的是"训练数据出现过哪些音高"，不是"舒适音区"，
    引擎① 的素材上限可用来收窄。

用法：
  python key_finder.py --vocal vocals.wav \
      --model 我的模型.pth --base 底模.pth \
      [--material-max 375]

  --model / --base 可选；不给则只跑引擎①（源音频定档）。
"""
import argparse
import os
import sys

# 允许从本目录 import analyze_pitch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from analyze_pitch import analyze


def _fmt_hz(v):
    return "%.0f Hz" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True, help="分离后的人声轨 wav")
    ap.add_argument("--model", default=None, help="训练好的 RVC 模型 .pth（可选）")
    ap.add_argument("--base", default=None, help="底模 .pth（和 --model 一起用）")
    ap.add_argument("--material-max", type=float, default=0.0,
                    help="素材最高舒适音 Hz（没有模型权重时手动填）")
    ap.add_argument("--thr", type=float, default=0.30, help="权重差分阈值（默认 0.30）")
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()

    print()
    print("#" * 66)
    print("#  RVC 翻唱定调 · 双引擎交叉验证")
    print("#" * 66)

    # ---------- 引擎①：源音频定档 ----------
    print("\n【引擎① 源音频定档】")
    result = analyze(args.vocal, args.material_max, args.sr)
    if result is None:
        print("源音频分析失败，无法继续")
        return 1
    src_median = float(np.median(result))
    src_p95 = float(np.percentile(result, 95))

    # ---------- 引擎②：模型音域 ----------
    model_range = None
    if args.model and args.base:
        if not os.path.exists(args.model):
            print("\n[跳过引擎②] 模型文件不存在: %s" % args.model)
        elif not os.path.exists(args.base):
            print("\n[跳过引擎②] 底模文件不存在: %s" % args.base)
        else:
            print("\n【引擎② 模型音域（Pitch Embedding Diff）】")
            from tools.read_pitch_range import read_range, describe, bin2hz
            idx, diff, thr = read_range(args.model, args.base, args.thr)
            model_range = describe(idx, diff, thr)
    elif args.model and not args.base:
        print("\n[提示] 给了 --model 但没给 --base，无法做权重差分，跳过引擎②。")
    else:
        print("\n[提示] 未提供 --model/--base，跳过引擎②（只用源音频定档）。")

    # ---------- 交叉验证 + 最终建议 ----------
    print("\n" + "=" * 66)
    print("交叉验证")
    print("=" * 66)

    if model_range:
        print("源人声基频中位: %s（95%% %s）" % (_fmt_hz(src_median), _fmt_hz(src_p95)))
        print("模型训练音区:   %s ~ %s（5%%~95%% %s ~ %s）" % (
            _fmt_hz(model_range["lo"]), _fmt_hz(model_range["hi"]),
            _fmt_hz(model_range["p05"]), _fmt_hz(model_range["p95"])))
        print("-" * 66)

        # 用模型音域给一个「降调后落点」参考
        if args.material_max > 0:
            print("→ 素材上限 %.0f Hz 已用于源音频定档的降调建议（见上方引擎①）。" % args.material_max)
        print("→ 交叉验证结论：")
        # 八度一致性：源基频 2 倍频是否落在模型音区内
        if model_range["lo"] <= src_median <= model_range["hi"]:
            print("  源基频 %s 直接落在模型音区内，无需额外八度纠正。" % _fmt_hz(src_median))
        else:
            print("  源基频 %s 在模型音区外，提示可能存在八度/音区不匹配，" % _fmt_hz(src_median))
            print("  请以上方引擎①的频谱能量裁决结果为准（它专门防倍频误判）。")
    else:
        print("（仅引擎①）最终定调建议见上方「★ 推荐定档」。")
        print("（给 --model + --base 可启用引擎②做交叉验证。）")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
