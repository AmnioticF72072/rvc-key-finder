#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BS-RoFormer 分离（6GB 显存优化版）—— 基于 audio-separator 现成实现，无需重写。

为什么不用重写：audio-separator 0.47 自带完整 BS-RoFormer 推理，
只需三个参数即可在 6GB 卡上跑（实测 CMP 30HX 显存峰值 1.21GB / 6GB）：
  1. use_native_fp16=True      —— CUDA+bs_roformer 已验证支持，激活显存减半
  2. segment_size=801          —— chunk=8秒（原厂值），6GB 下显存仍宽裕
  3. overlap=2                 —— 官方 yaml 是 4，降到 2 减少一半计算量

必要的运行时补丁（audio-separator 0.47 的 regression，与精度无关）：
  _float32_frequencies 绕过 RotaryEmbedding.forward 直接调 get_seq_pos，
  而库里 default(dtype, self.cached_freqs.dtype) 会对 None 缓存提前求值，
  第一次推理必崩 AttributeError。这里给 get_seq_pos 打安全补丁。

用法：
  python sep_bsroformer.py 输入.wav [--outdir 输出目录] [--segment 801] [--overlap 2]

速度参考（CMP 30HX）：约 3.8x 实时（4分钟的歌 ~15分钟）。
日常快速分离仍推荐 5_HP-Karaoke-UVR（VR架构，约1x实时）；BS 用于追求音质的场合。
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"G:/AIVoice/models/hf_cache")
os.environ["PYTHONPATH"] = ""

# 模型目录：可通过 --model-dir 覆盖，默认从环境变量或常见路径找
MODEL_DIR = os.environ.get("BS_MODEL_DIR", r"G:/AIVoice/models/separator")
MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def patch_rotary():
    """修复 audio-separator 0.47 的 RotaryEmbedding.get_seq_pos None 缓存崩溃。"""
    import torch
    from rotary_embedding_torch.rotary_embedding_torch import RotaryEmbedding

    def _safe_get_seq_pos(self, seq_len, device=None, dtype=None, offset=0):
        if dtype is None:
            dtype = self.cached_freqs.dtype if self.cached_freqs is not None else torch.float32
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    RotaryEmbedding.get_seq_pos = _safe_get_seq_pos


def run(src, outdir, segment=801, overlap=2, fp16=True):
    import torch
    from audio_separator.separator import Separator

    patch_rotary()
    os.makedirs(outdir, exist_ok=True)

    sep = Separator(
        log_level=20,
        model_file_dir=MODEL_DIR,
        output_dir=outdir,
        output_format="WAV",
        use_native_fp16=fp16,
        mdxc_params={"segment_size": segment, "override_model_segment_size": True, "overlap": overlap},
    )
    sep.load_model(model_filename=MODEL_NAME)

    t0 = time.time()
    out = sep.separate(src)
    dt = time.time() - t0
    print("分离完成: %.1fs" % dt)
    for f in out:
        print("  输出:", f)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("显存峰值: %.2f GB / %.1f GB" % (peak, total))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="输入音频（wav/flac/mp3）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认：输入文件旁边 _bs_out）")
    ap.add_argument("--segment", type=int, default=801, help="chunk 段长（801=8秒；显存紧就降到 512）")
    ap.add_argument("--overlap", type=int, default=2, help="重叠窗口数（默认2；要更高边缘质量用4）")
    ap.add_argument("--model-dir", default=None, help="模型目录（默认 BS_MODEL_DIR 或内置路径）")
    ap.add_argument("--no-fp16", action="store_true", help="禁用 fp16（仅调试用）")
    args = ap.parse_args()

    global MODEL_DIR
    if args.model_dir:
        MODEL_DIR = args.model_dir

    if not os.path.exists(args.input):
        print("输入文件不存在:", args.input)
        return 1

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "_bs_out")

    try:
        run(args.input, outdir, args.segment, args.overlap, not args.no_fp16)
    except Exception as e:
        # OOM 自动降级：降到 5 秒 chunk 重试一次
        if "out of memory" in str(e).lower():
            print("!! 显存不足，自动降级到 segment=512 重试...")
            run(args.input, outdir, 512, args.overlap, not args.no_fp16)
        else:
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
