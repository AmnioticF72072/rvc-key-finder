# RECIPES —— 参数配方表

直接抄作业。所有数值都来自本项目实战验证。

## RVC 推理参数（唱歌）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| protect | **0.10** | 0.0=最大保护清辅音，0.5=不保护。官方默认 0.33 是语音折中，唱歌会糊 |
| volume_envelope | **0.4** | 0.4=主要跟源。1.0=完全听模型的，句间会自我衰减（忽大忽小） |
| index_rate | **0.50** | 0.4–0.6 黄金区间。<0.3 原声残留，>0.7 金属感 |
| f0_method | **rmvpe** | 默认。素材周期性低时可试 crepe（jitter 8.4→7.6 音分），但先解决和声 |
| filter_radius | 0.03 | pipeline 默认 |

## 定档（降调量）

- **降调是全局决策，错一次全曲报废**，值得多花两分钟验证。
- 判据：降调后**副歌 70 分位落在 250–320Hz**、全曲 95 分位 ≤ 素材上限。
- 降调量取**偶数半音**（-2/-4/-6/-8/-10/-12）。
- 倍频/次谐波误判：用 `analyze_pitch.py` 的双范围 + 频谱能量裁决，别裸用自相关峰值。

## 源准备

- **和声是最常见的坑**：副歌段"唱得乱"，多数是和声叠层混进了人声轨，不是参数问题。
  用 UVR Karaoke 模型（`5_HP-Karaoke-UVR.pth`）剥和声。
- 低配显卡（6GB）：BS-RoFormer(639MB) 能跑（见 PITFALLS 5.3），VR 架构（5_HP-Karaoke 127MB）更快。

## 零原唱收尾

| 参数 | 值 | 说明 |
|---|---|---|
| 达标线 GOOD | 0.46 | 只有音高确实不明确才衰减 |
| 静音处理 | 沉到 -30dB | 绝不硬切 0（会产生咔哒/宽带噪声） |
| 过渡 | 500ms smoothstep | 每 50ms 变化 <4dB |
| 衰减段过渡 | 400ms smoothstep | 同上 |

## Pro Max 后期链

| 环节 | 参数 | 实测依据 |
|---|---|---|
| 修正 EQ | 280Hz -2.4 / 620Hz -1.5 / 4200Hz +1.6 / 9500Hz +1.0 | 去 mud/boxy，提咬字区+air |
| 串行压缩 | 3.2:1 起音12ms → 2:1 起音45ms | 让辅音瞬态通过 + 慢速胶水 |
| 并行压缩 | 8:1 快起音支路 28% | 提密度不糊辅音 |
| de-esser | 7kHz 宽Q -1.8dB，放压缩后 | 压缩后齿音更明显，此时处理更透明 |
| 混响 | 板式，返回高通 300Hz，预延迟 35ms，湿量 11%，衰减 1.5s | 低频混响是"糊"的主因 |
| 母带 | 测 LUFS → 线性平移，不用 loudnorm 动态模式 | loudnorm 单遍是动态模式，前奏会被抬 +28dB |

## 全流程

```bash
# 1. 解密（可选，酷狗加密源）
python tools/decrypt_kgm.py "源.kgma" "源.flac"

# 2. 分离 + 去和声（audio-separator + Karaoke 模型）
#    用你习惯的 UVR 工具，或 audio-separator 命令行

# 3. 定档
python pipeline/analyze_pitch.py --vocal vocals.wav --material-max 375

# 4. 谐波门控清理（可选，f0 乱跳时）
python tools/clean_vocal.py --src vocals.wav --out vocals_clean.wav

# 5. RVC 转换（用 Applio 等，参数按上表）
#    protect 0.10 / volume_envelope 0.4 / index_rate 0.5

# 6. 零原唱收尾
python pipeline/finish.py --vocal-src vocals_clean.wav \
    --a rvc_out.wav --single --out 成品人声.wav --shift -8 --overhead-hz 375

# 7. 异常体检
python tools/find_glitch.py --src vocals_clean.wav --voc 成品人声.wav

# 8. Pro Max 后期
python pipeline/mix_promax.py --vocal 成品人声.wav --acc 伴奏_降8.wav \
    --name "歌名 翻唱" --outdir ./成品
```

## 声音克隆（训练侧）红线

- **禁止**用"喉咙硬挤压吼出来"的素材训练，不要靠硬吼扩音域。
- 新增素材只取**舒适音区**，宁可音域窄也不要畸变极限音。
- 弃用检查点只备份，绝不删除；别无脑选最低轮数（可能欠拟合发虚），统一参数盲对比。
