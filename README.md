# keepsharp

**要更锐，但必须留住原图。**

keepsharp 对 4K 低质量实拍做分辨率不变的 **Diffusion 可控增强**：在提高锐度的同时，主体、文字、人脸、钟表指针保持与原图一致，结果尽量接近 Ground Truth。它是 2026 年 CSIG / 华为「Camera学术之星」[赛题一](https://tianchi.aliyun.com/competition/entrance/532499/information) 的实现。

| | |
| --- | --- |
| 作品 / 包名 | `keepsharp` |
| 测量平台 | Windows · RTX 4060 Laptop 8GB · Python 3.12 · `torch 2.6.0+cu124` |
| 当前阶段 | 方案筛查完成；Lite 基线与 OSEDiff 均已在官方验证集出分 |
| LQ → GT | PSNR **28.034** / SSIM **0.778** |
| Lite + fuse | **27.982 / 0.775** · 0.31 GB · 约 22.5 s/张 |
| OSEDiff + fuse | **28.030 / 0.781** · 3.03 GB · 约 27.6 s/张 |

完整调研、逐图消融与机制分析见 [docs/handoff_phase1.md](docs/handoff_phase1.md)。

---

## 1. 问题

移动端生成式增强常见失败模式是：图变「清晰」了，但脸换了、字写错了、指针歪了、叶子变成油画。本题要求模型必须是 Diffusion，并在五类元素上同时做到清晰与内容一致：

- 小人脸
- 文字（印刷体、书脊小字）
- 密集绿植
- 钟表
- 鸟类

输入输出分辨率相同（官方图为 \(4096\times3072\) 或 \(3072\times4096\)，约 12.58 MP），不是 \(4\times\) 超分。本地代理指标为 PSNR / SSIM；官方总分公式入围后公布。

数据：

| 集合 | 数量 | 说明 |
| --- | --- | --- |
| 验证集 | 5 对 LQ + GT | 本地评测 |
| 初赛测试集 | 100 张 LQ | GT 不公开 |
| 决赛 | 现场 20 张 | 评委 + 现场测图 |

**keepsharp** 对应这条双目标：Sharp 是锐度，Keep 是不漂离原图。第一阶段的模型筛选和后处理都按这个取舍做。

---

## 2. 已完成的工作

1. 接通官方验证集 / 测试集（本机放置，不在本仓库）；`case*_lq.jpg` / `case*_gt.jpg` 自动配对，输出名为 `caseN.jpg`。
2. 按 4K × 8 GB、是否从 LQ 起步、文字/几何幻觉风险，筛选开源 Diffusion 方案。
3. 实现并冻住 **Lite 条件 DDPM + 残差限幅融合**。
4. 接入 **OSEDiff**（官方 LoRA + SD 2.1-base，`upscale=1`），在同一 5 对上对比裸输出与 +fuse。
5. 完成 Lite 后处理消融（fuse 强度、LAB、高通结构锁）。

裸 Lite（27.65 / 0.742）和裸 OSEDiff（23.22 / 0.740）都劣于直接拷贝 LQ，已不作为质量参照。对照列是原始 LQ，以及两条带 fuse 的端到端结果。验证集 \(n=5\)，没有用这 5 对 GT 微调后再报同一集合。

---

## 3. 方法

```text
LQ
  → 场景路由（验证集为人工标注）
  → 场景控制器（给出 strength / prompt）
  → Diffusion 骨干：LiteCondUNet 或 OSEDiff
  → 4K 分块（512，重叠 64，Hann 融合；横图 63 tile）
  → 锁定输出尺寸 = 输入尺寸
  → fuse_fidelity：y = LQ + 0.25 · clip(Pred − LQ, −12, 12)
  → 写出 caseN.jpg
```

**Lite** 是小容量条件 UNet：`concat(x_t, LQ)` 预测噪声，\(T=50\)，DDIM 12 步。`strength` 是加噪起点占 \(T\) 的比例，越大越接近从噪声生成。

**OSEDiff** 把 LQ 编码到 SD 2.1 潜空间，在 \(t=999\) 做一步 \(x_0\) 预测。本仓库用场景控制器的 prompt，不加载 RAM / DAPE。

**保真融合** 把每通道每像素相对 LQ 的改动限制在约 3 个灰阶。LAB 色彩对齐与高通结构锁在官方验证集上降低 PSNR，默认关闭。

骨干实现 `BaseEnhancer.enhance(image, context)`，由 `src/models/factory.py` 注册为 `lite` 或 `osediff`。分块与融合在 `src/inference/`。

冻结超参见 `configs/inference.yaml`：

```text
tile_size=512  overlap=64
lite: timesteps=50  sample_steps=12  base_channels=32
fidelity_fuse=true  fuse_mix=0.25  fuse_max_delta=12
color_match=false  lock_content=false
strength: text 0.18 / clock 0.20 / face 0.22 / general 0.25 / bird 0.28 / plant 0.30
```

---

## 4. 验证集与结果

官方图像不在本仓库。本机按 `configs/default.yaml` 放置后，文件名约定为：

```text
验证集：case{1–5}_lq.jpg 与 case{1–5}_gt.jpg
测试集：case{1–100}.jpg
```

| ID | 画面 | LQ PSNR / SSIM | 角色 |
| --- | --- | --- | --- |
| case1 | 印刷中文试卷 | 32.03 / 0.948 | 文字一致性 |
| case2 | 中英文书脊 | 28.19 / 0.831 | 小字 |
| case3 | 滩涂海鸟 | 35.62 / 0.935 | 已接近 GT，过增强敏感 |
| case4 | 密叶与花 | **18.06 / 0.309** | 误差预算主体 |
| case5 | 建筑罗马钟 | 26.28 / 0.864 | 指针与数字几何 |

验证集没有小人脸。`inference.yaml` 里 `case1`…`case5` 的场景标注只对这 5 张验证图有效；测试集同名文件是另一组图像。

| 设定 | PSNR | SSIM | 秒/张 | 显存 |
| --- | --- | --- | --- | --- |
| LQ 原图 | 28.034 | 0.778 | — | 0 |
| 裸 Lite | 27.649 | 0.742 | 28.7 | 0.31 GB |
| **Lite + fuse** | **27.982** | **0.775** | **22.5** | **0.31 GB** |
| 裸 OSEDiff | 23.216 | 0.740 | 28.9 | 3.03 GB |
| **OSEDiff + fuse** | **28.030** | **0.781** | **27.6** | **3.03 GB** |

OSEDiff + fuse 相对 LQ：均值 −0.005 dB；case2 +0.61 dB，case5 +0.35 dB，case1 −0.47 dB，case4 约 0。裸 OSEDiff 在 case1 / case3 上分别掉 8.1 / 9.5 dB。仓库内数字：`data/outputs/eval/eval_report.json`、`data/outputs/phase1/osediff_val.json`。推理拼接图不进仓库。

---

## 5. 方案对照

| 方案 | 本阶段结论 | 依据 |
| --- | --- | --- |
| OSEDiff + fuse | 已出分的开源骨干 | 一步、从 LQ 出发；fuse 后 28.03 / 0.781 |
| Lite + fuse | 可运行保真基线 | Diffusion、0.31 GB、不劣于 LQ |
| 裸 OSEDiff / 裸 Lite | 弃用 | 分别 −4.82 dB / −0.39 dB |
| SeeSR | 不作为第一基线 | ~50 步；语义标签改写文字/指针 |
| DiffBIR | 不作为第一基线 | 640² 约 11 GB |
| SUPIR | 不作为第一基线 | SDXL 级，8 GB 不适合 |
| SD-Turbo | 未出分 | 权重未装完；非复原模型 |

OSEDiff：Wu et al., NeurIPS 2024, [代码](https://github.com/cswry/OSEDiff)。权重文件不在本仓库，本机下载见 [weights/README.md](weights/README.md)。

---

## 6. 复现

官方图与权重需在本机准备（见 `configs/default.yaml`、[weights/README.md](weights/README.md)），二者均不在本仓库。

```powershell
python -m pip install -e ".[dev]"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pytest -q
python scripts/infer.py --split eval --backend lite
python scripts/evaluate.py
python scripts/run_osediff_val.py
```

| 脚本 | 作用 |
| --- | --- |
| `scripts/infer.py` | `--split eval\|test`，`--backend lite\|osediff` |
| `scripts/evaluate.py` | 验证集 PSNR / SSIM |
| `scripts/run_osediff_val.py` | OSEDiff 裸输出与 +fuse |
| `scripts/benchmark_val.py` | 多后端对照 |
| `scripts/ablate_fidelity.py` | fuse / LAB / 高通消融 |
| `scripts/train_lite.py` | 训练 Lite 权重 |

---

## 7. 目录

```text
keepsharp
├── README.md
├── configs/                  default.yaml、inference.yaml
├── docs/competition.md       赛题摘要
├── docs/handoff_phase1.md    调研报告
├── src/models/lite/          Lite 条件 DDPM
├── src/models/osediff.py     OSEDiff 一步推理
├── src/inference/            分块与 fuse_fidelity
├── scripts/                  推理 / 评测 / 消融 / 本机下载
├── tests/
├── data/outputs/eval/        eval_report.json
├── data/outputs/phase1/      benchmark / ablation / osediff_val
└── weights/README.md         本机权重下载（权重文件不在仓库）
```

---

## 8. 尚未解决

- 裸 OSEDiff 改写文字并过增强海鸟，质量上必须带 fuse。
- case4 绿植在 Lite 与 OSEDiff+fuse 上都停在约 18.06 dB。
- SeeSR / DiffBIR / SUPIR / SD-Turbo 没有本机验证集分数。
- 测试集 100 张还没有独立的场景标注。
