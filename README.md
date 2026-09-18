# MMS 评测流水线（Mode Mixture Score Pipeline）

本仓库是论文 *MMS: Adaptive Evaluation for Mode Mixture in Generated Images* 的纯代码部分：一套**真实数据锚定、自适应阈值**的生成样本语义评测流水线。它接收任意生成模型的输出（图像 / 文本 / 频谱图），用冻结的分类器计算后验熵，用独立真实数据校准阈值，输出候选比例（MMS）及完整的逐样本诊断。

一句话原理：**分类器对一张图有多"犹豫"（熵），超过真实图正常犹豫度（校准阈值）的算一张候选，候选占比就是 MMS。** 换数据集、换分类器、换温度时阈值自动重校准，规则固定、不针对任何生成模型调参。

> 本仓库只含代码、配置与测试；图片、权重、冻结参考、实验报告等大资产保存在各实验服务器，不在 Git 内。

## 目录结构

```
mms_eval/          通用图像评测包（AFHQ、CIFAR-10 等走这里）
  semantic.py        核心算法：softmax/熵/秩校准/候选聚合（论文算法 1）
  pipeline.py        prepare→train→reference→evaluate 编排、冻结参考
  cli.py             命令行入口 python -m mms_eval（约 30 个子命令）
  comparison.py      固定 N 抽样比较、evaluate-batch、compare
  controls.py 等     受控混入/退化对照、人工标注、检测统计等研究工具
mms_multimodal/    跨模态包：共享评分核心 + 各数据域适配器
  core.py            跨模态冻结参考与评测契约（image/text/audio 通用）
  mnist28.py(.py_pipeline)  MNIST 28×28 原生灰度适配器
  crc100k.py         CRC-HE 组织病理 9 类适配器
  esc50.py           ESC-50 音频（256×256 梅尔频谱图）适配器
  agnews.py          AG News 文本（TF-IDF + 逻辑回归，纯 CPU）适配器
  intervals.py       Wilson + Hoeffding 双置信区间（后处理，不进冻结链）
  support.py         新适配器的共享工具函数
scripts/           各域一键入口（run_*_mms.py）与分析脚本
configs/           域配置（类别、划分、种子、图像策略）与评估器训练配置
tests/             117 项测试，覆盖算法数学、防篡改、各域适配器端到端
```

## 安装

```sh
python -m pip install -e '.[paper,test]'   # 完整（含 torch、clean-fid、torch-fidelity）
python -m pip install -e '.[test]'         # 纯 CPU：语义指标 + AGNews 全流程可跑
python -m mms_eval --help
pytest tests/ -q
```

Python 3.10–3.12。已验收组合：PyTorch 2.5.1 / torchvision 0.20.1、clean-fid 0.1.35、torch-fidelity 0.3.0。首次运行质量指标会自动下载官方后端权重（需联网）。

## 核心概念

| 概念 | 是什么 | 何时重做 |
|---|---|---|
| 语义评估器 | 用真实数据训练的分类器（如 ResNet-50），冻结后打分 | 每个数据域建一次 |
| 真实数据划分 | fit（训练）/ validation（选超参）/ calibration（定阈值）/ audit（检查），零重叠 | 每个数据域建一次 |
| 冻结参考 `reference.json` | 记录类别、阈值、评估器哈希、校准样本身份、评分代码哈希 | 同域同协议复用 |
| MMS 候选率 | 生成样本熵 > 阈值的比例 | 每批生成样本算一次 |
| 逐样本分数 | `scores.jsonl`：每样本的概率、熵、5 种分数、候选标记 | 每次评测输出 |

**候选率是"分类器异常不确定"的筛查比例，不等于人工确认的语义混淆率。** 高熵也可能来自模糊、噪声或域外内容；论文用人工标注和负对照实验验证语义有效性。

## 算法与论文的对应（算法 1）

| 论文公式/步骤 | 代码 | 输出字段 |
|---|---|---|
| 式(2) 带温度 softmax | `mms_eval/semantic.py::posterior_scores` | `scores.jsonl: probabilities` |
| 式(3) 自然对数熵 | 同上 | `scores.jsonl: entropy` |
| 式(4)(5) 秩校准阈值 τ（k=⌈(n+1)(1−α)⌉，k>n 时 τ=∞） | `semantic.py::rank_calibrate` | `report.json: calibrations.entropy.threshold` |
| 式(7) 严格大于判候选 d_i | `semantic.py::apply_threshold` | `scores.jsonl: flag_entropy` |
| 式(9) MMŜ = 候选数/N | `semantic.py::aggregate_mms` | `report.json: mms.value / candidate_count` |
| 置信区间（Wilson） | `semantic.py::wilson_interval` | `report.json: mms.ci95` |
| 置信区间（Hoeffding，式(10)） | `mms_multimodal/intervals.py` + `scripts/report_intervals.py` | `intervals.json: hoeffding_interval` |
| 式(12) 类别对矩阵 M_ab，总和=MMS | `aggregate_mms`（带断言） | `mms.pair_rates_upper_triangle` |

除主熵分数外，`aggregate_mms` 同时对 4 个基线分数（第二大概率、1−max、top1−top2 间隔、Gini）做同校准阈值统计，对应论文 4.3 节的同校准基线对照。

### 两个置信区间的区别

- **Wilson（`mms.ci95`）**：二项比例的标准得分区间。在"样本独立抽取"假设下较紧，是报告默认区间。
- **Hoeffding（`intervals.json`）**：论文式(10) 的分布式无关界，ε = √(ln(2/δ)/(2N))，对任意分布的独立样本都成立，因此更宽、更保守。

两者描述同一个点估计（候选率）在不同假设下的抽样不确定性，都**不是**语义准确性的保证。生成命令：

```sh
# 对任意已保存的 report.json（两条流水线通吃）打印并落盘 intervals.json
PYTHONPATH=. python scripts/report_intervals.py /path/to/evaluation_dir
# 加 --no-write 只看不下盘；--confidence 0.95 可调
```

## 不同模态、不同尺寸的数据怎么进流水线

| 模态 | 入口 | 输入要求 | 分类器 |
|---|---|---|---|
| 图像（任意尺寸） | `python -m mms_eval evaluate` | 单文件 / 目录 / CSV / JSONL 清单 | ResNet-50 / ViT-B/16 / EfficientNet-B0 |
| 图像（MNIST 特例） | `scripts/run_mnist28_mms.py` | 原生 28×28 灰度 PNG + JSONL 清单 | 专用 4 层 CNN（原生分辨率，不拉伸） |
| 音频（ESC-50） | `scripts/run_esc50_mms.py` | 256×256 灰度梅尔频谱 PNG（不收 WAV） | ResNet-50 / EfficientNet-B0 |
| 组织病理（CRC100K） | `scripts/run_crc100k_mms.py` | 官方数据布局 + 生成图目录 | ResNet-50 |
| 文本（AG News） | `scripts/run_agnews_mms.py` | JSONL（`sample_id`+`text` 或 `path`）或 `.txt` 目录 | TF-IDF + 逻辑回归（纯 CPU） |

**图像尺寸的处理规则：**
1. 输入端：任意尺寸/长宽比都可进入；域配置 `image_policy.canonical_size` 决定统一到什么尺寸（`null` = 保持原尺寸，`[32,32]` = CIFAR-10，等等）。
2. 语义分类器端：除 MNIST 走原生 28×28 灰度通道外，其余图像一律双线性拉伸到 **224×224 RGB** 再喂给分类器——所以 32×32 的 CIFAR-10 和 256×256 的 AFHQ 走完全相同的打分链，这是"自适应尺寸"的实现方式，且已写进各域冻结参考的预处理身份里。
3. 质量指标端：FID/KID/IS/Precision-Recall 按域配置冻结各自的后端与真实参考，不同协议的结果不可混排。

**新模态的接入方式：** 写一个适配器模块（参考 `mms_multimodal/agnews.py`），负责把模态原始输入解码成"样本身份记录 + [N,C] logits"，然后调用 `core.freeze_reference()`（建参考）和 `core.evaluate_logits()`（评测）。熵、校准、候选聚合、防篡改全部由 `core.py` + `mms_eval/semantic.py` 统一完成，适配器不需要碰任何公式。类别必须互斥（`category_mode: exclusive`），多标签属性任务不能直接套用。

## 使用方法

### 通用图像域（AFHQ / CIFAR-10 等）：四步建域，之后一键复评

```sh
# 1. 校验域配置（configs/cifar10.json、afhq.json 等）
python -m mms_eval domain-check --config configs/cifar10.json --evaluator-config configs/evaluator_cifar10.json
# 2. 冻结真实数据划分（需 train/<类别>/*、test/<类别>/* 目录布局）
python -m mms_eval prepare --data /path/to/real_manifest.jsonl --config configs/cifar10.json --out /path/to/splits
# 3. 训练语义评估器（GPU）
python -m mms_eval train --splits /path/to/splits --config configs/evaluator_cifar10.json --out /path/to/evaluator
# 4. 冻结参考（真实校准集定阈值；GPU 或 CPU）
python -m mms_eval reference --splits /path/to/splits --checkpoint /path/to/evaluator/best.pt \
  --config configs/cifar10.json --out /path/to/reference --device cuda:0 --cache-dir /path/to/cache

# 日常：评测新生成模型的已保存图片（输入=文件/目录/清单）
python -m mms_eval evaluate --input /path/to/generated_images \
  --reference /path/to/reference/reference.json --out /path/to/result \
  --device cuda:0 --batch-size 32 --cache-dir /path/to/cache
```

同域评新模型**只需最后一条命令**：分类器、阈值、预处理全部复用冻结参考。另有 `quality`（只算质量指标）、`evaluate-batch`（多来源固定 N 抽样）、`compare`（同口径模型比较，无需 GPU）。

### 各专用域（均为 prepare→train→reference→evaluate 模式）

```sh
# MNIST 28×28（详见脚本 --help）
PYTHONPATH=. python scripts/run_mnist28_mms.py evaluate \
  --classifier-dir ... --reference .../reference.json \
  --generated-manifest .../input_manifest.jsonl --out ... --device cuda:0 --batch-size 256

# CRC-HE 100K
PYTHONPATH=. python scripts/run_crc100k_mms.py evaluate \
  --checkpoint .../best.pt --reference .../reference.json \
  --generated-dir /path/to/gan_output --out ... --count 5000 --device cuda:0 --batch-size 32

# ESC-50（频谱图）
PYTHONPATH=. python scripts/run_esc50_mms.py evaluate \
  --checkpoint .../best.pt --reference .../reference.json --splits .../split_v2 \
  --generated-dir .../spectrograms --source-id my_model_run1 \
  --expected-total 50000 --count 5000 --sample-seed 20260917 --out ...

# AG News 文本（纯 CPU）
PYTHONPATH=. python scripts/run_agnews_mms.py prepare --train-csv train.csv --test-csv test.csv --out splits
PYTHONPATH=. python scripts/run_agnews_mms.py train --splits splits --config configs/agnews_evaluator.json --out classifier
PYTHONPATH=. python scripts/run_agnews_mms.py reference --classifier-dir classifier --splits splits --out reference
PYTHONPATH=. python scripts/run_agnews_mms.py evaluate \
  --classifier-dir classifier --reference reference/reference.json --splits splits \
  --generated generated.jsonl --out result --source-id my_lm_run1
```

## 输出怎么读

每次评测输出 `report.json`（汇总）、`scores.jsonl`（逐样本）、输入清单与哈希；图像流水线另有 `features.npz` 和 HTML 报告。

**一键导出 Excel**（逐图后验熵、概率、全部分数与候选标记 + 汇总 Sheet + 类别对矩阵）：

```sh
PYTHONPATH=. python scripts/export_excel.py /path/to/evaluation_dir
# 生成 evaluation_dir/scores.xlsx；需要 openpyxl（pip install openpyxl）
```

核心字段：

```jsonc
// report.json
{
  "n": 5000,                       // 实评样本数
  "mms": {
    "value": 0.169,                // MMS 候选率（式 9）
    "candidate_count": 845,
    "ci95": [0.1589, 0.1796],      // Wilson 95% 区间
    "mean_entropy": 0.044,         // 平均熵（与 MMS 是两个量）
    "pair_rates_upper_triangle": [[...]],  // 式(12) M_ab，总和恒等于 value
    "predicted_class_counts": [...],       // 分类器预测类别分布（查坍缩用）
    "baselines": { "gini": {...}, ... }    // 4 个同校准基线的候选率
  }
}
```

## 冻结与防篡改机制（改代码前必读）

每个 `reference.json` 记录了：评估器权重哈希、校准/审计样本身份与内容哈希、**评分代码自身的哈希**（`mms_eval/` 全部 `.py` 的联合哈希 + `mms_multimodal/core.py` 的哈希）。评测时任何一项不匹配都会**硬报错**。这意味着：

- 改 `mms_eval/` 里任何一个字节（哪怕加一行注释）或改 `core.py`，所有已冻结参考将无法再用于新评测，必须重建参考。
- 因此**扩展功能一律放在冻结链之外**。本仓库的示范就是 `intervals.py`：它只读已保存的 report.json 计算区间、写独立的 `intervals.json`，不碰核心模块，对全部历史报告可用。
- 有意修改评分算法时，正确流程是：新开版本目录 → 改代码 → 重建该域参考 → 新旧结果分开保存，不得删除旧校验记录强行续用。

## 测试

```sh
pytest tests/ -q    # 117 项 + 3 子测试，全部纯 CPU，约 25 秒
```

覆盖：算法 1 的数学（熵/秩校准/严格阈值/区间）、图像规范化与缓存、参考冻结与防篡改（篡改样本/换代码/换权重都会被拒）、四个专用域适配器的端到端流程（合成数据）、配置契约校验、对照与检测统计。论文核心公式曾与独立重写的算法 1 参考实现做过 1e-12 精度的数值对拍。

## 边界与注意事项

- MMS 候选率 ≠ 人工混淆率；语义效度需独立人工标注验证（论文 4.2 节）。
- 类别必须互斥；CelebA 等多标签属性域只能跑质量指标，不能直接套熵规则。
- 不同评估器（A/B）阈值各自校准，结果不可互相换算；跨模型比较必须同域同参考同图数。
- ESC-50 输入是频谱图不是音频波形，其图像指标不能当音频质量解释。
- ImageNet 预训练权重的评估器不能直接用于 ImageNet 生成图（预训练泄漏），域配置中的 `pretraining_audit` 必须先填写。
