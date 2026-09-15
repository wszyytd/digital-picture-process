# YOLOv5n 本机 CPU 流程验证记录

实验编号：EXP-001。日期：2026-09-15。状态：通过。

本次完成了预训练权重加载、数据检查、1 轮 CPU 训练、验证和权重保存，确认 YOLOv5n 的基础训练链路可运行。该实验是 COCO8 冒烟验证，不能作为正式精度比较、收敛判断、部署性能或 SLAM 改善的证据。独立 predict 命令未纳入本次记录。

## 配置与环境

| 项目 | 实际配置 |
|---|---|
| 模型 | YOLOv5n，官方 YOLOv5 v7.0 |
| 上游源码 | `915bbf294bb74c859f0b41f1c23bc395014ea679` |
| 初始权重 | 官方 COCO 预训练 yolov5n.pt |
| 初始权重 SHA256 | `4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3` |
| 运行系统与设备 | Windows，本机 CPU；未使用核显加速 |
| Python / PyTorch | 3.10.20 / 2.5.1+cpu |
| 数据 | COCO8：4 张训练图、4 张验证图，80 类定义 |
| 本次验证目标数 | 17 个实例；不是每个类别都有样本 |
| 输入分辨率 / batch | 320 / 2 |
| epoch / workers / seed | 1 / 0 / 1 |
| 优化器 | SGD；初始 lr0=0.01，momentum=0.937 |
| 线程环境参数 | OMP_NUM_THREADS=2，MKL_NUM_THREADS=2；非实测全局线程上限 |
| 开始时间 | 2026-09-15 14:03:06（北京时间） |
| 结束时间 | 2026-09-15 14:05:47（北京时间） |
| 包装入口总耗时 | 161.52 秒，包含加载、检查、训练、验证和保存，不能当推理延迟 |
| 退出状态 | completed，exit_code=0 |

实际运行命令（在仓库根目录执行）：

```powershell
python scripts/run.py --model yolov5 --profile cpu-smoke --name v5_cpu_fontfix_20260915
```

该名字对应的实验目录已存在，再次运行应更换 `--name`，不要覆盖原结果。

## Loss

以下取自原始 results.csv，epoch=0 表示第 1 轮。只有一个点，无法判断收敛趋势。

| 指标 | 训练集 | 验证集 |
|---|---:|---:|
| box_loss | 0.066499 | 0.068003 |
| obj_loss | 0.037575 | 0.017002 |
| cls_loss | 0.087115 | 0.033932 |

训练采用数据增强，训练 loss 和验证 loss 的计算条件并不完全相同，不宜仅按两者大小判断过拟合。

## 检测指标

| 验证集指标 | 数值（0～1） |
|---|---:|
| Precision | 0.68683 |
| Recall | 0.38333 |
| mAP@0.5 | 0.76434 |
| mAP@0.5:0.95 | 0.36337 |

这些指标来自 4 张验证图，样本量很小，且 COCO 预训练与 COCO8 有数据来源重叠，不能据此声称模型具有室内低机位泛化能力。终端显示值经过舍入，表格保留 CSV 中的精度。

曲线副本：[results.png](results.png)。数值副本：[metrics.json](metrics.json)。

## 本次故障与修复

先前实验 `v5_cpu_smoke` 在数据检查阶段下载 Arial.ttf 时遇到 HTTP 308，未进入训练。已在项目入口增加本地字体准备：优先使用 Windows 系统 Arial；其他环境可使用 Matplotlib 内置 DejaVu Sans，适用于当前英文类别标签。字体只保存在被 Git 忽略的机器缓存中，未修改官方源码。

运行记录中的主仓库版本是 `bbe396ed1b2d545d32cf4af12a9e24611cc27af6`，且 `workspace_dirty=true`：因为运行时字体修复尚未提交。该字体修复随后保存于提交 `58cf733`，复现应包含该修复，不能仅检出原始 bbe396e。代码回归测试 11 项通过。

## 结果归档

原始完整目录：`runs/yolov5/v5_cpu_fontfix_20260915/`（不进入 Git）。

- 已核对 best.pt 和 last.pt 均生成，各 3,981,542 字节。
- 本记录旁保留实际 experiment.json、environment.txt、opt.yaml、hyp.yaml、data.yaml 和 results.png 副本，以便 GitHub 同步后仍可查阅。
- [artifacts.json](artifacts.json) 保存原始权重、日志配置和结果文件的大小及 SHA256；权重本体仍只在原始 runs 目录。
- 快照中的绝对路径用于记录当时环境，不应直接作为服务器配置；服务器使用项目入口和自己的 `--data-root` 生成配置。

## 下一步

1. 使用本次 best.pt 执行独立图片预测并人工检查检测框。
2. 确定正式数据、标注类别和固定划分，记录数据版本。
3. 在服务器验证 GPU 环境后开展正式训练，并使用同一数据划分比较 YOLOv7-tiny 和 YOLOv8n。
