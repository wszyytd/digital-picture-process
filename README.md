# 室内机器人 YOLO 感知实验

本项目用于比较 YOLOv5n、YOLOv7-tiny 和 YOLOv8n，为后续动态 SLAM 提供目标检测结果。工作流为 Windows CPU 验证 → GitHub 同步代码 → Linux GPU 训练。当前原始资料保存在 `init/`。

## 项目结构

```text
configs/
  models.json             # 官方来源、固定 commit、预训练权重
  datasets/               # 共用类别表和数据路径
  profiles/               # CPU 冒烟配置与服务器训练配置
environments/             # 三个模型各自的依赖入口
scripts/
  setup.py                # 创建独立环境，获取官方源码和权重
  prepare_smoke.py         # 下载 COCO8，验证目录
  run.py                  # train / val / predict / dry-run
  legacy_entry.py         # v5/v7 官方入口的 Windows 中文图片路径适配
  workspace.py            # 路径、下载和版本检查
tests/                    # 不需要 PyTorch 的离线入口测试
docs/                     # 数据规范、同步流程、实验协议、验证记录
reports/                  # 人工审核后的实验结论，可提交 Git
init/                     # 原始项目资料
data/                     # 数据集，不提交 Git
weights/                  # 预训练/导出权重，不提交 Git
runs/                     # 日志、配置快照、训练结果，不提交 Git
third_party/              # 按固定版本获取的官方源码，不提交 Git
.venvs/                   # 每个模型独立环境，不提交 Git
.python/                  # uv 管理的项目内 Python，不提交 Git
```

采用独立环境的原因：YOLOv7 是旧版实现，要求 NumPy < 1.24。YOLOv5 使用原始官方仓库，不使用 YOLOv5u 替代它。YOLOv7-tiny 使用自己的 tiny 网络和超参数配置。YOLOv8 使用单独的 Ultralytics 包。

当前固定基线是 Python 3.10、PyTorch 2.5.1 / torchvision 0.20.1、YOLOv5 v7.0、YOLOv7 固定 commit、Ultralytics 8.3.0。固定版本用于复现，不表示它们是最新版。升级任何依赖后都要重新验证三个模型。

## 本机首先验证 YOLOv5

下面命令在仓库根目录的 PowerShell 运行。现有 Python 3.14 只用来运行轻量管理脚本；模型环境由 uv 单独获取 Python 3.10，不修改系统 Python。

需要 `git` 和 `uv` 在 PATH。uv 安装说明见 [官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

```powershell
cd D:\Work\数字图像处理
python scripts/setup.py --model yolov5 --backend cpu
python scripts/prepare_smoke.py
python scripts/run.py --model yolov5 --profile cpu-smoke --name v5_cpu_smoke
```

首次需要联网下载源码、权重、Python 和 CPU PyTorch，依赖体积明显大于 COCO8。网络失败时可重试 setup；脚本不会重置已经修改的官方仓库。仅使用官方来源的可信权重。

CPU 配置为 320 像素、batch 2、1 epoch、workers 0，并设置 2 线程环境参数（官方框架可能另行调整内部线程）。核显不作为 CUDA 设备；本机始终使用 CPU。COCO8 的成绩仅用于确认流程，不能放入正式模型对比表。

运行前可随时查看将执行的命令，不安装依赖、不下载数据、不创建实验目录：

```powershell
python scripts/run.py --model yolov5 --profile cpu-smoke --dry-run
```

训练成功后，验证权重和预测图片：

```powershell
python scripts/run.py --model yolov5 --action val --weights runs/yolov5/v5_cpu_smoke/weights/best.pt --name v5_cpu_val
python scripts/run.py --model yolov5 --action predict --weights runs/yolov5/v5_cpu_smoke/weights/best.pt --source data/coco8/images/val --name v5_cpu_predict
```

每次使用新的实验名。入口拒绝覆盖已有目录，防止不同实验互相污染。

## 后续验证另外两个模型

```powershell
python scripts/setup.py --model yolov7-tiny --backend cpu
python scripts/run.py --model yolov7-tiny --profile cpu-smoke --name v7tiny_cpu_smoke

python scripts/setup.py --model yolov8 --backend cpu
python scripts/run.py --model yolov8 --profile cpu-smoke --name v8_cpu_smoke
```

入口已经为三个模型分别设置官方训练命令。是否已经实际通过训练，请看 [验证记录](docs/verification.md)，不要把 dry-run 当成训练通过。

## 服务器正式训练

完整步骤见 [GitHub 与服务器工作流](docs/server-workflow.md)。正式数据要求见 [数据规范](docs/data.md)。

先准备 `data/indoor4/` 或服务器数据盘中的同名结构。服务器选择匹配 GPU/驱动的 PyTorch 后端；下例的 cu121 仅是经过官方发布的可选 wheel，不是对未知服务器硬件的自动判断。

```bash
python3 scripts/setup.py --model yolov5 --backend cu121
python3 scripts/run.py --model yolov5 --profile server --data-root /data/indoor4 --name v5n_indoor_seed1
```

同样配置可替换 `--model yolov7-tiny` 或 `--model yolov8`。每个模型先 setup。可以用 `--batch 4` 调小显存占用。

## 输出与复现

每次运行的 `runs/<model>/<name>/` 包含：

- `experiment.json`：完整参数、官方源码版本、输入权重 SHA256、主仓库 commit/脏状态、运行时间与退出状态。
- `data.yaml`：该机器解析后的数据路径及类别表，自动生成。
- `environment.txt`：实际安装的 Python 包版本。
- `console.log`：完整运行日志。
- 官方生成的 `weights/best.pt`、`last.pt`、曲线和指标。v5/v8 通常输出 results.csv，v7 通常输出 results.txt。

GPU/CPU 的运行环境不应拷贝到另一台机器，按配置重建即可。正式评测协议见 [模型比较规范](docs/experiments.md)。

## 开发检查

```powershell
python -m unittest discover -s tests -v
git diff --check
git status --short
```

Git 只保存代码、配置、文档和 init 原始资料。不要强制添加被忽略的数据、权重、环境或密钥。当前不附加项目许可证：发布前由项目组决定自有代码许可，并保留各官方实现/数据的来源与许可要求。
