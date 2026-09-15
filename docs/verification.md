# 验证记录

## 2026-09-15 字体修复后的实际 CPU 验证

用户已完成环境和数据下载。旧版 YOLOv5 在 check_dataset 中尝试下载 Arial.ttf，遇到 HTTP 308 导致训练终止。入口已改为预先缓存本机 Arial，Linux 可使用 Matplotlib 内置 DejaVu Sans 作为英文标签字体。官方源码未修改，字体不进入 Git。

实际运行：python scripts/run.py --model yolov5 --profile cpu-smoke --name v5_cpu_fontfix_20260915。

结果：Python 3.10.20、torch 2.5.1+cpu、YOLOv5 915bbf2，COCO8 的 4 张训练图和 4 张验证图，imgsz=320、batch=2、workers=0、1 epoch。训练与最终验证完成；experiment.json 的 status=completed、exit_code=0。已核对 weights/best.pt 和 last.pt 均为 3,981,542 字节，results.csv 与 results.png 存在。指标仅为冒烟验证，不作为正式性能结论。

11 项离线测试通过。YOLOv7-tiny、YOLOv8 和服务器 GPU 训练仍未在此处验证。以下保留初始化时的历史记录，其中“未完成”描述仅适用于当时。

## 初始化历史记录

验证时间：2026-09-15。

## 已执行

- Python 标准库 unittest：8 项测试通过，覆盖三模型 CPU 命令、Unicode 数据路径、实验名限制、权重/图片参数校验、验证 workers 参数和 ZIP 路径穿越拒绝。
- 3 个模型 × 2 套 profile × train/val/predict：18 种 dry-run 均成功解析。dry-run 不执行网络请求、不写实验目录、不启动训练。
- scripts 与 tests 的 compileall 通过。
- COCO80 类别表检查：80 类，chair 索引为 56。
- Git 忽略规则确认覆盖数据、权重、结果、官方源码、虚拟环境、Python 缓存、tmp 与 .env。
- 官方 YOLOv5 v7.0 tag 与 YOLOv7 HEAD 曾通过 git ls-remote 核对并固定到 configs/models.json。
- 独立代码审阅提出的验证 workers 和 PyTorch 后端切换问题已修复，并补充回归验证。

## 未完成的运行时验证

已尝试执行 `python scripts/setup.py --model yolov5 --backend cpu`，但官方源码下载阶段连续出现 TLS 握手失败（unexpected EOF / SSL connection failed）。随后只读检查 GitHub Release、codeload 和 PyPI 同样失败。

因此本次没有完成模型依赖安装、官方权重下载或实际 YOLOv5 CPU 训练。YOLOv7-tiny、YOLOv8 也没有执行实际训练。三个入口的 dry-run 通过不代表依赖组合和模型运行时已经通过验证。

`third_party/yolov5/` 中可能保留未完成下载的 Git 目录，setup 会重试获取指定 commit。网络恢复后从 README 的本机验证命令继续即可。没有下载完整 COCO，没有启动正式训练，也没有连接服务器或发布 GitHub。

## Git 本机说明

仓库初次由沙箱账号创建，普通用户 Git 的所有权检查会拒绝访问。已为当前用户增加仅针对 D:/Work/数字图像处理 的 safe.directory 条目，没有添加通配符。主仓库分支为 main；未设置 origin。初次提交内容包含原始 init 资料和项目脚本/配置/文档，不包含本机大文件。

## 后续验收

网络恢复后依次运行 YOLOv5 的 setup、COCO8 准备、CPU train、val 和 predict。检查 experiment.json 的 exit_code=0、weights/best.pt 存在及预测图像正常。服务器需根据真实 GPU/驱动验证 PyTorch wheel 兼容性，再跑正式训练。
