# YOLO 多模型实验工作区设计

目标：在 Windows CPU 上先验证 YOLOv5，在 Linux NVIDIA GPU 服务器上训练，并支持 YOLOv7-tiny 与 YOLOv8 的对比。用户已授权设计结构并初始化当前目录的 Git 仓库。

采用单仓库保存自己的脚本、配置和文档。三个模型采用独立虚拟环境，避免旧版 YOLOv7 的 NumPy 等依赖限制影响其他模型。官方源码保存在被忽略的 third_party 目录，由固定 Git commit 重建；权重、数据和运行结果不提交 Git。保留 init 原始资料。

候选方案：把三个官方库直接复制进主仓库会增大仓库且不易追踪来源；Git submodule 需要额外学习与递归同步；固定 commit 的下载脚本对目前的实验更简单，因此选择后者。

统一入口支持模型选择、CPU 冒烟配置、GPU 正式配置、train/val/predict 和 dry-run。路径以工作区和数据根目录解析，生成各模型可读取的绝对路径 YAML，因此代码可以迁移到不同服务器路径。每次运行保存命令、数据配置、软件环境及退出状态，拒绝复用已有实验目录。

第一版模型：官方 YOLOv5 v7.0 的 yolov5n、官方 YOLOv7 的 yolov7-tiny、Ultralytics 8.3.0 的 yolov8n。这是可复现兼容性基线，不代表最新版本或已验证的最佳模型。Python 3.10，PyTorch 2.5.1；CPU 和 CUDA wheel 分开安装。新架构 GPU 必须重新确认兼容性。

数据：COCO8 仅检查训练流程；正式四类配置采用 person/cat/dog/chair，要求事先准备图片与 YOLO 检测标签。工具不会自动下载完整 COCO，也不会把示例数据成绩当成研究结果。

范围：目录、Git、本地/服务器文档、环境安装入口、数据和权重获取入口、统一运行入口及离线测试。初始化不发布 GitHub 仓库、不连接服务器、不承诺完成三个模型的实际训练。优先尝试本机 YOLOv5 CPU 验证，实际通过情况写入验证记录。
