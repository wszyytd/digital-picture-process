# GitHub 与服务器工作流

## 本机与服务器的职责

本机 Windows 无独显：运行离线测试、COCO8 CPU 训练/预测、检查标注和调试接口。服务器 NVIDIA GPU：运行正式数据训练、重复实验与性能测量。

## 上传 GitHub

本地仓库已初始化为 main，未设置远程地址。先在 GitHub 创建空仓库，建议先使用私有仓库；不要在网页上同时初始化 README，避免无关的初始历史。

在 PowerShell 输入自己创建的仓库地址：

```powershell
$repoUrl = Read-Host '粘贴你的 GitHub 仓库地址'
git remote add origin $repoUrl
git push -u origin main
```

如果远程地址已存在，用 `git remote -v` 检查，不要重复添加。此文档中的命令尚未执行，不会自动创建或公开发布仓库。

后续修改：

```powershell
git status --short
git add README.md configs scripts environments docs tests reports
git diff --cached --stat
git commit -m "Update perception experiment configuration"
git push
```

data、weights、runs、third_party 和虚拟环境均不进 Git。不要使用 git add -f 强制添加。原始 init 中的资料已在初始提交范围内，若将来公开仓库，先检查其中个人信息和图片来源。

## 服务器克隆

在 Linux 服务器安装 Git 和 uv；准备好普通 Python 3 来运行项目管理脚本：

```bash
read -r -p "GitHub repository URL: " REPO_URL
git clone "$REPO_URL" indoor-perception
cd indoor-perception
nvidia-smi
```

按照显卡与驱动选择 `cpu/cu118/cu121/cu124`。当前固定 PyTorch 2.5.1；较新 GPU（例如需要更新架构支持的型号）可能不能使用这个兼容性基线，应先验证支持再升级环境配置，不能仅根据 nvidia-smi 顶部的 CUDA 字样照抄。

```bash
python3 scripts/setup.py --model yolov5 --backend cu121
python3 scripts/setup.py --model yolov7-tiny --backend cu121
python3 scripts/setup.py --model yolov8 --backend cu121
```

安装脚本会显示 CUDA 是否可用。先跑一个 GPU 小测试：

```bash
python3 scripts/prepare_smoke.py
python3 scripts/run.py --model yolov5 --profile cpu-smoke --device 0 --name v5_gpu_smoke
```

再把正式数据另行传到服务器，例如 /data/indoor4，核对图片与标签数量和内容校验文件，然后训练：

```bash
python3 scripts/run.py --model yolov5 --profile server --data-root /data/indoor4 --name v5n_indoor_seed1
python3 scripts/run.py --model yolov7-tiny --profile server --data-root /data/indoor4 --name v7tiny_indoor_seed1
python3 scripts/run.py --model yolov8 --profile server --data-root /data/indoor4 --name v8n_indoor_seed1
```

上面的三个命令顺序执行；不要在显存/资源不够时同时启动。推荐在服务器的 tmux 会话运行，以便 SSH 断开后继续。运行期间不更新仓库或数据；下一轮开始前用 `git pull --ff-only` 同步。

## 评估最终测试集

以 YOLOv5 为例：

```bash
python3 scripts/run.py --model yolov5 --profile server --action val --split test --data-root /data/indoor4 --weights runs/yolov5/v5n_indoor_seed1/weights/best.pt --name v5n_indoor_test
```

测试集不参与选择轮次和阈值。测试输出归档在独立目录。将需要展示的曲线、对比表和结论整理进 reports 后提交；权重通过独立存储或 GitHub Release 另行管理。

## 中断与续训

当前统一入口不提供 resume，避免三个官方实现的差异被误隐藏。保留完整 runs 目录和 last.pt，按该模型官方 resume 方式继续。不要把 last.pt 作为普通 --weights 就声称是精确续训；这可能只是重新微调，优化器和训练进度语义不同。
