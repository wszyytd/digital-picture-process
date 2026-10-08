# YOLOv8n：ODSR14类训练

快捷脚本 `scripts/train_v8.py` 复用现有setup.py和run.py，使用固定的ultralytics8.3.0、YOLOv8n官方权重。不会使用v5/v7权重；正式训练默认不接续3轮检查权重。

默认：imgsz640、batch32、训练workers4、GPU0、SGD、seed1、patience0。现有v8入口关闭AMP（amp=False），保留该配置便于追踪；不要把不同模型同为50轮称作全部超参数完全一致。

## 服务器操作

先把新增scripts/train_v8.py同步到服务器相同目录；已有scripts/run.py、setup.py、workspace.py、configs及environments须保留。无需重新上传数据和v5/v7结果。

```bash
cd /mnt/fast18/sunbo/digital-picture-process
python3 scripts/train_v8.py setup
```

安装复用独立.venvs/yolov8环境，默认cu121。首次需要下载官方预训练权重及依赖，等待安装结束、CUDA available为True。

切换框架时确认没有其他任务使用这份数据，再清理可重建的旧标签缓存（不删除图片和TXT标注）：

```bash
python3 - <<'PY'
from pathlib import Path
for p in Path('data/odsr-ihs/labels').glob('*.cache'):
    p.unlink()
    print('Removed cache:', p)
PY
```

不要让不同框架同时写同一数据目录的cache。需要并行训练时，准备各自的数据副本，并用--data-root指定。

先检查命令，不启动训练：
```bash
python3 scripts/train_v8.py train --dry-run
```

运行3轮检查：
```bash
python3 scripts/train_v8.py check
```

确认正常结束后，再从官方权重开始50轮正式训练：
```bash
mkdir -p logs
nohup python3 -u scripts/train_v8.py train > logs/v8n_odsr_b32_e50_01.log 2>&1 < /dev/null &
echo "后台进程 PID: $!"
tail -f logs/v8n_odsr_b32_e50_01.log
```

Ctrl+C只退出tail查看。日志不再增加不代表卡住：tail会持续等待。

输出目录：runs/yolov8/v8n_odsr_b32_e50_01，权重在weights/best.pt；包含console.log、experiment.json和Ultralytics生成的results.csv等。检查experiment.json的status=completed、exit_code=0，再判断成功。

如显存不足，先用`python3 scripts/train_v8.py check --batch 16 --name v8n_odsr_check_b16_01`确认，再执行正式训练 `python3 scripts/train_v8.py train --batch 16`；正式名称自动变为v8n_odsr_b16_e50_01，外层日志名也相应修改。已有同名实验会被拒绝，用--name新名称，避免覆盖。

## 测试固定权重

完成训练后根据验证集选定模型，再执行测试；不要根据测试集反复调参：
```bash
python3 scripts/train_v8.py test --weights runs/yolov8/v8n_odsr_b32_e50_01/weights/best.pt
```

默认test split、batch32、workers0、conf0.001、iou0.6、max_det300；960张、2590目标。预期输出目录runs/yolov8/v8n_odsr_test_01。评估参数用于AP计算，不是部署置信度阈值。

## 本机与复现说明

管理脚本只需标准库，模型环境由setup.py准备。本机CPU可运行`python scripts/train_v8.py setup --backend cpu`；小样本CPU验证仍可使用既有run.py的cpu-smoke配置。快捷check默认对完整ODSR运行3轮，本机无GPU时不建议直接使用默认check。

运行`python scripts/train_v8.py --help`查看可覆盖参数。所有相对路径相对于项目根目录。初次50轮结果与v5n、v7-tiny首次50轮比较；记录架构、增广、AMP和训练实现差异，v5n后续微调结果另列。

本次新增代码通过无GPU命令规划及回归测试；本机尚未安装YOLOv8环境，未声称已完成实际v8训练，实际环境与训练通过服务器3轮检查确认。
