# ODSR-IHS 数据清洗与训练

原始数据：作者仓库 https://github.com/lvyongshjd/A-new-benchmark-dataset-for-sweeping-robot ，使用 6000 张版本。保留原始压缩包；原始 XML 是转换依据，不使用会省略异常类别目标的现成 TXT。

## 已确认的首版规则

- 保留原版 train/val/test，忽略实际上等于 val+test 的 trainval 文件。
- 整张排除 004235、005107、005214、005423，原因是目标类别或范围仍有歧义。
- 修正规则在 configs/datasets/odsr-ihs-cleaning.json；按图片 ID、零基目标序号、原类别和原框坐标精确匹配，源数据变化即报错。
- 类别顺序沿用源数据正常 TXT 对应编号：trashcan, slippers, wire, socks, carpet, book, feces, curtain, stool, bed, sofa, closestool, table, cabinet。
- bbox 转换沿用原数据 TXT 约定：中心为 (min+max)/2，宽高为 max-min，不额外加减一个像素。此约定不等于断言所有 VOC 数据均使用相同原点。
- 校验原图可解码、XML 尺寸与实际尺寸一致、框有限且有效、类别已审核、划分完整无重复。跨集合的完全相同图片文件会阻止转换；未证明近重复和房间级泄漏不存在。
- 不覆盖已有输出，不修改源数据。输出已存在时，先核对是否已成功，或用 --output 指定新的版本目录，并在训练时传 --data-root。

## 服务器操作

先同步包含清洗脚本和配置的新代码。在项目根目录执行。使用模型环境 Python，因为图像检查依赖 Pillow。

```bash
cd /mnt/fast18/sunbo/digital-picture-process
python3 -m zipfile -e data/raw/odsr-ihs/robot6000.zip data/raw/odsr-ihs/extracted
.venvs/yolov5/bin/python scripts/prepare_odsr.py --source data/raw/odsr-ihs/extracted
```

中文顶层目录显示乱码不影响脚本：自动寻找唯一的 Annotations 目录。解压命令应对已检查过的官方压缩包执行一次。

成功后检查 data/odsr-ihs/summary.json 和 review 中的标注预览。manifest.json 记录每张图、源 XML、生成标签的 SHA256。两台机器用相同源文件和规则时，应生成相同 manifest 和 summary。

先检查训练命令，再在服务器 GPU 上短训三轮：

```bash
python3 scripts/run.py --model yolov5 --profile server --dataset configs/datasets/odsr-ihs.json --epochs 3 --batch 4 --workers 2 --name v5n_odsr_check_01 --dry-run
python3 scripts/run.py --model yolov5 --profile server --dataset configs/datasets/odsr-ihs.json --epochs 3 --batch 4 --workers 2 --name v5n_odsr_check_01
```

初始权重为官方 yolov5n.pt，输入 640；模型根据数据配置改为 14 类。仅验证集用于短训期间评价，不提前使用 test。三轮不是正式性能结论。先检查完整日志、曲线和预测效果再进行正式实验；显存不足时减小 batch。

正式 50 轮基线使用新实验名，并重新从官方预训练权重开始：

```bash
python3 scripts/run.py --model yolov5 --profile server --dataset configs/datasets/odsr-ihs.json --batch 4 --workers 2 --name v5n_odsr_baseline_01
```

v7-tiny 和 v8 须先准备对应环境。三个模型保持同一数据版本和划分；当前包装器的 AMP 等模型训练设置并非完全一致，正式比较需记录差异。不同框架的标签 cache 不应混用，按 docs/data.md 处理。

数据、权重和训练结果不进 Git。只同步脚本、配置、清洗记录和报告。此数据没有人、猫狗和机器人，实验结论限于其 14 类低机位物体检测，不代表动态 SLAM 已通过验收。
