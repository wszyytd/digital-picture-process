# 数据准备规范

## 两套数据配置

- `configs/datasets/coco8.json`：官方 COCO8，80 类、4 张训练图和 4 张验证图，只做环境验证。
- `configs/datasets/indoor4.json`：正式实验的初始四类 person/cat/dog/chair，数据尚需准备；此配置不代表数据已下载或已经选定最终类别。

项目管理配置使用 JSON，便于在没有 PyYAML 的系统 Python 中执行。运行时自动生成兼容三个官方框架的 YAML。YOLOv7 不应直接接收只有 path 字段的新版 YOLO YAML，所以每个 split 都转换成绝对路径，并同时输出 nc 和列表形式的 names。

## 目录

```text
data/indoor4/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

图片与 TXT 同名，每行一个目标：

```text
0 0.5 0.5 0.2 0.4
```

格式为 class_id、x_center、y_center、width、height。坐标归一化到 0～1，宽高必须为正，类别编号必须在类别表范围内。空场景用空 TXT 明确记录为负样本。当前入口面向目录式 split；如果要使用 TXT 图片列表，需要另外统一各框架的路径解析。

COCO 原编号不能直接用于四类配置。person/cat/dog/chair 必须映射为项目编号 0/1/2/3，并保留图片中所有属于目标类别的标注。没有把原始 COCO 全部数据自动下载到项目中。

## 数据来源

- [COCO](https://cocodataset.org/#download)：通用检测/实例分割，适合人、猫、狗、椅子基线。
- [SUN RGB-D](https://rgbd.cs.princeton.edu/)：室内 RGB-D，需转换原标注格式。
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)：动态 SLAM 评测。
- [Bonn Dynamic](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)：动态 SLAM 评测。

TUM/Bonn 的位姿真值不是 YOLO 检测框标注。保留独立序列用于最终 SLAM 测试。

## 数据划分与迁移

先按视频/房间/采集批次划分，再做训练增强。相邻帧不能跨训练和测试。三个模型使用同一份划分，保留数据来源、许可、样本数量和类别分布记录。

将正式数据的版本号、图片清单、划分方法和校验文件放进 reports 或自建小型清单目录；大文件另行传输到服务器。当前 experiment.json 记录配置和权重校验值，不会自动为整个数据集生成内容校验；正式实验前需要固定数据版本。

不要同时让三个框架在同一可写数据目录中生成标签 cache。官方 cache 格式可能不同。建议顺序运行，切换框架前清理自己生成的标签 cache，或为每个模型准备独立的可写数据副本并验证内容一致。

`--data-root` 可以指向服务器路径，例如 `/data/indoor4`。不用修改已提交的配置文件。

Windows 上 YOLOv7 的 train/val 均强制 workers=0，避免子进程绕过中文图片路径适配。Linux 上按 profile 的 workers 设置。

## ODSR-IHS 低机位数据

新增 14 类配置 `configs/datasets/odsr-ihs.json`。清洗、服务器复现和训练步骤见 [ODSR-IHS 操作说明](odsr-ihs.md)。原始 XML 经精确修正后重新生成标签；不使用原包中省略异常目标的 TXT。
