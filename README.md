# 基于多线激光雷达的果园语义分割

**Orchard Semantic Segmentation with Multi-line LiDAR**

本仓库开源发布 **果园全场景点云语义分割与树干点云实例分割** 的完整代码与标注数据集，可用于算法复现、基准测试与二次开发。

> 不同于通用的城市道路点云数据集，本数据集面向我国标准化种植果园场景，包含 **9 类**典型环境要素的逐帧精细语义标注，可直接用于果园机器人感知算法研究与基准测试。

---

## ✨ 主要工作

1. **果园多类别点云数据集构建** —— 基于 VLP-16 多线 LiDAR 与阿克曼转向移动平台实地采集，采用 CloudCompare 逐帧标注树冠、树干、地面、斜坡、栅栏、电线杆、线缆、灌木、人员共 9 类目标，并提出基于点云水平密度约束的前向半球 ROI 提取方法（ROI 半径 10 m）。
2. **基于球面投影的轻量化全场景语义分割** —— 将三维点云经球面坐标变换映射为 5 通道二维多模态图像（径向距离、xyz 坐标、回波强度），设计 **仅在水平方向多级下采样、垂直方向保持分辨率不变** 的轻量化非对称编码—解码网络，结合加权交叉熵缓解类别不平衡，最后经球面逆投影得到点级语义标注。
3. **树干语义 + 实例分割** —— 先通过类别映射策略得到树干语义分割，再经球面逆投影恢复三维树干点云，最后结合密度聚类（DBSCAN）区分相邻树干个体。

## 🗂 数据集

### 规格概览

| 项目 | 说明 |
| :--- | :--- |
| 传感器 | Velodyne **VLP-16**（16 线，垂直 FOV −15° ~ +15°） |
| 采集平台 | 轮趣科技阿克曼转向小车 |
| 采集场景 | 江苏省句容市后白镇薄壳山核桃（Carya illinoinensis）标准化种植基地 |
| 处理流程 | ROI 提取（半径 10 m）→ CloudCompare 逐帧标注 → 旋转/平移数据增强 → 转 KITTI 格式 → 按序列 8:1:1 划分 |
| 序列数 | **20**（`00`–`19`） |
| 帧数 | **1 779** 帧原始标注点云（增强后训练集可扩充至 28 464 帧，增强系数 ×16） |
| 单帧点数 | ≈ 8 500（ROI 内） |
| 格式 | KITTI / SemanticKITTI 序列格式 |

### 类别体系

| ID | 名称 | 中文 | 点云占比 |
| :-: | :--- | :--- | ---: |
| 0 | `unlabeled` | 未标注 | 0.37 % |
| 1 | `tree_trunk` | 树干 | 4.50 % |
| 2 | `pedestrian` | 人员 | 0.82 % |
| 3 | `pole` | 电线杆 | 0.29 % |
| 4 | `pole_cable` | 线缆 | 0.10 % |
| 5 | `fence` | 栅栏 | 0.82 % |
| 6 | `slope` | 斜坡 | 7.03 % |
| 7 | `shrub` | 灌木 | 2.76 % |
| 8 | `ground` | 地面 | 64.20 % |
| 9 | `crown` | 树冠 | 19.11 % |

> ⚠️ 地面 + 树冠两类合计超过 80 %，树干仅占 4.5 %，**类别严重不平衡**，这正是本项目引入加权交叉熵损失的原因。

### 数据划分（按序列划分，避免同一场景泄漏到不同子集）

| 子集 | 序列 | 帧数 |
| :--- | :--- | ---: |
| Train | `00`–`15` | 1 431 |
| Valid | `16`–`17` | 178 |
| Test | `18`–`19` | 170 |

### 目录与文件格式

```
train/tasks/semantic/dataset/sequences/
└── 00/
    ├── velodyne/          # *.bin  —— 点云
    │   ├── 000000.bin
    │   └── ...
    └── labels/            # *.label —— 逐点语义标签（与 bin 同名）
        ├── 000000.label
        └── ...
```

- **`.bin`**：每点 4 个 `float32`（x, y, z, intensity），共 16 B/点。
- **`.label`**：每点 1 个 `uint32`，值为上表类别 ID，与同名的 `.bin` 逐点一一对应。

## 🛠 环境依赖

实验环境为 Python 3.7 / CUDA 11.1：

```bash
pip install -r requirements.txt
```

主要依赖：`torch==1.8.0+cu111`、`torchvision==0.9.0+cu111`、`numpy`、`PyYAML`、`scipy`、`matplotlib`、`opencv-python`。

> 若使用较新的 GPU/驱动，可按本机 CUDA 版本替换 torch 版本，代码逻辑无需改动。

## 🚀 使用

所有命令均在 `train/tasks/semantic/` 目录下执行。

### 训练

```bash
cd train/tasks/semantic

python train.py \
  -d ./dataset \
  -ac config/arch/darknet26.yaml \
  -dc config/labels/semantic-kitti.yaml \
  -l ~/logs/orchard_seg
```

- `-d` 数据集根目录（其下含 `sequences/`）
- `-ac` 网络结构配置（`config/arch/darknet26.yaml`，DarkNet26 非对称编码—解码）
- `-dc` 类别定义与划分（`config/labels/semantic-kitti.yaml`）
- `-l` 日志与权重输出目录（默认 `~/logs/日期-时间`）

### 推理

```bash
python infer.py \
  -d ./dataset \
  -m ~/logs/orchard_seg \
  -l ~/logs/orchard_pred
```

预测结果保存在 `-l` 目录下，每个序列的 `predictions/*.label`。

### 评估（mIoU / mF1）

```bash
python evaluate_iou.py \
  -d ./dataset \
  -p ~/logs/orchard_pred \
  -dc config/labels/semantic-kitti.yaml \
  -s test
```

### 可视化

```bash
# 查看数据集本身的标注（默认序列 00）
python visualize.py -d ./dataset -s 00

# 查看推理结果（与标注对比）
python visualize.py -d ./dataset -p ~/logs/orchard_pred -s 18
```

> 注：`train.py` 中 `-dc` 的默认值假定从仓库根目录运行。若按上文在 `train/tasks/semantic/` 目录下执行，请像上面那样显式传入 `-dc config/labels/semantic-kitti.yaml`。

## 📊 实验结果

| 任务 | 指标 | 结果 |
| :--- | :--- | ---: |
| 全场景语义分割（9 类） | mIoU / mF1 | **0.840** / **0.910** |
| 全场景语义分割 | 单帧推理耗时 | **9.0 ms** |
| 树干语义分割 | IoU / F1 | **0.806** / **0.873** |
| 树干实例分割 | mIoU / mF1 | **0.876** / **0.938** |
| 树干实例分割 | 单帧耗时 | **11.2 ms** |

## 📁 仓库结构

```
lidar-bonnetal/
├── train/                          # 训练框架（基于 lidar-bonnetal 改造）
│   ├── backbones/                  # DarkNet 主干
│   ├── common/                     # 球面投影、Laserscan、可视化等通用工具
│   └── tasks/semantic/
│       ├── config/
│       │   ├── arch/darknet26.yaml         # 网络与训练超参（img 900×16）
│       │   └── labels/semantic-kitti.yaml  # 9 类定义、占比、划分
│       ├── dataset/
│       │   ├── kitti/parser.py             # 数据读取与球面投影
│       │   └── sequences/00..19/           # ★ 标注数据集
│       ├── decoders/                       # 解码器
│       ├── modules/                        # trainer / segmentator / ioueval / user
│       ├── postproc/KNN.py                 # 后处理
│       ├── train.py                        # 训练入口
│       ├── infer.py                        # 推理入口
│       ├── evaluate_iou.py                 # 评估入口
│       └── visualize.py                    # 可视化入口
├── requirements.txt
├── LICENSE
└── README.md
```

## 📚 引用

若本仓库的代码或数据集对您的研究有帮助，请引用：

```bibtex
@article{li2026realtime,
  title   = {Real-time semantic segmentation of orchard LiDAR point clouds
             with a spherical-image convolutional neural network},
  author  = {Li, Q. and Jing, S.},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  pages   = {112634},
  doi     = {10.1016/j.compag.2026.112634}
}
```

> DOI: [10.1016/j.compag.2026.112634](https://doi.org/10.1016/j.compag.2026.112634)

## 📜 许可证与致谢

- 本仓库代码基于开源框架 [lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal)（MIT License, © 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, University of Bonn）改造，遵循 **MIT License**，原始版权声明见 [LICENSE](LICENSE)。
- 数据集由作者实地采集并逐帧标注，随本仓库发布供学术研究使用；使用数据集时请注明出处。
- 感谢导师 **李秋洁 副教授** 的悉心指导。

---

**联系方式**：景思淼 · 南京林业大学 · 控制科学与工程（农林信息化）
