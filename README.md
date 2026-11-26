# Real-time semantic segmentation of orchard LiDAR point clouds with a spherical-image convolutional neural network

### 基于球面图像卷积神经网络的果园 LiDAR 点云实时语义分割

This repository releases the complete source code and annotated dataset for *Real-time semantic segmentation of orchard LiDAR point clouds with a spherical-image convolutional neural network*, supporting algorithm reproduction, benchmark evaluation and secondary development.

本仓库开源发布基于球面图像卷积神经网络的果园 LiDAR 点云实时语义分割的完整代码与标注数据集，可用于算法复现、基准测试与二次开发。

> Different from general urban-road LiDAR datasets, this dataset is built for standard-planted orchard scenarios in China. It contains frame-by-frame fine-grained semantic annotations of 9 typical environmental elements, and can be directly applied to perception-algorithm research and benchmark testing for orchard robots.
>
> 不同于通用的城市道路点云数据集，本数据集面向我国标准化种植果园场景，包含 9 类典型环境要素的逐帧精细语义标注，可直接用于果园机器人感知算法研究与基准测试。

---

## 主要工作 / Main Contributions

### 1. 果园九类别激光雷达点云数据集构建

使用搭载 VLP-16 激光雷达的阿克曼移动机器人采集核桃果园原始点云；采用半径 10 m 前向半球 ROI 剔除遮挡与远距离无效点云；CloudCompare 逐点标注 9 类目标：树干、树冠、地面、灌木、斜坡、电线杆、线缆、栅栏、行人；按 8:1:1 划分训练、验证、测试集，采用少数类帧复制、在线随机旋转数据增强。

**Construction of nine-class orchard LiDAR point-cloud dataset** — Raw point clouds are collected by an Ackermann mobile robot equipped with a VLP-16 LiDAR sensor. A forward-hemisphere ROI with a 10 m radius is adopted to remove occluded and distant invalid points. Nine classes (tree trunk, tree crown, ground, shrub, slope, utility pole, power line, fence and pedestrian) are annotated point-by-point using CloudCompare. The dataset is split into training, validation and test sets at a ratio of 8:1:1. Minority-class frame duplication and online random rotation are applied for data augmentation on the training set.

### 2. 基于球面投影的非对称编-解码轻量化语义分割网络

球面投影将无序 3D 点云转为 5 通道多模态球面图像（坐标、径向距离、回波反射率）并做通道归一化；提出 **Darknet26-AS** 非对称编码器，仅水平下采样，保留垂直分辨率；解码器采用非对称转置卷积；对数平滑加权交叉熵缓解类别不平衡；分割完成后执行球面逆投影，把图像预测标签映射回原始三维点云。

**Lightweight asymmetric encoder-decoder semantic segmentation network based on spherical projection** — Unordered 3D point clouds are projected into 5-channel multimodal spherical images (coordinates, radial range, echo reflectivity), followed by channel-wise normalization. An asymmetric encoder named Darknet26-AS is proposed, which performs down-sampling only along the horizontal direction to preserve vertical resolution. The decoder adopts asymmetric transposed convolution. Log-smoothed weighted cross-entropy loss is utilized to mitigate class imbalance. After segmentation, spherical back-projection maps predicted labels from 2D images back to the original 3D point clouds.

### 3. 图像域高斯距离加权投票边界细化后处理模块

在球面图像二维空间做边界优化；滑动窗口选取邻域，筛选 k 个三维最近邻，基于欧氏距离高斯加权投票；网格搜索完成超参数调优；修正边界误分割，提升细长圆柱目标轮廓效果。

**Post-processing module for boundary refinement with Gaussian distance-weighted voting in image domain** — Boundary optimization is conducted in the 2D spherical-image space. A sliding window is used to select neighboring pixels, then k 3D nearest neighbors are filtered for Gaussian-weighted voting based on Euclidean distance. Hyper-parameters are tuned via grid search. This module corrects mis-segmentation at object boundaries and improves contour quality for slender cylindrical targets.

## 采集场景信息 / Orchard Scene Information

- 数据采集于江苏省句容市后白镇薄壳山核桃标准化种植基地（31.88°N, 119.15°E）。
- 果园采用行距 6 m × 株距 4 m 规范化密植模式；果树枝下高 0.7–1.4 m，胸径 10–25 cm。
- 地头两侧由栅栏、灌木构成边界；当机器人行驶到果园末行作业路径时，右侧灌木与斜坡共同构成作业区域边界。
- 场景中包含行人、电线杆、线缆、灌木、栅栏、斜坡等典型障碍物。
- 采集时间为夏季晴天下午。

* Data were collected at a standardized pecan orchard in Houbai Town, Jurong, Jiangsu Province (31.88°N, 119.15°E).
* The orchard adopts a dense-planting pattern with row spacing 6 m × tree spacing 4 m. The trunk base height ranges from 0.7–1.4 m, and diameter at breast height ranges from 10–25 cm.
* Fences and shrubs form the boundaries on both sides of the orchard headland. When the robot travels along the last row path, shrubs and slopes on the right side jointly define the operation boundary.
* Typical objects in this scene include pedestrians, utility poles, power lines, shrubs, fences and slopes.
* Field acquisition was conducted on sunny afternoons in summer.

## 数据集 / Dataset

| 项目 | 说明 |
| :--- | :--- |
| 传感器 | Velodyne **VLP-16**（16 线，垂直 FOV −15° ~ +15°） |
| 采集平台 | 轮趣科技阿克曼转向移动机器人 |
| 处理流程 | 前向半球 ROI（半径 10 m）→ CloudCompare 逐点标注 → 转 KITTI 格式 → 按序列 8:1:1 划分 |
| 数据增强 | 少数类帧复制 + 在线随机旋转（±5°，训练时启用） |
| 序列数 | **20**（`00`–`19`） |
| 帧数 | **1 779** 帧标注点云 |
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

> 地面 + 树冠两类合计超过 80 %，树干仅占 4.5 %，**类别严重不平衡** —— 这正是本项目引入对数平滑加权交叉熵损失（log-smoothed weighted cross-entropy）的原因。

### 数据划分（序列级划分，避免同一场景泄漏到不同子集）

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

## 标注样例 / Annotation Samples

果园点云标注样例可视化：2 个行间，1 个地头

![Orchard point cloud annotation samples](pics/annotation_samples.png)

*（a）（b）行间场景，（c）地头场景，（d）局部细节。图例：树干 / 树冠 / 地面 / 人员 / 栅栏 / 电线杆 / 线缆 / 斜坡 / 灌木*

## 实验结果 / Experimental Results

本模型在本开源果园 LiDAR 测试集上完成 9 类目标语义分割评估。实验硬件环境：Intel i9-12900K + NVIDIA RTX 3090 Ti；单帧耗时包含球面投影、网络推理、边界细化、逆投影完整流程。

| 任务 | 评价指标 | 结果 |
| :--- | :--- | ---: |
| 果园 9 类全场景点云语义分割 | mIoU | **89.9 %** |
| 果园 9 类全场景点云语义分割 | 单帧完整流水线推理耗时 | **12.2 ms** |
| 果园 9 类全场景点云语义分割 | FPS | **82.0** |

模型参数量：**3.8 M**

### 各类别 IoU 详细结果 / Per-class IoU

| 类别 ID | 名称 | 中文 | IoU (%) |
| :-: | :--- | :--- | ---: |
| 1 | `tree_trunk` | 树干 | 82.9 |
| 2 | `pedestrian` | 人员 | 95.3 |
| 3 | `pole` | 电线杆 | 89.1 |
| 4 | `pole_cable` | 线缆 | 74.2 |
| 5 | `fence` | 栅栏 | 78.5 |
| 6 | `slope` | 斜坡 | 97.8 |
| 7 | `shrub` | 灌木 | 95.5 |
| 8 | `ground` | 地面 | 98.8 |
| 9 | `crown` | 树冠 | 97.1 |

> The proposed model is evaluated for 9-class semantic segmentation on the open-source orchard LiDAR test set. Hardware configuration: Intel i9-12900K + NVIDIA RTX 3090 Ti. The per-frame latency covers the full pipeline including spherical projection, network inference, boundary refinement and back-projection.

## 环境依赖

实验环境为 Python 3.7 / CUDA 11.1：

```bash
pip install -r requirements.txt
```

主要依赖：`torch==1.8.0+cu111`、`torchvision==0.9.0+cu111`、`numpy`、`PyYAML`、`scipy`、`matplotlib`、`opencv-python`。

> 若使用较新的 GPU/驱动，可按本机 CUDA 版本替换 torch 版本，代码逻辑无需改动。

## 使用

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
- `-ac` 网络结构配置（`config/arch/darknet26.yaml`，Darknet26-AS 非对称编码—解码，球面图像 900×16）
- `-dc` 类别定义与划分（`config/labels/semantic-kitti.yaml`）
- `-l` 日志与权重输出目录（默认 `~/logs/日期-时间`）

### 启用边界细化后处理

后处理模块由 `config/arch/darknet26.yaml` 的 `post.KNN.use` 控制，默认关闭。启用后推理时对球面图像边界做高斯距离加权投票细化：

```yaml
post:
  KNN:
    use: True
    params:
      knn: 5         # 三维最近邻个数 k
      sigma: 1.0     # 高斯核带宽
      cutoff: 1.0    # 距离截断阈值
```

实现见 `train/tasks/semantic/postproc/KNN.py`（滑动窗口 3×15，基于三维欧氏距离的高斯加权投票）。

### 推理

```bash
python infer.py \
  -d ./dataset \
  -m ~/logs/orchard_seg \
  -l ~/logs/orchard_pred
```

预测结果保存在 `-l` 目录下，每个序列的 `predictions/*.label`。

### 评估（mIoU）

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

## 仓库结构

```
lidar-bonnetal/
├── pics/annotation_samples.png      # 标注样例可视化
├── train/                          # 训练框架（基于 lidar-bonnetal 改造）
│   ├── backbones/                  # DarkNet 主干（Darknet26-AS 非对称编码器）
│   ├── common/                     # 球面投影、Laserscan、可视化等通用工具
│   └── tasks/semantic/
│       ├── config/
│       │   ├── arch/darknet26.yaml         # 网络与训练超参（球面图像 900×16）
│       │   └── labels/semantic-kitti.yaml  # 9 类定义、点云占比、序列划分
│       ├── dataset/
│       │   ├── kitti/parser.py             # 数据读取、在线增强与球面投影
│       │   └── sequences/00..19/           # 标注数据集（1779 帧）
│       ├── decoders/                       # 非对称转置卷积解码器
│       ├── modules/                        # trainer / segmentator / ioueval / user
│       ├── postproc/KNN.py                 # 边界细化后处理（高斯加权投票）
│       ├── train.py                        # 训练入口
│       ├── infer.py                        # 推理入口
│       ├── evaluate_iou.py                 # 评估入口
│       └── visualize.py                    # 可视化入口
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## 引用 / Citation

若本仓库的代码或数据集对您的研究有帮助，请引用：

> [1]. Li Q\*, Jing S. Real-time semantic segmentation of orchard LiDAR point clouds with a spherical-image convolutional neural network[J/OL]. *Computers and Electronics in Agriculture*, 2027, 258: 112634. <https://doi.org/10.1016/j.compag.2026.112634>

```bibtex
@article{li2027realtime,
  title   = {Real-time semantic segmentation of orchard LiDAR point clouds
             with a spherical-image convolutional neural network},
  author  = {Li, Q. and Jing, S.},
  journal = {Computers and Electronics in Agriculture},
  year    = {2027},
  volume  = {258},
  pages   = {112634},
  doi     = {10.1016/j.compag.2026.112634}
}
```

## 许可证与致谢

- 本仓库代码基于开源框架 [lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal)（MIT License, © 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, University of Bonn）改造，遵循 **MIT License**，原始版权声明见 [LICENSE](LICENSE)。
- 数据集由作者实地采集并逐帧标注，随本仓库发布供学术研究使用；使用数据集时请注明出处。
