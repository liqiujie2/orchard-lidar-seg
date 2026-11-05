# 果园点云语义分割数据集 / Orchard Point Cloud Dataset

面向标准化种植果园场景的 **9 类全场景语义标注点云数据集**，随本仓库开源发布。

## 采集信息

| 项目 | 说明 |
| :--- | :--- |
| 传感器 | Velodyne VLP-16（16 线，垂直 FOV −15° ~ +15°，水平 360°，10 Hz） |
| 采集平台 | 轮趣科技阿克曼转向小车 |
| 采集地点 | 江苏省句容市后白镇，薄壳山核桃（Carya illinoinensis）标准化种植基地 |
| 果树布局 | 行列式种植，株行距统一，行间留有作业通道 |
| ROI | 前向半球，半径 10 m（依据点云水平密度约束推导） |
| 标注工具 | CloudCompare，逐帧人工精细标注（树冠为自动 + 人工结合） |

## 规模

- **20** 个序列（`00`–`19`），共 **1 779** 帧原始标注点云
- 单帧约 **8 525** 点（ROI 内，平均）
- 磁盘占用约 315 MB
- 训练时按旋转/平移增强 ×16 可扩充至 **28 464** 帧

## 类别

| ID | 名称 | 中文 | 占比 |
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

> 树干标注范围：自地面至第一主分枝点的主干部分，主要分枝一并计入树干类。
> 树冠标注：以树干平均高度 + 0.1 m 为下界，自动提取其上方垂直 ROI 内的点并赋类。

## 划分（序列级，杜绝场景泄漏）

| 子集 | 序列 | 帧数 |
| :--- | :--- | ---: |
| Train | `00`–`15` | 1 431 |
| Valid | `16`–`17` | 178 |
| Test | `18`–`19` | 170 |

划分定义见 `../../config/labels/semantic-kitti.yaml` 的 `split` 字段。

## 文件格式

```
sequences/
└── 00/
    ├── velodyne/000000.bin      # float32 × 4 (x, y, z, intensity) = 16 B/点
    └── labels/000000.label      # uint32 × 1 (类别 ID)          =  4 B/点
```

- `.label` 与同名 `.bin` **逐点一一对应**，无额外文件头。
- 读取示例：

```python
import numpy as np

pts   = np.fromfile("sequences/00/velodyne/000000.bin",   dtype=np.float32).reshape(-1, 4)
label = np.fromfile("sequences/00/labels/000000.label",   dtype=np.uint32)

assert pts.shape[0] == label.shape[0]
```

## 使用条款

本数据集由作者实地采集与标注，随本仓库发布，**仅供学术研究使用**。发表成果时请注明数据集出处。
