# LightMocap 重构需求文档

> **项目定位：** LightMocap 是从 EasyMocap 重构而来的独立子项目，放在仓库的 `lightmocap/` 目录中。
> 原始的 `easymocap/`、`myeasymocap/` 等目录保持不动，作为 V1 参考实现随时可对照查阅。

## 设计原则

在展开细节之前，先明确 LightMocap 重构的 **四项核心原则**：

1. **算法精度不降级** — LightMocap 是工程现代化改造，不是算法重写。所有三角化、SMPL 拟合、损失函数、优化器必须与 EasyMocap V1 保持数值一致，输出结果的精度不得降低。
2. **渐进式交付** — 第一阶段只完成 MV1P（多视图单人）完整流水线，MVMP（多视图多人）延后。代码架构需提前为 MVMP 预留接口。
3. **技术栈现代化** — uv 管理环境与依赖、pyproject.toml 构建、MVP 阶段使用简化版 Hydra 配置、NPZ 模型格式、类型注解、Ruff 代码风格。
4. **做减法** — 移除 Mirror（镜面模式）、Neural Body（NeRF 新视图合成）、Monocular（单目）、Realtime（Socket 实时可视化）等非核心功能。LightMocap 只聚焦一个场景：**多视图 RGB 图像序列 → 人体 SMPL-X 参数**。

---

## 一、现有项目技术分析

### 0. 当前开发环境与测试数据约束

为了让 LightMocap 在当前机器上可直接开发与验证，先固定以下环境前提：

- 当前 CUDA 版本：`11.8`
- 如需安装 PyTorch，优先安装 **cu118** 对应版本
- Python 环境与依赖管理统一使用 `uv`
- 国内下载统一优先使用清华源加速

#### 0.1 PyTorch 安装约束

LightMocap 文档中的 PyTorch 依赖虽然写为通用版本范围，但在当前开发机上，实际安装策略应明确为 **CUDA 11.8 对应的官方 wheel**。

推荐安装方式示例：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

如果后续需要在 `pyproject.toml` 或安装文档中给出示例，应优先给出 cu118 版本说明，避免默认安装 CPU wheel 或错误的 CUDA 版本。

#### 0.2 uv 与清华源配置约束

LightMocap 默认使用 `uv` 管理虚拟环境、锁文件与依赖安装。为适配国内网络环境，文档中应加入清华源配置示例。

推荐方式：

```bash
# 配置 PyPI 镜像
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 如需安装 PyTorch，则单独使用官方 cu118 源
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其他依赖继续走清华源
uv sync
```

说明：
- 常规 Python 包优先走清华 PyPI 镜像
- PyTorch 仍建议走官方 `cu118` 源，避免镜像源中 wheel 不完整或 CUDA 版本不匹配

#### 0.3 当前测试数据集：`CoreView_377`

当前仓库内已有测试数据集：`/home/ymj/code/python/EasyMocap/CoreView_377`

根据现有目录结构，这个数据集至少包含：

| 路径 / 文件 | 观察结果 | 备注 |
|-------------|----------|------|
| `Camera_B1` ~ `Camera_B23` | 23 个相机图像目录 | 多视图输入主数据 |
| `intri.yml` / `extri.yml` | 完整存在 | 23 个相机的内外参与畸变参数 |
| `keypoints2d/` | 23 个相机目录 | 看起来是 OpenPose 风格的 2D 关键点 JSON |
| `annots/` | 仅 4 个相机目录 | 可能是部分转换后的内部 annotation |
| `mask/` / `mask_cihp/` | 23 个相机目录 | 人体分割相关数据，MVP 可暂不使用 |
| `match_info.json` | 存在 | 记录原始图像文件名与时序映射 |
| `annots.npy` / `annots_python2.npy` | 存在 | 由 `get_annots.py` 汇总出的整体标注数据 |
| `params/` / `vertices/` / `new_params/` / `new_vertices/` | 当前为空 | 可能用于参数或网格结果缓存 |

从样本文件看：

- `keypoints2d/Camera_B11/000003_keypoints.json` 是 **OpenPose 官方输出格式**，包含：
  - `pose_keypoints_2d`
  - `face_keypoints_2d`
  - `hand_left_keypoints_2d`
  - `hand_right_keypoints_2d`
- `get_annots.py` 会将 23 个相机的：
  - 相机参数
  - 图像路径
  - 2D 关键点
  聚合为 `annots.npy`

这说明 `CoreView_377` 非常适合作为 LightMocap 的 MVP 验证集：

- 可以直接复用现有 pinhole 相机 YAML
- 可以直接对照 OpenPose 风格 2D 标注
- 可以用于验证新 detector adapter 是否能产出兼容 V1 的 canonical annotation

### 1. Pipeline 整体架构（LightMocap 范围）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据准备阶段                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  多视图图像序列 + 相机参数 → 2D 关键点检测                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           三角化重建阶段（MV1P / MVMP 分歧点）                │
├─────────────────────────────────────────────────────────────────────────────┤
│  MV1P: 视图间一一映射 → 直接三角化 → 3D 关键点                                │
│  MVMP: 亲和矩阵 + SVT + 关联 → 三角化 → 3D 关键点 + ID 跟踪                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SMPL-X 拟合阶段                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  形状优化 → 姿态优化 → 手部/面部/表情优化（可选）                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出与可视化                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  SMPL-X 参数导出 (JSON) → 3D 渲染 → 重投影叠加                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**LightMocap 范围声明：**
- ✅ 多视图图片序列输入
- ✅ MV1P 完整流水线（Phase 1）
- ✅ MVMP 完整流水线（后续 Phase，架构预留接口）
- ✅ SMPL-X 模型（唯一支持的体模型）
- ❌ Mirror 镜面模式 — 已移除
- ❌ Neural Body / NeRF — 已移除
- ❌ 单目视频 — 已移除
- ❌ 实时 Socket 可视化 — 已移除
- ❌ 视频文件输入 — 已移除（仅支持图片序列）

### 2. MV1P 与 MVMP 的关系

#### 共享层（同一套代码）

| 模块 | 说明 |
|------|------|
| 相机参数加载 / 去畸变 | `core/camera/` |
| 2D 检测器 | `detection/` |
| 三角化算法原语 | `core/triangulation/triangulate.py` |
| SMPL-X 模型前向传播 | `models/smplx.py` |
| 优化器 (L-BFGS) / 损失函数 | `core/fitting/` |
| 可视化 (渲染 / 重投影) | `viz/` |

#### 分歧层（不同策略）

| 阶段 | MV1P | MVMP |
|------|------|------|
| 检测 | 单 bbox（选置信度最高） | 多 bbox（面积过滤） |
| 关联 | 不需要——直接三角化 | `affinity` → `SVT` → `associate` → `criterion` |
| 身份管理 | 不需要 | `PeopleGroup` + `TrackBase` 跨帧维护 |
| SMPL-X 拟合 | 单模型一次拟合 | 逐人循环拟合 |

#### MV1P 与 MVMP 的共存架构设计

LightMocap 采用 **统一接口 + 模式分发** 的设计：

```
                    ┌──────────────────────┐
                    │    MoCapPipeline     │  ← 统一入口
                    │  mode: "single" |    │
                    │  mode: "multi"       │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼──────┐ ┌───▼────┐  ┌──────▼──────┐
         │  SingleMode  │ │Shared  │  │  MultiMode   │
         │  (MV1P)      │ │Core    │  │  (MVMP)      │
         │              │ │Modules │  │              │
         │ - detect     │ │        │  │ - detect     │
         │ - triangulate│ │        │  │ - match      │
         │ - fit        │ │        │  │ - track      │
         │              │ │        │  │ - fit_each   │
         └──────────────┘ └────────┘  └──────────────┘
```

- `MoCapPipeline` 是顶层编排类，通过 `mode` 参数选择单人/多人策略
- 共享核心模块（相机、三角化、SMPL-X、拟合）只有一份实现
- `SingleMode` 和 `MultiMode` 分别封装各自的关联 + 跟踪逻辑
- Phase 1 只实现 `SingleMode`，但 `MultiMode` 的接口契约（Abstract Base Class）提前定义好

### 3. 各模块技术选型详解

#### 3.1 2D 姿态检测模块

**EasyMocap V1 现状：**

| 检测器 | 技术栈 | 依赖模型 | 安装复杂度 | 问题 |
|--------|--------|----------|------------|------|
| OpenPose | 外部 C++ 程序 | 官方编译版本 | 高 | 需单独安装，已停止维护 |
| MediaPipe 0.10.0 | Python 包 | 内置模型 | 低 | 版本较旧 |
| YOLOv4 + HRNet | PyTorch | 外部权重文件 | 中 | Darknet 格式老旧，模型难获取 |

**LightMocap 更新：**

| 检测器 | 技术栈 | 安装方式 | 特点 |
|--------|--------|----------|------|
| **rtmlib** | rtmlib | `uv add rtmlib` | 封装完整 pose pipeline，无需自行拼装 detector + pose estimator |
| **MediaPipe (最新)** | mediapipe | `uv add mediapipe` | 轻量级，无需 GPU |

**检测内容支持：**
- 身体姿态（必需）
- 手部关键点（支持）
- 面部关键点（支持）

#### 3.1.1 EasyMocap V1 中不同 detector 的输出与统一方式

EasyMocap V1 的一个关键工程特点是：**检测器前端可以不同，但进入三角化和拟合之前，都会被规整到统一的 annotation JSON 结构**。LightMocap 保留这一设计思想，只替换老旧的 2D 检测前端。

##### V1 的统一 annotation 结构

V1 最终写盘的 2D 结果统一为：

```json
{
  "filename": "images/01/000000.jpg",
  "height": 1080,
  "width": 1920,
  "annots": [
    {
      "personID": 0,
      "bbox": [l, t, r, b, conf],
      "keypoints": [[x, y, conf], ...],
      "bbox_handl2d": [l, t, r, b, conf],
      "handl2d": [[x, y, conf], ...],
      "bbox_handr2d": [l, t, r, b, conf],
      "handr2d": [[x, y, conf], ...],
      "bbox_face2d": [l, t, r, b, conf],
      "face2d": [[x, y, conf], ...]
    }
  ],
  "isKeyframe": false
}
```

其中：
- `keypoints` 默认使用 `body25`
- `handl2d` / `handr2d` 为 21 点手部关键点
- `face2d` 在读取阶段会裁成 51 个与拟合兼容的面部点
- 下游 `read_annot()` 会把这些字段再拼接成 `body25` / `bodyhand` / `bodyhandface`

##### V1 各检测器的原始输出与转换方式

| 检测器 | 原始输出 | V1 中间处理 | 最终统一格式 |
|--------|----------|-------------|--------------|
| OpenPose | OpenPose 官方 JSON，含 `pose_keypoints_2d`、`hand_left_keypoints_2d`、`hand_right_keypoints_2d`、`face_keypoints_2d` | 读取官方 JSON，生成 bbox，字段重命名为 `keypoints` / `handl2d` / `handr2d` / `face2d` | V1 annotation JSON |
| YOLOv4 + HRNet | YOLO bbox + HRNet 17 点人体关键点 | 先做人框检测，再做人体姿态，保存为 bbox + `keypoints`；多人按面积排序重排 ID | V1 annotation JSON |
| MediaPipe | 33 点 body / 21 点 hand / 468 点 face | 可选映射到 OpenPose body25；手脸保留 MediaPipe 原始点数，再存成统一字段名 | V1 annotation JSON |

##### V1 中具体的转换逻辑

1. **OpenPose 路径**
   - `scripts/preprocess/extract_video.py` 或 `openpose_wrapper.py` 调用 OpenPose 二进制
   - 读取 `pose_keypoints_2d` 等官方字段
   - 通过关键点反算 `bbox`
   - 重命名为项目内部字段：`keypoints`、`handl2d`、`handr2d`、`face2d`

2. **YOLOv4 + HRNet 路径**
   - YOLOv4 先输出人框 `bbox`
   - HRNet 在 `bbox` crop 上输出人体关键点
   - 写入统一 annotation，人体关键点字段仍叫 `keypoints`
   - 该路径主要负责 body，不直接覆盖手脸

3. **MediaPipe 路径**
   - body 原始输出为 33 点
   - 若 `to_openpose=True`，则映射到 body25：补 Neck / MidHip，重排索引
   - hand 输出 21 点，face 输出 468 点
   - 最终也写入统一 annotation 结构

##### V1 下游如何消费这些结果

下游不直接关心 detector 类型，而只依赖统一 annotation 结构：

| 下游阶段 | 读取内容 | 约束 |
|----------|----------|------|
| 三角化 | `keypoints` 或拼接后的 `bodyhand` / `bodyhandface` | 需要固定关节点顺序 |
| SMPL-X 拟合 | `keypoints3d` + 多视图 `keypoints2d` | 关键点语义必须稳定 |
| 可视化 | `bbox`、`keypoints`、`handl2d`、`handr2d`、`face2d` | 仅依赖统一字段名 |

##### LightMocap 的继承策略

LightMocap 不改变 V1 后端拟合算法，因此新 detector 必须先转换为 **与 V1 一致的 canonical annotation 结构**，再进入三角化和拟合模块。

这意味着：
- `rtmlib`、`MediaPipe` 的原始输出不能直接喂给后端
- 必须增加一层 `adapter / converter`
- 该层的职责是把不同 detector 的原始 keypoint schema 转成 V1 使用的 `body25` / `bodyhandface` 语义

#### 3.1.2 新检测器的预期原始输出格式

| 检测器 | 原始输出格式 | LightMocap 适配后格式 |
|--------|--------------|----------------------|
| rtmlib | 完整检测 + pose 结果，通常输出每人关键点数组与 bbox | 转为 V1 annotation JSON 的 `bbox` + `keypoints` |
| MediaPipe | body/hand/face landmarks 对象或数组 | 转为 V1 annotation JSON 的 `keypoints` / `handl2d` / `handr2d` / `face2d` |

##### 新旧 detector 统一后的目标格式

无论前端来自 OpenPose、YOLOv4+HRNet、rtmlib 还是 MediaPipe，进入后端前都必须满足：

- body 使用固定的 canonical 语义
- bbox 统一为 `[l, t, r, b, conf]`
- 2D 点统一为 `[[x, y, conf], ...]`
- 手部和面部作为可选字段存在
- 写盘格式与 V1 annotation JSON 保持兼容

#### 3.2 相机参数模块

**支持的相机模型：**

| 模型 | 参数 | 说明 |
|------|------|------|
| **Pinhole + OpenCV distortion** | `[k1, k2, p1, p2, k3]` 或 `[k1, k2, p1, p2]` | LightMocap 唯一支持的相机模型 |

**明确限制：**
- LightMocap **不支持 fisheye 相机模型**
- LightMocap **不修改** EasyMocap V1 的 `intri.yml` / `extri.yml` 结构
- 原因是需要直接复用现有数据集与测试集，保证 V1/V2 可直接对照

**相机参数文件格式（YAML）：**

```yaml
# intri.yml - 内参
%YAML:1.0
---
names:
  - "01"
  - "02"
K_01: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_01: !!opencv-matrix
  rows: 1
  cols: 5
  dt: d
  data: [k1, k2, p1, p2, k3]
H_01: 1080
W_01: 1920

# extri.yml - 外参
%YAML:1.0
---
names:
  - "01"
  - "02"
R_01: !!opencv-matrix
  rows: 3
  cols: 1
  dt: d
  data: [rx, ry, rz]
T_01: !!opencv-matrix
  rows: 3
  cols: 1
  dt: d
  data: [tx, ty, tz]
```

**参数说明：**

| 参数 | 形状 | 说明 |
|------|------|------|
| `K_<cam>` | (3, 3) | 内参矩阵 |
| `dist_<cam>` | (1, 5) 或 (1, 4) | 畸变系数 |
| `R_<cam>` | (3, 1) | Rodrigues 旋转向量 |
| `T_<cam>` | (3, 1) | 平移向量 |
| `H_<cam>` | int | 图像高度（可选） |
| `W_<cam>` | int | 图像宽度（可选） |

#### 3.3 SMPL-X 模型模块

**LightMocap 方案：** 仅支持 SMPL-X，仅使用 NPZ 格式，不提供 PKL 支持，不提供转换工具。

这里的核心约束不是改写后续算法，而是**仅替换模型文件的输入组织形式**。V1 中 SMPL-X 的 LBS、手 PCA、expression、优化损失和参数更新逻辑全部保留；改动只发生在“模型数据如何读入内存”这一层。

| 模型 | 文件格式 | 加载方式 | 说明 |
|------|----------|----------|------|
| **SMPL-X** | `.npz` | `numpy.load()` + V1 等价字段整理 | 唯一支持的体模型 |

用户需从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de) 注册下载 NPZ 格式模型文件。

**NPZ 模型结构：**

| 键名 | 形状 | 说明 |
|------|------|------|
| `f` | (F, 3) | 面索引 |
| `v_template` | (V, 3) | 模板顶点 |
| `J_regressor` | (J, V) | 关节回归器（稀疏矩阵） |
| `weights` | (V, J) | LBS 蒙皮权重 |
| `posedirs` | (V, 3, P) | 姿态混合形状 |
| `shapedirs` | (V, 3, S) | 形状混合形状 |
| `kintree_table` | (2, J) | 骨骼树 |
| `hands_meanl/r` | (45,) | 手部平均姿态 |

**实现原则：**
- 保留 V1 的后续算法逻辑不变
- NPZ 与 PKL 只被视为不同的数据组织形式
- 加载后需要整理成与 V1 后端等价的内存字段，再交给同样的前向与拟合逻辑

#### 3.4 优化拟合模块

**EasyMocap V1 现状：**

| 组件 | 技术选型 | 问题 |
|------|----------|------|
| 配置管理 | YAML + yacs | yacs 已停止维护 |
| 优化器 | PyTorch L-BFGS | 无问题 |

**LightMocap 更新：**

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 配置管理 | Hydra（MVP 简化使用） | 先满足单流程配置，不追求 V1 复杂 workflow 编排 |
| 优化器 | PyTorch L-BFGS（保持 V1 实现） | **算法不变，确保数值一致性** |

#### 3.5 多视图匹配模块（MVMP，后续 Phase）

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| SVT/ALS 算法 | C++ + Eigen | 高性能矩阵分解，Phase 1 不构建 |
| Python 绑定 | pybind11 | 跨平台支持 |
| 构建系统 | scikit-build-core | 现代 Python 构建工具 |

#### 3.6 可视化模块

**LightMocap 方案：**
- 仅支持图片序列输出
- 移除系统级 `ffmpeg` 依赖
- 如需 GIF/MP4 导出，允许通过 `imageio[ffmpeg]` 安装 Python 打包版 ffmpeg 运行时
- pyrender 用于 3D 渲染和重投影叠加（可选依赖）

### 4. 依赖管理（uv）

LightMocap 使用 [uv](https://github.com/astral-sh/uv) 作为 Python 包管理器和环境管理工具。

#### 4.1 核心依赖

```toml
[project]
name = "lightmocap"
version = "1.0.0"
requires-python = ">=3.11"
description = "Light Human Motion Capture Toolbox"
license = {text = "BSD-3-Clause"}

dependencies = [
    "torch>=2.0.0",        # 当前开发机安装时使用 cu118 wheel
    "torchvision>=0.15.0", # 当前开发机安装时使用 cu118 wheel
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "opencv-python>=4.8.0",
    "pyyaml>=6.0",
    "hydra-core>=1.3.0",
    "tqdm>=4.65.0",
    "joblib>=1.3.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
detection = [
    "mediapipe>=0.10.0",
    "rtmlib>=1.2.0",
]
viz = [
    "pyrender>=0.1.45",
    "open3d>=0.17.0",
    "matplotlib>=3.7.0",
    "imageio>=2.31.0",
    "imageio[ffmpeg]>=2.31.0",  # 可选：提供 Python 打包的 ffmpeg runtime，而非系统依赖
]
native = [
    "easymocap-native>=2.0.0",  # C++ 扩展，预编译 wheel
]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[project.scripts]
lmc = "lightmocap.cli.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

#### 4.2 uv 工作流

```bash
# 在仓库根目录创建 lightmocap 项目
mkdir lightmocap && cd lightmocap

# 初始化 pyproject.toml
uv init --no-readme --no-license --name lightmocap

# 配置国内镜像
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 先安装 PyTorch cu118
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
uv sync

# 安装可选依赖
uv sync --extra detection --extra viz

# 开发模式安装（从 EasyMocap 仓库内引用源码）
uv pip install -e ".[dev]"

# 运行命令
uv run lmc pipeline --config configs/mv1p.yaml
```

补充说明：
- 当前机器的 CUDA 版本为 `11.8`，因此 PyTorch 统一按 `cu118` 安装
- `uv sync` 适合安装普通 Python 包
- `torch` / `torchvision` 建议显式指定官方 cu118 源

#### 4.3 移除的依赖

| 包名 | 移除原因 |
|------|----------|
| yacs | 停止维护，替换为 Hydra |
| chumpy | Python 2/3 兼容问题，NPZ 格式不再需要 |
| pytorch-lightning | 过度设计，L-BFGS 直接调用即可 |
| tensorboard | 不需要训练监控 |
| setuptools | 使用 hatchling 构建 |
| OpenPose | 替换为 rtmlib / MediaPipe |
| YOLOv4 Darknet | 不再作为 LightMocap 的目标替代前端 |

---

## 二、重构需求规格

### 1. 项目结构

LightMocap 的代码存放在仓库根目录下的 `lightmocap/` 目录中，与原始 `easymocap/` 并列存在，互不干扰。

```
EasyMocap/                        # 仓库根目录
├── easymocap/                    # [V1 原版，不动，供参考]
├── myeasymocap/                  # [V1 原版，不动，供参考]
├── apps/                         # [V1 原版，不动，供参考]
├── config/                       # [V1 原版，不动，供参考]
├── scripts/                      # [V1 原版，不动，供参考]
├── library/                      # [V1 原版，不动，供参考]
│
├── lightmocap/                   # ====== [重构后的新项目] ======
│   ├── pyproject.toml            # uv 兼容的项目配置
│   ├── uv.lock                   # uv 锁定文件
│   ├── src/lightmocap/
│   │   ├── __init__.py
│   │   ├── core/                 # 核心算法（MV1P / MVMP 共享）
│   │   │   ├── fitting/          # SMPL-X 拟合
│   │   │   │   ├── optimizer.py  # L-BFGS 优化器（保持 V1 算法一致）
│   │   │   │   ├── losses.py     # 损失函数（保持 V1 算法一致）
│   │   │   │   └── pipeline.py   # 拟合流程编排
│   │   │   ├── triangulation/    # 三角化
│   │   │   │   ├── triangulate.py # 三角化算法（保持 V1 算法一致）
│   │   │   │   └── match.py      # 多视图匹配（MVMP，后续 Phase）
│   │   │   └── camera/           # 相机工具
│   │   │       ├── params.py     # 相机参数类
│   │   │       └── undistort.py  # 去畸变
│   │   ├── models/               # 身体模型
│   │   │   ├── __init__.py
│   │   │   ├── smplx.py          # SMPL-X（唯一支持的模型）
│   │   │   ├── lbs.py            # 线性蒙皮算法
│   │   │   └── io.py             # NPZ 加载
│   │   ├── detection/            # 2D 检测
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # 检测器基类
│   │   │   ├── rtmlib.py         # rtmlib 封装
│   │   │   └── mediapipe.py      # MediaPipe
│   │   ├── data/                 # 数据处理
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py        # 数据集类
│   │   │   ├── camera.py         # 相机参数 YAML 读写
│   │   │   └── skeleton.py       # 骨架定义
│   │   ├── viz/                  # 可视化
│   │   │   ├── __init__.py
│   │   │   ├── render.py         # 3D 渲染
│   │   │   └── plot.py           # 2D 绘图
│   │   ├── modes/                # 模式分发（MV1P / MVMP 共存架构）
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Abstract Base Class（定义模式契约）
│   │   │   └── single.py         # MV1P 模式（Phase 1）
│   │   │   └── # multi.py        # MVMP 模式（后续 Phase，暂不实现）
│   │   ├── pipeline/             # 顶层流水线
│   │   │   ├── __init__.py
│   │   │   └── mocap.py          # MoCapPipeline（统一入口）
│   │   ├── config/               # Hydra 配置
│   │   │   ├── config.yaml       # 主配置
│   │   │   ├── mode/             # 模式配置
│   │   │   │   ├── single.yaml   # MV1P 配置
│   │   │   │   └── multi.yaml    # MVMP 配置模板（预留）
│   │   │   ├── model/            # SMPL-X 配置
│   │   │   ├── detection/        # 检测器配置
│   │   │   └── fitting/          # 拟合配置
│   │   └── cli/                  # 命令行工具
│   │       ├── __init__.py
│   │       ├── main.py           # lmc 入口
│   │       └── download_models.py # 检测模型下载
│   ├── native/                   # C++ 扩展（MVMP，后续 Phase）
│   │   ├── pymatch/
│   │   │   ├── src/
│   │   │   ├── include/
│   │   │   └── CMakeLists.txt
│   │   └── pyproject.toml        # scikit-build-core 构建配置
│   ├── configs/                  # 示例配置
│   │   ├── mv1p.yaml             # MV1P 一键流程
│   │   └── ...
│   ├── tests/                    # 测试
│   └── docs/                     # 文档
│
├── data/                         # 共享数据目录（SMPL-X 模型等）
└── docs/                         # 文档目录（含本重构文档）
```

### 2. Python API 设计

```python
# ============================================================
# MV1P — 多视图单人
# ============================================================
from lightmocap import MoCapPipeline, Camera

# 加载相机参数
camera = Camera.from_yaml("intri.yml", "extri.yml")

# 创建 MV1P 流水线
mocap = MoCapPipeline(
    mode="single",              # MV1P 模式
    camera=camera,
    detector="rtmlib",          # 可选: "mediapipe"
    model_path="data/smplx/",   # SMPL-X NPZ 模型目录
    gender="neutral",           # 可选: "neutral", "male", "female"
)

# 处理单帧
images = {cam: f"images/{cam}/000000.jpg" for cam in camera.names}
result = mocap.process_frame(images, frame_id=0)

# 获取结果
keypoints_3d = result.keypoints3d      # (N, J, 4)  [x, y, z, confidence]
smplx_params = result.smplx_params     # dict: poses, shapes, expression, ...
vertices = result.vertices             # (V, 3)

# 保存结果
result.save("output/000000.json")

# 批量处理
mocap.process_directory(
    image_dir="images/",
    output_dir="output/",
    start=0,
    end=1000,
)

# ============================================================
# 带 Hydra 配置的使用
# ============================================================
from lightmocap import MoCapPipeline
from omegaconf import DictConfig

config = DictConfig({
    "mode": "single",
    "detector": {"type": "rtmlib", "solution": "wholebody", "mode": "lightweight"},
    "fitting": {"opt_shape": True, "opt_pose": True},
    "model": {
        "model_path": "data/smplx/",
        "gender": "neutral",
        "num_shape": 10,
        "num_expression": 10,
        "use_hands": True,
        "use_face": True,
    },
})
mocap = MoCapPipeline.from_config(config, camera=camera)

# ============================================================
# MVMP — 多视图多人（后续 Phase，接口预留）
# ============================================================
# mocap = MoCapPipeline(
#     mode="multi",              # MVMP 模式
#     camera=camera,
#     detector="rtmlib",
#     model_path="data/smplx/",
#     gender="neutral",
#     # MVMP 专属参数
#     match={
#         "max_dist": 0.5,       # 极线距离阈值
#         "use_svt": True,       # 使用 SVT 低秩补全
#     },
#     track={
#         "max_frames": 10,      # 跟踪窗口大小
#     },
# )
```

### 3. 命令行 API

```bash
# 下载检测模型（rtmlib 权重）
lmc download-models --detector rtmlib

# 2D 关键点检测
lmc detect \
    --images images/ \
    --output annots/ \
    --detector rtmlib \
    --hand --face

# 三角化（MV1P）
lmc triangulate \
    --images images/ \
    --annots annots/ \
    --cameras cameras/ \
    --output keypoints3d/

# SMPL-X 拟合（MV1P）
lmc fit \
    --images images/ \
    --keypoints3d keypoints3d/ \
    --cameras cameras/ \
    --output smplx/ \
    --model-path data/smplx/ \
    --gender neutral

# 一键流程（MV1P）
lmc pipeline \
    --images images/ \
    --cameras cameras/ \
    --output output/ \
    --config configs/mv1p.yaml

# 可视化
lmc visualize \
    --images images/ \
    --smplx smplx/ \
    --cameras cameras/ \
    --output vis/
```

### 4. Hydra 配置示例（MVP）

MVP 阶段只实现**单一流水线配置加载**，不追求 V1 的复杂 workflow 编排、alias 展开和多层实验组合能力。Hydra 在此阶段主要用于：

- 配置文件分组管理
- 命令行 override
- 区分 detector / model / fitting 的基本配置

```yaml
# configs/config.yaml — 主配置
defaults:
  - mode: single            # MV1P
  - model: smplx
  - detector: rtmlib
  - fitting: default

# 相机参数
cameras:
  intri: cameras/intri.yml
  extri: cameras/extri.yml

# 数据路径
data:
  images: images/
  output: output/

# 处理范围
frame:
  start: 0
  end: -1                   # -1 表示处理到最后一帧
  step: 1
```

```yaml
# configs/mode/single.yaml — MV1P 配置
type: single
# MV1P 无额外参数
```

```yaml
# configs/mode/multi.yaml — MVMP 配置模板（预留，后续 Phase 实现）
type: multi
match:
  max_dist: 0.5
  use_svt: true
  svt_tol: 1e-3
track:
  window_size: 10
  min_confidence: 0.3
```

```yaml
# configs/model/smplx.yaml — SMPL-X 配置
model_path: data/smplx/
gender: neutral
ext: npz                    # 固定为 npz
num_shape: 10
num_expression: 10
use_hands: true
use_face: true
```

```yaml
# configs/detector/rtmlib.yaml
type: rtmlib
solution: wholebody
mode: lightweight
device: cpu
backend: onnxruntime
```

```yaml
# configs/fitting/default.yaml — 拟合配置（保持 V1 算法参数）
stages:
  - name: shape
    opt_R: true
    opt_T: true
    opt_shape: true
    opt_pose: false
    max_iter: 50
  - name: pose
    opt_R: true
    opt_T: true
    opt_shape: false
    opt_pose: true
    max_iter: 100
  - name: hand
    opt_hand: true
    max_iter: 50
  - name: face
    opt_expression: true
    max_iter: 30

loss_weights:
  k3d: 1000.0
  k2d: 100.0
  smooth_pose: 10.0
  reg_pose: 1.0
```

### 5. C++ 扩展构建（MVMP，后续 Phase）

```toml
# native/pyproject.toml
[project]
name = "easymocap-native"
version = "2.0.0"
requires-python = ">=3.11"

[build-system]
requires = ["scikit-build-core>=0.5.0", "pybind11>=2.11.0"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.version = ">=3.15"
wheel.install-dir = "lightmocap/native"
```

### 6. 算法精度保证

**这是 LightMocap 重构的最高约束。** 以下算法组件必须与 EasyMocap V1 保持数值一致性：

| 算法组件 | V1 实现位置（easymocap/） | LightMocap 要求 |
|----------|---------------------------|-----------------|
| 三角化 (`batch_triangulate`) | `mytools/triangulator.py` | **逐行对比，输出差异 < 1e-6** |
| 迭代三角化 (`iterative_triangulate`) | `mytools/triangulator.py` | **逐行对比，输出差异 < 1e-6** |
| L-BFGS 优化器 | `pyfitting/lbfgs.py` | **直接使用或等价重写** |
| 损失函数 | `pyfitting/lossfactory.py` | **权重、公式完全一致** |
| SMPL-X 前向传播 (LBS) | `bodymodel/lbs.py` | **逐行对比，顶点差异 < 1e-6** |
| SMPL-X NPZ 加载 | `bodymodel/smpl.py` / `bodymodel/smplx.py` | **仅替换输入格式，加载后参数值完全一致** |
| 相机去畸变 | `mytools/camera_utils.py` | **逐像素一致** |
| 多视图匹配 (SVT) | `affinity/matchSVT.py` | **矩阵输出差异 < 1e-6** |

**验证方法：**
1. 使用 EasyMocap V1 的测试数据（图像 + 相机参数），分别运行 V1 和 LightMocap
2. 对比输出 JSON 文件中的数值差异
3. 对比渲染图像的差异（像素级）
4. 所有差异必须 < 1e-6（浮点精度范围内）

### 7. 质量保证

| 方面 | 要求 |
|------|------|
| **类型注解** | 所有公开 API 添加类型注解 |
| **文档** | 每个模块 docstring，包含使用示例 |
| **测试** | Phase 1 覆盖率 ≥ 50%，核心算法模块 ≥ 80% |
| **代码风格** | Ruff 格式化，遵循 PEP 8 |
| **CI/CD** | GitHub Actions 自动测试 |
| **精度回归** | 使用 V1 测试数据进行数值对比测试 |

### 8. 兼容性

| 方面 | 要求 |
|------|------|
| Python | ≥ 3.11 |
| PyTorch | ≥ 2.0 |
| CUDA | 11.8 / 12.1 |
| 操作系统 | Linux（主要）、Windows、macOS |

---

## 三、重构优先级

### Phase 1：MV1P 完整流水线（高优先级）

**目标：交付可用的多视图单人 → SMPL-X 参数管线**

1. **基础设施**
   - 在 `lightmocap/` 目录创建 `pyproject.toml`，配置 uv 依赖管理
   - 搭建 `src/lightmocap/` 项目结构
    - 集成 Hydra 配置系统（MVP 简化版）
    - 实现 NPZ SMPL-X 模型加载

2. **核心算法（保持 V1 精度）**
   - 从 `easymocap/mytools/triangulator.py` 迁移三角化模块
   - 从 `easymocap/mytools/camera_utils.py` 迁移相机去畸变模块
   - 从 `easymocap/pyfitting/` 迁移 L-BFGS 优化器和损失函数
   - 从 `easymocap/bodymodel/` 迁移 SMPL-X 前向传播（LBS）

3. **MV1P 流水线**
   - 实现 `SingleMode`（检测 → 三角化 → SMPL-X 拟合）
   - 实现 `MoCapPipeline(mode="single")` 统一入口
   - 定义 `MultiMode` 的 Abstract Base Class（预留接口）

4. **2D 检测器**
   - 集成 rtmlib（作为完整 pose pipeline，而非仅 RTMPose 单模型）
   - 集成 MediaPipe（可选后备方案）
   - 实现检测模型自动下载

5. **Detector 适配层**
   - 明确 V1 canonical annotation 格式
   - 为 rtmlib / MediaPipe 实现统一 adapter
   - 保证进入三角化与拟合前的数据语义与 V1 保持一致

6. **CLI 工具**
    - `lmc detect` — 2D 检测
    - `lmc triangulate` — 三角化
    - `lmc fit` — SMPL-X 拟合
    - `lmc pipeline` — 一键流程
    - `lmc download-models` — 模型下载

7. **可视化**
    - 3D 渲染（pyrender，可选）
    - 重投影叠加
    - 图片序列输出

8. **测试与精度验证**
    - 核心算法数值对比测试（V1 vs LightMocap）
    - 端到端 MV1P 流水线测试

### Phase 1 具体开发清单

这一节将 Phase 1 进一步整理为**可直接开工的目录、文件和任务清单**。目标不是一次性把所有架构都做完，而是先搭起一个能跑通 `CoreView_377` 的 MVP。

#### 1. Phase 1 交付目标

Phase 1 完成后，应至少满足：

1. 可以在 `lightmocap/` 下通过 `uv` 创建并管理环境
2. 可以在当前机器上安装 `torch` + `torchvision` 的 `cu118` 版本
3. 可以读取 `CoreView_377/intri.yml` 与 `CoreView_377/extri.yml`
4. 可以读取 `CoreView_377/Camera_B*/` 多视图图片序列
5. 可以运行一个 detector 前端，并把输出适配为 V1 canonical annotation
6. 可以完成 MV1P 的三角化
7. 可以完成 SMPL-X 拟合，并输出 JSON
8. 可以使用 `lmc pipeline` 跑通最小端到端流程

#### 2. Phase 1 建议里程碑

| 里程碑 | 目标 | 验收标准 |
|--------|------|----------|
| M1 | 项目骨架可安装 | `uv sync` 成功，`uv run lmc --help` 可用 |
| M2 | 相机和数据可读 | 能读取 `CoreView_377` 的 23 路相机和图像 |
| M3 | detector adapter 打通 | 新 detector 输出可转成 V1 annotation |
| M4 | 三角化打通 | 产出合法的 `keypoints3d/*.json` |
| M5 | SMPL-X 拟合打通 | 产出合法的 `smplx/*.json` |
| M6 | MVP 端到端可跑 | `lmc pipeline` 跑通一个短序列 |

#### 3. 首批必须创建的目录结构

Phase 1 不需要把文档里的所有目录都实现完整，首批只创建最小可运行集合：

```text
lightmocap/
├── pyproject.toml
├── README.md
├── uv.lock
├── src/lightmocap/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── download_models.py
│   ├── config/
│   │   ├── config.yaml
│   │   ├── detector/
│   │   │   ├── rtmlib.yaml
│   │   │   └── mediapipe.yaml
│   │   ├── fitting/
│   │   │   └── default.yaml
│   │   ├── mode/
│   │   │   └── single.yaml
│   │   └── model/
│   │       └── smplx.yaml
│   ├── core/
│   │   ├── camera/
│   │   │   ├── __init__.py
│   │   │   ├── params.py
│   │   │   └── undistort.py
│   │   ├── triangulation/
│   │   │   ├── __init__.py
│   │   │   └── triangulate.py
│   │   └── fitting/
│   │       ├── __init__.py
│   │       ├── optimizer.py
│   │       ├── losses.py
│   │       └── pipeline.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── dataset.py
│   │   ├── io.py
│   │   └── skeleton.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── adapter.py
│   │   ├── rtmlib.py
│   │   └── mediapipe.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── lbs.py
│   │   └── smplx.py
│   ├── modes/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── single.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── mocap.py
│   └── utils/
│       ├── __init__.py
│       ├── paths.py
│       └── typing.py
├── configs/
│   └── mv1p.yaml
└── tests/
    ├── test_camera_io.py
    ├── test_detector_adapter.py
    ├── test_triangulation.py
    ├── test_smplx_loader.py
    └── test_mvp_pipeline.py
```

#### 4. 首批文件职责清单

| 文件 | 作用 | Phase 1 是否必须 |
|------|------|------------------|
| `lightmocap/pyproject.toml` | uv 项目配置、依赖、CLI 入口 | 必须 |
| `src/lightmocap/__init__.py` | 暴露公开 API | 必须 |
| `src/lightmocap/cli/main.py` | `lmc` 命令入口 | 必须 |
| `src/lightmocap/config/config.yaml` | 主配置 | 必须 |
| `src/lightmocap/data/camera.py` | 读取 `intri.yml` / `extri.yml` | 必须 |
| `src/lightmocap/data/dataset.py` | 组织多视图图像序列与帧访问 | 必须 |
| `src/lightmocap/detection/base.py` | detector 抽象接口 | 必须 |
| `src/lightmocap/detection/adapter.py` | 新 detector 到 V1 annotation 的统一转换 | 必须 |
| `src/lightmocap/detection/rtmlib.py` | rtmlib 前端封装 | 建议首批实现 |
| `src/lightmocap/detection/mediapipe.py` | MediaPipe 前端封装 | 可放在首批后半段 |
| `src/lightmocap/core/triangulation/triangulate.py` | V1 三角化迁移 | 必须 |
| `src/lightmocap/models/io.py` | NPZ 模型加载 | 必须 |
| `src/lightmocap/models/lbs.py` | LBS 迁移 | 必须 |
| `src/lightmocap/models/smplx.py` | SMPL-X 前向与参数封装 | 必须 |
| `src/lightmocap/core/fitting/optimizer.py` | L-BFGS 迁移 | 必须 |
| `src/lightmocap/core/fitting/losses.py` | 损失函数迁移 | 必须 |
| `src/lightmocap/core/fitting/pipeline.py` | 拟合流程编排 | 必须 |
| `src/lightmocap/modes/single.py` | MV1P 模式执行器 | 必须 |
| `src/lightmocap/pipeline/mocap.py` | 顶层 pipeline 编排 | 必须 |
| `configs/mv1p.yaml` | MVP 运行配置 | 必须 |
| `tests/test_mvp_pipeline.py` | MVP 端到端验证 | 必须 |

#### 5. 推荐开发顺序

建议严格按下面顺序推进，避免前面接口没定稳就开始写后面的适配：

1. **项目初始化**
   - 创建 `lightmocap/pyproject.toml`
   - 配置 `uv`、清华源说明、`lmc` CLI 入口
   - 保证 `uv run lmc --help` 可执行

2. **数据与相机层**
   - 实现 `data/camera.py`
   - 实现 `data/dataset.py`
   - 在 `CoreView_377` 上验证 23 路相机和帧序列读取

3. **V1 canonical schema 固定**
   - 定义 Phase 1 使用的 annotation 结构
   - 明确 body 的 canonical joint order
   - 明确 `bbox`、`keypoints`、`handl2d`、`handr2d`、`face2d` 的约定

4. **detector 层**
   - 先接入 `rtmlib.py`
   - 用 `adapter.py` 转为统一 annotation
   - 最后再补 `mediapipe.py`

5. **三角化层**
   - 迁移 V1 三角化函数
   - 用统一 annotation 输出测试多视图 3D 关键点

6. **SMPL-X 模型层**
   - 实现 `models/io.py` 的 NPZ 加载
   - 迁移 `lbs.py`
   - 迁移 `smplx.py` 前向逻辑

7. **拟合层**
   - 迁移 `optimizer.py`
   - 迁移 `losses.py`
   - 在 `pipeline.py` 中拼装 shape / pose / hand / face 阶段

8. **MVP pipeline 层**
   - 实现 `modes/single.py`
   - 实现 `pipeline/mocap.py`
   - 实现 `lmc detect` / `lmc triangulate` / `lmc fit` / `lmc pipeline`

9. **测试与回归**
   - 单元测试
   - `CoreView_377` 小样本端到端测试
   - 与 V1 输出对比

#### 6. 首批文件的最小完成标准

| 文件 | 最小完成标准 |
|------|--------------|
| `data/camera.py` | 能无改动读取 V1 YAML，相机名、K、dist、R、T 正确 |
| `data/dataset.py` | 能按帧返回多视图图片路径或图像数据 |
| `detection/adapter.py` | 能把至少一种新 detector 输出转成 V1 annotation |
| `core/triangulation/triangulate.py` | 能从多视图 2D 点输出 3D 点 |
| `models/io.py` | 能加载 SMPL-X NPZ 并返回后续前向所需字段 |
| `models/smplx.py` | 能完成一次前向并输出 joints / vertices |
| `core/fitting/pipeline.py` | 能从 3D keypoints 输出 SMPL-X 参数 |
| `modes/single.py` | 能串起 detect → triangulate → fit |
| `pipeline/mocap.py` | 能暴露 `process_frame()` 和 `process_directory()` |
| `cli/main.py` | 能触发 `lmc pipeline --config configs/mv1p.yaml` |

#### 7. Phase 1 首批实现建议范围

为了确保 MVP 尽快落地，首批实现建议再做一次收缩：

1. 首先只保证 **MV1P**
2. 首先只保证 **body + SMPL-X 主体拟合**
3. `hand` / `face` 可以先保留接口，第二批补齐
4. 首先只接入 **rtmlib** 作为默认 detector
5. `mediapipe` 放在首批后半段
6. 可视化先只做最基本的重投影图片输出

#### 8. 推荐首批任务拆分

| 任务编号 | 任务 | 产出 |
|----------|------|------|
| T1 | 初始化 `lightmocap/` 项目骨架 | `pyproject.toml`、CLI、配置目录 |
| T2 | 实现相机与数据读取 | `camera.py`、`dataset.py` |
| T3 | 定义 canonical annotation schema | `skeleton.py`、`adapter.py` |
| T4 | 接入默认 detector | `rtmlib.py` |
| T5 | 迁移三角化 | `triangulate.py` |
| T6 | 实现 SMPL-X NPZ 加载 | `models/io.py` |
| T7 | 迁移 LBS 与前向 | `models/lbs.py`、`models/smplx.py` |
| T8 | 迁移拟合流程 | `optimizer.py`、`losses.py`、`pipeline.py` |
| T9 | 串联单人 pipeline | `single.py`、`mocap.py` |
| T10 | 补测试与样例配置 | `tests/`、`configs/mv1p.yaml` |

#### 9. 不在 Phase 1 首批范围内的内容

以下内容在文档中保留设计位，但**不应阻塞 MVP 开发**：

- MVMP 的 `match.py`
- C++ `native/pymatch/`
- `modes/multi.py`
- 复杂 Hydra workflow 编排
- 高级可视化与视频输出
- mask / segmentation / neural rendering 相关能力

### Phase 2：MVMP 完整流水线（后续 Phase）

**目标：在现有架构上补全多人支持**

1. 实现 `MultiMode`
2. 迁移 affinity / SVT / associate / criterion 模块
3. 迁移 TrackBase / PeopleGroup 跨帧跟踪
4. 构建 C++ 扩展（`native/pymatch/`）
5. 提供预编译 wheel
6. MVMP 端到端测试

---

## 四、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **算法精度回归** | 高 | 逐模块数值对比测试，差异 > 1e-6 视为回归 |
| PyTorch API 变更 | 中 | 锁定主版本 (>=2.0, <3.0) |
| SMPL-X 模型许可 | 低 | 保留手动下载，不提供分发 |
| C++ 跨平台编译 | 高 | 提供预编译 wheel，CI 多平台构建 |
| detector 关键点语义不一致 | 高 | 增加 adapter 层，统一到 V1 canonical annotation 结构 |
| rtmlib 版本兼容 | 低 | 轻量依赖，锁定主版本 |
| uv 生态变化 | 低 | 锁定 uv 版本，保留 fallback 到 pip |

---

## 五、数据格式规范

### 输入数据结构

```

### MVP 测试数据建议

MVP 阶段优先使用仓库内已有的 `CoreView_377` 做回归测试：

```bash
CoreView_377/
├── intri.yml
├── extri.yml
├── Camera_B1/
├── Camera_B2/
├── ...
├── Camera_B23/
├── keypoints2d/
│   ├── Camera_B1/
│   ├── ...
│   └── Camera_B23/
├── annots/                # 仅部分相机存在
├── match_info.json
├── annots.npy
└── annots_python2.npy
```

建议用途：

- `Camera_B*/` 作为多视图图像输入
- `intri.yml` / `extri.yml` 作为 pinhole 相机参数输入
- `keypoints2d/` 作为 V1 OpenPose 风格参考标注
- `annots/` 作为部分内部 annotation 参考
- `annots.npy` 作为历史聚合格式参考，不作为 LightMocap MVP 的主输入格式
<sequence>/
├── intri.yml          # 相机内参（必需）
├── extri.yml          # 相机外参（必需）
└── images/            # 多视图图像（必需）
    ├── 01/
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    ├── 02/
    │   └── ...
    └── ...
```

### 输出数据结构

```
<output>/
├── keypoints3d/       # 3D 关键点
│   ├── 000000.json
│   └── ...
├── smplx/             # SMPL-X 参数
│   ├── 000000.json
│   └── ...
└── vis/               # 可视化结果（可选）
    ├── smplx/
    │   ├── 01/
    │   │   ├── 000000.jpg
    │   │   └── ...
    │   └── ...
    └── repro/
        └── ...
```

### 输出 JSON 格式

```json
{
    "frame_id": 0,
    "keypoints3d": [[x, y, z, conf], ...],
    "smplx_params": {
        "global_orient": [3],
        "body_pose": [63],
        "jaw_pose": [3],
        "leye_pose": [3],
        "reye_pose": [3],
        "left_hand_pose": [45],
        "right_hand_pose": [45],
        "expression": [10],
        "betas": [10],
        "transl": [3]
    },
    "vertices": [[x, y, z], ...],
    "faces": [[i, j, k], ...]
}
```

---

## 六、技术选型对比总结

### 2D 姿态检测

| 方案 | 安装 | 精度 | 速度 | GPU 需求 |
|------|------|------|------|----------|
| rtmlib | `uv add rtmlib` | 高 | 快 | 可选 |
| MediaPipe | `uv add mediapipe` | 中 | 快 | 不需要 |

### SMPL-X 模型加载

| 方案 | 优点 | 缺点 |
|------|------|------|
| NPZ 格式 | 无需 chumpy，文件小，Python 兼容性好 | 需手动下载 |
| PKL 格式 | — | chumpy 依赖，编码问题，**LightMocap 已移除** |

### 配置管理

| 方案 | 优点 | 缺点 |
|------|------|------|
| Hydra | 模块化，支持组合，CLI 集成 | 学习曲线 |
| yacs | — | 停止维护，**LightMocap 已移除** |

### 环境管理

| 方案 | 优点 | 缺点 |
|------|------|------|
| uv | 快速、现代、锁定文件、项目级管理 | 需安装 uv |
| pip + requirements.txt | — | 依赖解析慢，无锁定，**LightMocap 已移除** |

---

## 七、V1 到 LightMocap 的迁移指南

### 对于已有 V1 用户

1. **安装 uv**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **在 EasyMocap 仓库内创建 lightmocap 目录**
   ```bash
   cd /path/to/EasyMocap
   mkdir lightmocap && cd lightmocap
   uv init --no-readme --no-license --name lightmocap
   uv sync
   ```

3. **下载 SMPL-X NPZ 模型**
   - 从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de) 注册并下载
   - 选择 NPZ 格式（非 PKL）
   - 放置到 `data/smplx/` 目录

4. **下载检测模型**
   ```bash
   uv run lmc download-models --detector rtmlib
   ```

5. **准备相机参数**
   - 保持 V1 的 `intri.yml` / `extri.yml` 格式不变

6. **运行流水线**
   ```bash
   uv run lmc pipeline \
       --images images/ \
       --cameras cameras/ \
       --output output/ \
       --config configs/mv1p.yaml
   ```

### 从 V1 迁移已有脚本

| V1 用法 | LightMocap 用法 |
|---------|-----------------|
| `python apps/mocap/run.py --cfg config/mv1p/xxx.yml` | `uv run lmc pipeline --config configs/mv1p.yaml` |
| `from easymocap.smplmodel import SMPLlayer` | `from lightmocap.models.smplx import SMPLX` |
| `from easymocap.mytools import triangulate` | `from lightmocap.core.triangulation import triangulate` |
| `from easymocap.pyfitting import optimizeShape` | `from lightmocap.core.fitting import optimize_shape` |

---

## 八、测试计划

### 数值精度测试（最高优先级）

- [ ] 使用 EasyMocap V1 测试数据，对比 LightMocap 三角化输出（差异 < 1e-6）
- [ ] 对比 SMPL-X 顶点输出（差异 < 1e-6）
- [ ] 对比 SMPL-X 拟合参数（差异 < 1e-6）
- [ ] 对比重投影像素位置（差异 < 1 像素）
- [ ] 对比 detector adapter 输出的 canonical annotation 是否满足 V1 下游输入要求

### 单元测试

- [ ] NPZ SMPL-X 模型加载测试
- [ ] 相机参数读写测试
- [ ] 三角化算法测试
- [ ] SMPL-X 前向传播测试
- [ ] 检测器接口测试
- [ ] Hydra 配置加载测试

### 集成测试

- [ ] 端到端 MV1P 流水线测试
- [ ] 多视图同步测试
- [ ] 不同检测器对比测试

### 性能测试

- [ ] 单帧处理速度
- [ ] 批量处理速度
- [ ] 内存占用测试
- [ ] GPU 利用率测试

---

## 九、文档计划

### 用户文档

- [ ] 快速开始指南（5 分钟上手）
- [ ] API 文档
- [ ] 配置指南（Hydra）
- [ ] 示例代码
- [ ] V1 迁移指南

### 开发文档

- [ ] 架构设计（MV1P / MVMP 共存架构）
- [ ] 贡献指南
- [ ] 代码风格（Ruff）
- [ ] 测试指南
- [ ] 算法精度验证方法

---

## 十、时间估算

| Phase | 工作内容 | 预计时间 |
|-------|----------|----------|
| **Phase 1** | **MV1P 完整流水线** | |
| | 基础设施（pyproject.toml, uv, 结构） | 2 天 |
| | NPZ SMPL-X 模型加载 | 1 天 |
| | 核心算法迁移 + 精度验证 | 5 天 |
| | MV1P 流水线编排 | 3 天 |
| | 2D 检测器集成 + adapter | 3 天 |
| | CLI 工具 | 2 天 |
| | 可视化 | 2 天 |
| | 测试 + 文档 | 3 天 |
| **Phase 1 小计** | | **约 3 周** |
| | | |
| **Phase 2** | **MVMP 完整流水线** | |
| | MultiMode 实现 + C++ 扩展 | 2 周 |
| | affinity / track / associate 迁移 | 2 周 |
| | 预编译 wheel + 测试 | 1 周 |
| **Phase 2 小计** | | **约 5 周** |
| | | |
| **总计** | | **约 2 个月** |

---

## 附录 A：相关资源

### SMPL-X
- 官网: https://smpl-x.is.tue.mpg.de
- GitHub: https://github.com/vchoutas/smplx
- 论文: https://smpl-x.is.tue.mpg.de

### 检测器
- rtmlib: https://github.com/Tau-J/rtmlib
- MediaPipe: https://github.com/google/mediapipe

### uv
- 文档: https://docs.astral.sh/uv

### Hydra
- 文档: https://hydra.cc

---

## 附录 B：完整 API 示例

```python
import numpy as np
from lightmocap import MoCapPipeline, Camera
from omegaconf import DictConfig

# 1. 加载相机参数
camera = Camera.from_yaml("intri.yml", "extri.yml")

# 2. 创建配置
config = DictConfig({
    "mode": "single",
    "detector": {
        "type": "rtmlib",
        "solution": "wholebody",
        "mode": "lightweight",
    },
    "fitting": {
        "opt_shape": True,
        "opt_pose": True,
        "opt_hand": True,
        "opt_expression": True,
    },
    "model": {
        "model_path": "data/smplx/",
        "gender": "neutral",
        "ext": "npz",
        "num_shape": 10,
        "num_expression": 10,
        "use_hands": True,
        "use_face": True,
    },
})

# 3. 创建 MoCapPipeline 实例
mocap = MoCapPipeline.from_config(config, camera=camera)

# 4. 批量处理
for frame_id in range(1000):
    images = {
        cam: f"images/{cam}/{frame_id:06d}.jpg"
        for cam in camera.names
    }
    result = mocap.process_frame(images, frame_id=frame_id)
    result.save(f"output/{frame_id:06d}.json")

# 5. 批量处理（推荐，支持并行）
mocap.process_directory(
    image_dir="images/",
    output_dir="output/",
    start=0,
    end=1000,
)
```

---

**文档版本：** 3.0
**最后更新：** 2026-04-11
**维护者：** LightMocap 团队
