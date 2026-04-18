# LightMocap

`LightMocap` 是 `EasyMocap` 的第一阶段重构实现，当前聚焦一条可工程使用的多视图单人链路：

- 输入：多视图图像序列 + 相机参数
- 中间结果：2D 检测、3D 三角化
- 输出：SMPL-X 参数、mesh 顶点、叠加渲染图

当前推荐默认配置：

- 检测器：`rtmlib wholebody`
- 模式：`balanced`
- 设备：`cuda`
- 拟合：`3D keypoints + 多视图 2D reprojection refinement`

## 环境准备

推荐环境：

- Python `3.12`
- `torch` / `torchvision` 需要用户自行安装与本机环境匹配的版本
- `uv` 管理环境和依赖

`LightMocap` 不在 `pyproject.toml` 中绑定 `torch` 版本，也不绑定固定 CUDA wheel。
这是为了避免在分发时把某个 CUDA 版本强加给所有用户。

推荐安装顺序：

```bash
cd /home/ymj/code/python/EasyMocap/lightmocap
uv venv .venv --python 3.12

# 1. 先安装适合自己环境的 PyTorch
# 下面只是 CUDA 11.8 的示例，其他 CUDA 版本或 CPU 环境请替换成对应官方命令
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 再安装 LightMocap 其余依赖
UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" uv sync --extra dev --extra detection
```

如果你已经提前装好了合适版本的 `torch` / `torchvision`，只需要执行：

```bash
UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" uv sync --extra dev --extra detection
```

## 输入数据格式

当前推荐输入目录结构：

```text
<sequence_root>/
├── cameras.yaml
├── Camera_B1/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── Camera_B2/
│   └── ...
└── Camera_B23/
    └── ...
```

要求：

- 目前只支持 `pinhole + OpenCV distortion`
- 畸变参数支持 5 参数 `k1, k2, p1, p2, k3`
- 不支持 fisheye
- 所有相机目录中应存在相同编号的同步帧

如果每个相机目录里只有一张图像，`LightMocap` 会自动把它当成单帧多视图输入。

## 相机参数格式

当前推荐使用单文件 `cameras.yaml`。它是普通 YAML，可读、可编辑，不依赖 OpenCV `FileStorage`。

示例：

```yaml
format_version: 1
camera_model: pinhole_opencv
cameras:
  - name: Camera_B1
    image_size:
      width: 1920
      height: 1080
    K:
      - [fx, 0.0, cx]
      - [0.0, fy, cy]
      - [0.0, 0.0, 1.0]
    dist: [k1, k2, p1, p2, k3]
    R:
      - [r11, r12, r13]
      - [r21, r22, r23]
      - [r31, r32, r33]
    T:
      - [tx]
      - [ty]
      - [tz]
```

说明：

- `K`：3x3 内参矩阵
- `dist`：OpenCV 5 参数畸变
- `R`：3x3 旋转矩阵
- `T`：3x1 平移矩阵

为了兼容历史数据，当前读取逻辑仍支持旧版 `intri.yml + extri.yml` 双文件格式，但内部不再依赖 OpenCV `FileStorage`。

把旧相机文件转换成新格式：

```bash
uv run lmc convert-cameras \
  --intri /home/ymj/code/python/EasyMocap/CoreView_377/intri.yml \
  --extri /home/ymj/code/python/EasyMocap/CoreView_377/extri.yml \
  --output /home/ymj/code/python/EasyMocap/CoreView_377/cameras.yaml
```

## 配置文件

当前推荐配置写法：

```yaml
mode:
  type: mvsp

cameras:
  path: /home/ymj/code/python/EasyMocap/CoreView_377/cameras.yaml

data:
  images: /home/ymj/code/python/EasyMocap/CoreView_377
  output: /home/ymj/code/python/EasyMocap/lightmocap/outputs
```

CLI 会优先读取配置里的路径，不再要求用户重复传已经在 config 中声明过的相机和输出参数。

其中：

- `mvsp`：multiview single person，当前已经实现
- `mvmp`：multiview multi person，当前仅预留配置语义，尚未实现

## CLI

当前主 CLI 只保留通用工具命令：

- `validate`
- `detect-image`
- `convert-openpose`
- `convert-cameras`
- `detect`
- `triangulate`
- `fit`
- `run`

`CoreView` 专用评测与 benchmark 已移动到 `scripts/`，和主 CLI 分层：

- `scripts/evaluate_coreview_end2end.py`
- `scripts/benchmark_coreview_sequence.py`

### `--frame` 语义

所有多视图命令里的 `--frame` 都只表示“处理哪一帧”。

```bash
--frame 3
--frame 000003
```

这两种写法等价，内部都会规范化为 `000003`。

### 校验配置和输入

```bash
uv run lmc validate \
  --config /home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml
```

### 单张图像检测

```bash
uv run lmc detect-image \
  --input /home/ymj/code/python/EasyMocap/CoreView_377/Camera_B1/000003.jpg
```

### 多视图单帧 2D 检测

多视图命令统一使用 `--output` 表示输出根目录；不传时默认使用 `config.data.output`。

```bash
uv run lmc detect \
  --config /home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml \
  --frame 3 \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_detect
```

输出：

```text
demo_detect/
└── annots/
    ├── Camera_B1/000003.json
    ├── Camera_B2/000003.json
    └── ...
```

### 多视图三角化

```bash
uv run lmc triangulate \
  --config /home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml \
  --frame 3 \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_triangulate
```

默认会从 `<output>/annots` 读取 annotation，并写出：

```text
demo_triangulate/
├── annots/
└── keypoints3d/000003.json
```

如果 annotation 在别处，可以显式覆盖：

```bash
uv run lmc triangulate \
  --config /home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml \
  --frame 3 \
  --annots /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_detect/annots \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_triangulate
```

### 单帧拟合 SMPL-X

```bash
uv run lmc fit \
  --config /home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml \
  --frame 3 \
  --annots /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_detect/annots \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_fit \
  --render-views 4
```

输出结构：

```text
demo_fit/
├── keypoints3d/000003.json
├── smplx/000003.json
├── vertices/000003.npy
└── renders/
    ├── Camera_B1_000003.jpg
    ├── Camera_B2_000003.jpg
    └── ...
```

### 单帧端到端运行

```bash
uv run lmc run \
  --config /home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml \
  --frame 3 \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/demo_run \
  --render-views 4
```

这个命令会自动执行：

1. 多视图 2D 检测
2. 多视图三角化
3. SMPL-X 拟合
4. 导出参数、顶点和叠加渲染图

输出结构：

```text
demo_run/
├── annots/
│   ├── Camera_B1/000003.json
│   ├── Camera_B2/000003.json
│   └── ...
├── keypoints3d/000003.json
├── smplx/000003.json
├── vertices/000003.npy
└── renders/
    ├── Camera_B1_000003.jpg
    ├── Camera_B2_000003.jpg
    └── ...
```

### 输出目录约定

`detect`、`triangulate`、`fit`、`run` 现在统一使用同一套输出根目录布局：

```text
<output>/
├── annots/
├── keypoints3d/
├── smplx/
├── vertices/
└── renders/
```

约定如下：

- `detect`：写 `annots/`
- `triangulate`：读 `annots/`，写 `keypoints3d/`
- `fit`：读 `annots/`，写 `keypoints3d/`、`smplx/`、`vertices/`、`renders/`
- `run`：写完整布局

因此更推荐把多个阶段串到同一个 `--output` 根目录下，而不是为每个阶段单独造一套结构。

## CoreView 评测脚本

如果需要跑 `CoreView_377` 专用评测，请直接使用 `scripts/`：

```bash
uv run python scripts/evaluate_coreview_end2end.py \
  --frame 000003 \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/coreview_e2e
```

```bash
uv run python scripts/benchmark_coreview_sequence.py \
  --start-frame 0 \
  --num-frames 12 \
  --output /home/ymj/code/python/EasyMocap/lightmocap/outputs/coreview_sequence_benchmark
```

## Python API 示例

```python
from lightmocap.pipeline.mocap import MoCapPipeline

pipeline = MoCapPipeline.from_yaml(
    "/home/ymj/code/python/EasyMocap/lightmocap/configs/mv1p.yaml"
)

result = pipeline.process_frame({
    "Camera_B1": "/home/ymj/code/python/EasyMocap/CoreView_377/Camera_B1/000003.jpg",
    "Camera_B2": "/home/ymj/code/python/EasyMocap/CoreView_377/Camera_B2/000003.jpg",
})

print(result["keypoints3d"].shape)
print(result["vertices"].shape)
```

## 当前局限

- 当前工程链路以多视图单人为主
- hand / face 目前主要依赖 detector 一致性和 reprojection 约束
- 多帧时域优化代码保留，但没有作为主 CLI 暴露
- 还没有加入 silhouette / mask loss

## 测试

运行测试：

```bash
uv run pytest
```

当前重点测试覆盖：

- 新旧相机 YAML 兼容读取
- 相机参数 round-trip
- annotation 转换
- 与 V1 三角化一致性
- `SMPL-X NPZ` 加载与前向
- 真实数据回归
