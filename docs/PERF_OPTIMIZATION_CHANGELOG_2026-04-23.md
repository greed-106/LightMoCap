# LightMoCap 本轮单帧性能优化说明

> 日期：2026-04-23  
> 项目：`/home/ymj/code/python/LightMoCap`  
> 范围：单帧 `detect -> triangulate -> fit -> optional render`  
> 说明：本文是对当前代码状态的最终说明，已经把本轮保留的优化、回退的实验性修改、以及新增的验证结果统一整理。

## 1. 这次最终保留了什么

这次最终保留的，不是所有之前试过的性能改动，而是对单帧链路真正有价值、且不打乱后续多帧联合优化计划的部分。

最终保留的改动有四类：

1. 单帧 SMPL-X 拟合阶段保持 torch-native 串接，在真正的 I/O 边界前尽量不退回 numpy。
2. `body25` 关节阶段走 joints-only 快路径，避免为不参与 loss 的 vertices 付费。
3. 默认不再输出 `vertices/*.npy`，也不再在无渲染路径下白算一次 mesh。
4. `run` 命令在需要渲染时复用已经读入的图像，减少重复 `cv2.imread()`。

对应代码位置：

- `src/lightmocap/core/fitting/pipeline.py`
- `src/lightmocap/core/fitting/losses.py`
- `src/lightmocap/models/smplx.py`
- `src/lightmocap/models/lbs.py`
- `src/lightmocap/pipeline/mocap.py`
- `src/lightmocap/workflow/frame.py`
- `src/lightmocap/cli/main.py`
- `run_actor5_frames.sh`
- `scripts/evaluate_coreview_end2end.py`
- `README.md`

## 2. 这次明确回退了什么

本轮明确回退了 sequence-level 的冷启动优化方案：

- `lmc run-sequence` 已从主 CLI 中移除。
- `run_actor5_frames.sh` 改回逐帧调用 `lmc run`。

这样做不是因为 `run-sequence` 没有效果，而是因为当前阶段用户明确希望专注单帧本身，不希望 sequence-level 编排影响后续多帧联合优化设计。

因此，现在的正式结论只讨论单帧热路径，不再把 cold-start / persistent process 作为主线能力。

## 3. 改动一：单帧 end-to-end 保持 tensor-native，到边界再转 numpy

### 3.1 做了什么

原先的拟合优化虽然已经引入了部分 tensor 快路径，但接口层仍然比较别扭：

1. `fit_global_torso()` / `fit_pose3d()` / `fit_pose2d()` 暴露了 `return_tensor` 这种控制参数。
2. `run` 路径虽然是端到端，但仍然较早回到 numpy 表示。
3. 对调用者来说，“什么时候应该拿 tensor，什么时候应该拿 numpy” 不够清晰。

现在的做法是：

1. `SMPLXFittingPipeline` 新增 `fit_tensor()`，作为内部单帧 end-to-end 快路径。
2. 对外公开的 `fit()` / `fit_global_torso()` / `fit_pose3d()` / `fit_pose2d()` 仍然返回 numpy，继续服务 CLI 分阶段使用场景。
3. `MoCapPipeline.process_frame()` 改为调用内部 tensor 路径 `_fit_keypoints3d_tensor()`，使 `run` 链路在拟合完成后仍持有 GPU tensor。
4. 落盘时由 `save_frame_result()` 统一把 torch tensor 转成 numpy/list，渲染时则直接使用内存中的 `smplx_params`。

### 3.2 为什么这样更优雅

相比“公开方法上挂一个 `return_tensor=True/False` 开关”，现在的接口边界更清楚：

1. 对外 API：默认 numpy，适合 CLI、JSON、分阶段保存。
2. 对内 API：显式 `fit_tensor()`，适合单帧端到端推理。

这样调用者不需要知道内部该不该切换 `return_tensor`，也不会把快路径策略泄漏到所有 public method 上。

### 3.3 为什么这样更高效

对单帧 `run` 来说，真正需要 numpy 的时刻只有两个：

1. 写 `keypoints3d/*.json` 与 `smplx/*.json`
2. 渲染最终 overlay 时把 vertices 作为 numpy 喂给渲染器

因此，中间的 SMPL-X 优化变量没有必要在 stage 间或 `process_frame()` 结束时就提前退回 CPU。

这个修改能减少：

1. `torch.Tensor -> numpy -> torch.Tensor` 的重复往返
2. 不必要的设备同步
3. 重复的 dtype/device 规范化

它没有改变优化目标，也没有改变 loss，只是把边界放在了更合理的位置。

## 4. 改动二：保留 joints-only 快路径，继续避免无效顶点计算

### 4.1 做了什么

这一部分延续了上一轮已经验证有效的优化：

1. `lbs()` 支持 `compute_verts=False`
2. `SMPLXLayer.forward()` 会根据 `return_verts` 和 `keypoint_mode` 判断当前是否真的需要 vertices
3. `shape` / `torso` / `body25` pose 这类只依赖关节的阶段，不再走完整 mesh skinning

### 4.2 为什么有效

在 `body25` 场景里，loss 关注的是 joints，而不是完整 mesh。  
如果每次 closure 都把所有 vertices 蒙皮算出来再丢掉，这属于纯浪费。

这个快路径带来的收益来源非常直接：

1. 少了一大块 LBS 顶点矩阵运算
2. 少了显存读写
3. LBFGS line search 每次 closure 更轻

需要特别说明的是：

- 这并不意味着 wholebody 模式完全不需要 vertices
- 当 `keypoint_mode == "bodyhandface"` 时，手指尖与面部 landmark 仍需要 mesh 顶点/面插值

所以这里优化的是“按阶段裁剪无效工作”，不是“一刀切去掉 vertices”。

## 5. 改动三：默认不再输出 `vertices/*.npy`

### 5.1 做了什么

现在标准输出布局里不再包含 `vertices/` 目录：

旧布局：

```text
<output>/
├── annots/
├── keypoints3d/
├── smplx/
├── vertices/
└── renders/
```

新布局：

```text
<output>/
├── annots/
├── keypoints3d/
├── smplx/
└── renders/
```

相应地：

1. `FrameOutputLayout` 去掉了 `vertices` 字段
2. `save_frame_result()` 不再保存 `vertices/*.npy`
3. `fit` / `run` 默认也不再提前计算 `vertices`

### 5.2 为什么这样是合理的

`vertices/*.npy` 本质上只是一次内部 mesh 前向结果的缓存：

1. 下游常见 DCC/可视化软件并不能直接把这个目录当最终交换格式使用
2. 真正稳定、可复用的结果是 `smplx/*.json` 参数
3. 如果后续真的需要 mesh，可以随时从 `smplx_params` 按需重建

也就是说，保存 `vertices/*.npy` 既不是必要产物，也不是标准接口，更多只是一个临时中间结果。

### 5.3 为什么这样还能省时间

此前 `run` 在 `--render-views 0` 时也会做这件事：

1. 拟合出 `smplx_params`
2. 立刻前向一次 body model 得到 `vertices`
3. 把 `vertices` 存成 `.npy`

如果当前既不渲染，也没有别的消费者读取这个 `.npy`，那么这整段计算和写盘都是白做的。

现在改成：

1. 无渲染时：只保存 `keypoints3d` 和 `smplx_params`
2. 需要渲染时：由 `render_smplx_result_views()` 按需现算一次 vertices

这样把 mesh 计算从“默认发生”收缩成了“确实需要时才发生”。

## 6. 改动四：`run` 路径保留图像复用

这部分虽然不是这轮最大的收益来源，但它与单帧目标不冲突，所以保留。

做法是：

1. `process_frame()` 返回已加载的 `images`
2. `render_result_views()` / `render_smplx_result_views()` 优先复用内存中的图像
3. 避免 `run` 路径在渲染阶段再次 `cv2.imread()`

这属于低风险的小优化，主要减少重复 I/O 和 JPEG 解码。

## 7. 为什么这套修改组合是有效的

这次最终保留的方案有一个很明确的工程原则：

1. 端到端单帧推理时，把参数尽量留在 GPU 上
2. 只在 JSON/渲染这些天然 I/O 边界前转 numpy
3. 只在真的需要 mesh 的阶段计算 vertices
4. 不把仅为调试服务的中间结果变成默认产物

从工程上看，这些优化和“改 loss 权重”“改优化器超参”不同。  
它们没有改变拟合问题本身，只是消除了结构性浪费，因此风险更低，也更容易验证。

## 8. 本次实测验证

## 8.1 修改前单帧基线

命令：

```bash
/usr/bin/time -p uv run lmc run \
  --config ./configs/actor5_scaled.yaml \
  --frame 613 \
  --output /home/ymj/code/python/LightMoCap/output/regression_before_single
```

结果：

- `real 24.13s`
- 输出中包含 `vertices/`

## 8.2 修改后单帧结果

命令：

```bash
/usr/bin/time -p uv run lmc run \
  --config ./configs/actor5_scaled.yaml \
  --frame 613 \
  --output /home/ymj/code/python/LightMoCap/output/regression_after_single
```

结果：

- `real 21.49s`
- 输出中不再有 `vertices/`

相对修改前：

- `24.13s -> 21.49s`
- wall-clock 下降约 `10.9%`

这部分收益主要来自：

1. 无渲染路径不再额外计算一次完整 mesh
2. `run` 路径把拟合参数保持在 tensor 直到保存边界

## 8.3 数值回归结果

对同一帧 `000613`，比较修改前后的结果：

- `keypoints3d max_abs_diff = 0.0`
- `poses max_abs_diff = 0.0`
- `Rh max_abs_diff = 0.0`
- `Th max_abs_diff = 0.0`
- `shapes max_abs_diff = 0.0`
- `expression max_abs_diff = 0.0`

也就是说，本次修改在该真实单帧样本上的数值结果完全一致，没有引入拟合退化。

## 8.4 渲染 smoke test

命令：

```bash
/usr/bin/time -p uv run lmc run \
  --config ./configs/actor5_scaled.yaml \
  --frame 613 \
  --render-views 1 \
  --output /home/ymj/code/python/LightMoCap/output/regression_render_single
```

结果：

- `real 29.44s`
- 成功输出：
  - `output/regression_render_single/renders/cam01_000613.jpg`

说明去掉默认 `vertices` 落盘后，渲染能力没有丢失，只是变成了按需生成 mesh。

## 8.5 自动化测试

已通过：

```bash
uv run pytest tests/test_fitting_phase_a.py tests/test_frame_workflow.py
```

结果：

- `4 passed`

新增测试覆盖了：

1. `save_frame_result()` 可以直接处理 torch tensor 参数
2. 新输出布局不会再创建 `vertices/`

同时还验证了旧的 Phase A 相关逻辑没有被破坏。

## 9. 代码整洁性上的整理

这次除了性能方向，也顺手把几处接口整理了一下：

### 9.1 去掉 public API 上的 `return_tensor` 分支

这是最重要的一点。  
现在不再让外部调用者在 `fit_global_torso()` / `fit_pose3d()` / `fit_pose2d()` 上显式决定“要 tensor 还是 numpy”，而是：

1. public API 默认 numpy
2. internal fast path 使用 `fit_tensor()`

这比把策略开关暴露给所有调用者更符合最佳实践。

### 9.2 落盘逻辑统一处理 tensor -> numpy

`save_frame_result()` 现在集中负责把 tensor 参数转成 numpy/list，避免转换逻辑散落在 CLI 和脚本里。

### 9.3 评测脚本与 README 同步更新

因为 `process_frame()` 不再默认返回 `vertices`，相关使用方也一起更新了：

1. `scripts/evaluate_coreview_end2end.py` 改成按需从 `smplx_params` 重建 vertices
2. `README.md` 不再声明 `vertices/` 是标准输出产物

这样仓库内部关于“什么是正式结果、什么是按需中间量”的语义是一致的。

## 10. 目前仍然存在的边界

虽然本轮已经把单帧 end-to-end 路径整理得更合理，但仍有几个边界需要明确：

1. 分阶段 CLI 依然天然会有 CPU/GPU 边界，因为它们本来就是文件化中间结果驱动的。
2. torso 初始化仍包含一次 numpy/OpenCV 边界，这部分还不是纯 torch。
3. wholebody 模式里，手脸关键点仍然会需要 mesh 顶点，这是 SMPL-X 表达方式本身决定的，不是冗余。
4. 当前结论只覆盖单帧，不覆盖未来的多帧联合优化。

## 11. 结论

本次最终修改是有效的，理由很明确：

1. 它去掉了没有正式产品价值的 `vertices/*.npy` 默认输出。
2. 它让单帧 `run` 在无渲染时少做了一次完整 mesh 前向和一次 `.npy` 落盘。
3. 它把 end-to-end 单帧路径整理成了更清晰的 internal tensor fast path，而不是把 `return_tensor` 策略暴露给调用者。
4. 它在真实单帧样本上实现了 `10.9%` 的 wall-clock 改善，同时数值结果完全一致。
5. 它保留了渲染能力，也没有破坏已有 Phase A 拟合改进。

如果后续继续做单帧优化，下一步更值得看的方向是：

1. torso 初始化中的 numpy/OpenCV 边界还能否继续收缩
2. 渲染器上下文是否值得缓存
3. 检测阶段跨视角能否有更深的批处理

但在当前阶段，这次修改已经把“无意义 mesh 输出”和“单帧 end-to-end 的接口不够干净”这两个问题处理到位了。
