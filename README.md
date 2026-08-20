# AeroLoop

AeroLoop 是连接 **UAV 导航模型** 与 **仿真器** 的轻量中间层，面向闭环测试、指标评测和数据采集。

它负责统一数据集、观测、动作、仿真生命周期和结果输出，但不要求模型与仿真器安装在同一个 Conda 环境中，也不依赖外部仿真 bridge。

> AeroLoop 提供通用的 AirSim、GS-AirSim 和 UnrealCV 直连适配器，但不随 Python 包发布大型场景资产。场景资产应放在本地运行目录或独立数据盘中。

## 1. 能做什么

```text
OpenFly / TravelUAV / 自定义数据集
                  |
                  v
             EpisodeSpec
                  |
                  v
Model runtime <- HTTP -> AeroLoop Runner -> Simulator SDK
独立 Conda 环境          |               AirSim / GS / UE
                         v
             JSONL / 指标 / 视频 / 碰撞帧
```

目前支持：

| 能力 | 当前支持 |
| --- | --- |
| 仿真器 | Mock、AirSim、GS-AirSim、UnrealCV、自定义插件 |
| 数据集 | OpenFly、TravelUAV、inline、自定义 loader |
| 模型连接 | 本地 Policy、HTTP 模型服务、Python 推理函数包装、插件 |
| 观测 | 多相机 RGB、历史帧、深度、位姿、相对状态、自定义辅助状态 |
| 动作 | 单步或 action chunk，统一为机体坐标系增量动作 |
| 评测 | SR、OSR、NE、SPL、路径长度、碰撞、stop、推理耗时 |
| 记录 | 完整 step trace、JSONL、环境级与全局汇总 |
| 可视化 | 数据集 HTML、实时图像窗口、MP4、碰撞截图 |
| 扩展 | Simulator、Policy、Dataset、Metric、Observer |

## 2. 五分钟跑通

### 2.1 安装

基础安装只依赖 PyYAML，可以直接运行 Mock 测试：

```bash
python -m pip install -e .
```

需要图像、视频和完整开发测试时：

```bash
python -m pip install -e '.[media,dev]'
```

需要直连仿真器时按需安装：

```bash
python -m pip install -e '.[airsim]'
python -m pip install -e '.[unrealcv]'
# 或一次安装所有通用仿真 SDK
python -m pip install -e '.[simulators]'
```

检查当前环境：

```bash
aeroloop doctor
```

### 2.2 运行 Mock 闭环

```bash
aeroloop run --config configs/mock.yaml
```

这会执行一条 10 米直线任务，并生成：

```text
eval_results/mock.jsonl
```

输出包含 run config、每个 episode 的结果、每步动作与位姿、环境汇总和全局汇总。

### 2.3 跑测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
ruff check src tests scripts
```

## 3. 模型与仿真使用不同 Conda 环境

推荐将模型作为 HTTP 服务运行：

```text
仿真环境                                  模型环境
AirSim / UnrealCV SDK                     PyTorch / Transformers / 模型代码
aeroloop run       <-- HTTP/JSON -->       aeroloop-model-server
```

这样两侧可以使用不同版本的 Python、CUDA、PyTorch，甚至运行在不同机器上。

标准接口：

- `GET /health`：模型健康状态和能力信息；
- `POST /v1/reset`：开始一个新 episode；
- `POST /v1/predict`：输入观测并返回动作。

模型返回的标准动作格式为：

```text
[dx_body, dy_body, dz_body, d_yaw, stop_probability]
```

- 位移单位是米；
- `d_yaw` 使用弧度；
- XYZ 位移位于 UAV 当前机体坐标系；
- 一次可以返回一个动作或一个 action chunk；
- 为兼容已有模型，也接受 `[dx, dy, dz, stop]`。

完整字段定义见 [HTTP API](docs/HTTP_API.md) 和 [仿真器契约](docs/SIMULATOR_CONTRACT.md)。

### 3.1 先用示例 HTTP 模型验证

终端 A：

```bash
python examples/mock_http_server.py --port 18080
```

终端 B：复制 `configs/mock.yaml`，将其中的 `policy` 替换为：

```yaml
policy:
  type: http
  kwargs:
    url: http://127.0.0.1:18080/v1/predict
    reset_url: http://127.0.0.1:18080/v1/reset
    timeout_s: 120
    observation_fields: [state]
    state_source: relative
```

然后运行新配置即可验证 HTTP 闭环。

## 4. 只包装自己的推理函数

模型作者通常不需要实现 HTTP Server 或 AeroLoop Backend，只需要保留自己的推理函数：

```python
def predict(instruction, image, state):
    # 运行自己的模型
    return [[1.0, 0.0, 0.0, 0.0, 0.0]]
```

在模型环境中编写字段映射：

```yaml
function:
  entrypoint: my_model.inference:predict
  inputs:
    instruction: instruction
    image: images.front
    state: state
  static_kwargs: {}
  actions_key: actions
```

启动服务：

```bash
aeroloop-model-server function \
  --config configs/models/function.yaml \
  --host 0.0.0.0 \
  --port 18080
```

推理函数可以直接返回动作列表，也可以返回：

```python
{
    "actions": [[1.0, 0.0, 0.0, 0.0, 0.0]],
    "metadata": {"raw_output": "..."},
}
```

## 5. 用 YAML 控制模型接收什么

仿真器可以产生冗余观测，模型侧只发送配置中声明的字段：

```yaml
policy:
  type: http
  kwargs:
    url: http://127.0.0.1:18080/v1/predict
    reset_url: http://127.0.0.1:18080/v1/reset

    # 模型需要的相机
    views: [front, down]

    # 只发送这些字段
    observation_fields: [images, state, camera_specs]

    # relative: 起点坐标系；world: 仿真世界坐标系
    state_source: relative

    # 发给模型前统一缩放，格式为 [width, height]
    image_size: [224, 224]

    # 包含当前帧，按 oldest -> current 发送
    history_steps: 4
```

可选字段：

| 字段 | 内容 |
| --- | --- |
| `images` | 命名相机图像和历史帧 |
| `state` | 配置所选坐标系下的 `[x,y,z,yaw]` |
| `pose` | 世界坐标位姿 |
| `auxiliary_state` | 仿真器提供的额外状态 |
| `camera_specs` | 相机分辨率、FOV 和外参 |

仿真采集哪些相机、模型接收哪些相机是两层独立配置，避免采集或传输不需要的数据。

## 6. 直连仿真器

内置仿真器配置位于：

- `configs/simulators/airsim.yaml`
- `configs/simulators/gs_airsim.yaml`
- `configs/simulators/unrealcv.yaml`

场景目录约定如下：

```text
.runtime/envs/
├── airsim/
│   └── env_airsim_16/LinuxNoEditor/start.sh
├── gs_airsim/
│   └── env_gs_ecust/gs.sh
└── ue/
    └── env_ue_bigcity/CitySample.sh
```

`.runtime/` 已被 Git 忽略。不要把场景可执行文件、压缩包或 SDK 缓存提交到仓库。

AirSim 示例：

```yaml
simulator:
  type: airsim
  kwargs:
    env_root: .runtime/envs/airsim
    launch_script: LinuxNoEditor/start.sh
    host: 127.0.0.1
    port: 41451
    vehicle_name: drone_1
    cameras:
      - name: front
        width: 896
        height: 896
        fov: 90
    camera_names:
      front: front_custom
    position_sign: [1, -1, -1]
    yaw_sign: -1
    channel_order: bgr
    ignore_collision: false
```

坐标系、比例、相机名和启动参数由适配器在仿真边界完成转换，模型始终使用 AeroLoop 的标准观测与动作。

更多配置见 [直连仿真指南](docs/DIRECT_SIMULATORS.md)。

## 7. 数据集适配

### 7.1 OpenFly

```yaml
benchmark:
  source: openfly
  kwargs:
    path: /datasets/openfly/test_data.json
    dataset_root: /datasets/openfly
    include_envs: env_airsim_18
    limit: 100
    path_rewrites:
      /old/machine/openfly: /datasets/openfly

metrics:
  profile: openfly
```

适配器读取测试清单以及每条轨迹中的 `pose_bbox_updated.json`。

### 7.2 TravelUAV

```yaml
benchmark:
  source: traveluav
  kwargs:
    path:
      seen: /datasets/TravelUAV/data/uav_dataset/seen_valset.json
      unseen: /datasets/TravelUAV/data/uav_dataset/unseen_valset.json
    dataset_root: /datasets/TravelUAV/dataset
    deduplicate: true

metrics:
  profile: traveluav
```

适配器会保留 Seen/Unseen、Unseen Map/Object 以及 Easy/Hard 分组信息。

### 7.3 检查数据

运行仿真前先检查路径和标准化结果：

```bash
aeroloop inspect-dataset \
  --config configs/datasets/openfly.yaml \
  --limit 5
```

数据格式和迁移方式见 [数据集指南](docs/DATASETS.md)。

## 8. 数据可视化

将数据集生成为自包含 HTML：

```bash
aeroloop visualize-dataset \
  --config configs/datasets/openfly.yaml \
  --output eval_results/openfly_preview.html \
  --limit 100
```

页面会展示：

- 指令、环境和 split；
- XY 轨迹、起点、目标和路径长度；
- 轨迹目录中的样例图片；
- Episode 元数据。

评测过程中的实时画面、视频和碰撞截图通过 `media` 配置控制：

```yaml
media:
  show_window: false
  save_video: true
  video_dir: eval_results/videos
  fps: 10
  save_collision_frame: true
  collision_dir: eval_results/collisions
```

## 9. Rollout 与终止条件

```yaml
rollout:
  max_steps: 200

  # 每次模型预测后执行多少个动作；null 表示执行完整原生 chunk
  execution_horizon: 1

  stop_threshold: 0.5
  terminate_on_collision: true
  terminate_on_success: false
  execute_motion_on_stop: false
```

Runner 支持：

- 每步重规划或执行模型原生 action chunk；
- 模型 stop、到达目标、碰撞、最大步数和 observer 主动终止；
- 每个环境复用一个仿真进程；
- 单 episode 错误隔离；
- 自定义 metric 和 observer。

## 10. 指标与输出

内置 metric profile：

- `standard`
- `openfly`
- `traveluav`

每个 episode 记录：

- Success/SR、OSR、NE、SPL；
- 最终二维/三维目标距离和最小距离；
- 实际路径长度与参考路径长度；
- 碰撞、stop、premature stop、timeout；
- 推理次数、总耗时和平均耗时；
- 可选的完整 step trace 和模型 metadata。

JSONL 顺序为：

1. `run_config`
2. `episode`
3. `environment_summary`
4. `overall_summary`
5. 可选的 benchmark split/difficulty 汇总

完整结构见 [输出格式](docs/OUTPUT_SCHEMA.md)。

## 11. 参考模型服务

仓库包含以下可选参考后端：

- AerialVLA
- OpenUAV
- DualVLN
- WorldVLN
- OmniNav
- PI0

这些后端只在启动对应服务时加载模型依赖，不会污染 AeroLoop 基础环境。新模型优先使用“推理函数包装”接入；只有需要复杂历史管理、原生分段执行或特殊 action codec 时，才建议新增 Backend。

启动方式和模型特有参数见 [模型服务指南](docs/MODEL_SERVERS.md) 与 [模型适配检查表](docs/MODEL_ADAPTER_CHECKLIST.md)。

## 12. 自定义扩展

可以通过 Python import path 或安装包 entry point 扩展：

- `aeroloop.simulators`
- `aeroloop.policies`
- `aeroloop.episode_sources`
- `aeroloop.metrics`
- `aeroloop.observers`

示例：

```yaml
simulator:
  type: custom
  entrypoint: my_package.simulator:MySimulator
  kwargs: {}
```

扩展接口见 [扩展指南](docs/EXTENDING.md)。

## 13. 项目结构

```text
src/aeroloop/
├── datasets/       # OpenFly、TravelUAV 和可视化
├── policies/       # HTTP、Mock 和组合 Policy
├── server/         # 模型 HTTP Server 与参考 Backend
├── simulators/     # AirSim、GS-AirSim、UnrealCV
├── runner.py       # 闭环执行
├── metrics.py      # 内置与自定义指标
├── media.py        # 视频、窗口和碰撞帧
└── cli.py          # aeroloop 命令

configs/
├── datasets/       # 数据集模板
├── simulators/     # 仿真器模板
├── models/         # 模型 HTTP 配置
└── jobs/           # 完整评测任务
```

## 14. 当前已知限制

- OpenFly profile 已输出统一指标，但建筑中心/表面距离和点云碰撞尚未完全迁移，因此目前结果不能直接视为官方完全等价分数。
- UnrealCV 场景只有在自身提供 collision query 时才能报告原生碰撞；否则需要额外点云碰撞 observer/metric。
- OpenFly 和 TravelUAV 数据集、仿真资产及模型 checkpoint 均需单独准备，不随项目发布。
- 当前支持数据集 HTML、视频和碰撞帧；完整的交互式 rollout 对比 Dashboard 尚未实现。

## 15. 常用文档

- [HTTP API](docs/HTTP_API.md)
- [仿真器契约](docs/SIMULATOR_CONTRACT.md)
- [直连仿真器](docs/DIRECT_SIMULATORS.md)
- [数据集适配](docs/DATASETS.md)
- [模型服务](docs/MODEL_SERVERS.md)
- [输出格式](docs/OUTPUT_SCHEMA.md)
- [扩展开发](docs/EXTENDING.md)
- [贡献指南](CONTRIBUTING.md)

项目使用 [Apache-2.0](LICENSE) 许可证。
