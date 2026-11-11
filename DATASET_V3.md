# Dataset v3 使用说明

在项目根目录的 PowerShell 执行采集命令，正式采集前启动 CARLA 0.9.16。

```powershell
cd D:\Projects\carla-uav-camera-lidar-dataset
conda activate carla
```

如果当前环境缺少原项目依赖 `ruamel.yaml`，先执行 `python -m pip install ruamel.yaml`。

## 录制命令

单条路线、晴天、开启扰动、录制完整路线：

```powershell
python scripts/collect_uavdataset.py --route-file routes/Town05_Opt.yaml --weather clear --run-id run_0001 --disturbance --output-root "D:\UAVDatasets\dataset_v3"
```

同样路线、关闭扰动，使用另一个根目录：

```powershell
python scripts/collect_uavdataset.py --route-file routes/Town05_Opt.yaml --weather clear --run-id run_0001 --no-disturbance --output-root "D:\UAVDatasets\dataset_v3_no_disturbance"
```

`--disturbance` / `--no-disturbance` 仅控制横滚、俯仰、航向扰动，不包含任何额外平移。关闭时三轴扰动逐帧为零；路线转弯、变速、20–50 m 高度变化和悬停继续执行。不传开关时使用 `configs/UAVdataset.yaml` 中 `uav.disturbance.enabled`，当前默认开启。

`--output-root` 支持绝对路径（含空格时加引号）；相对路径以项目根目录为基准。不传时输出到项目内 `dataset_v3`。

先对 Town05 五种环境各试采 1200 帧（120 s 帧时段，首末采样时间相差 119.9 s）：

```powershell
python scripts/collect_all_routes.py --route-file routes/Town05_Opt.yaml --weather all --num-frames 1200 --disturbance --output-root "D:\UAVDatasets\v3_trial"
```

这覆盖四阶段速度循环；立交桥可能位于更后面的路线位置，要检查该路段时录制 Town05 完整路线。

八张地图 × 五种环境，正式录制共 40 个 run：

```powershell
python scripts/collect_all_routes.py --weather all --run-id run_0001 --no-disturbance --output-root dataset_v3
```

关闭全部扰动时把 `--disturbance` 替换为 `--no-disturbance`。天气可选 `clear`、`overcast`、`night`、`dawn`、`rain`；`all` 仅用于批量入口。`--num-frames -1` 为完整路线，也是当前配置默认值。

仅生成计划，无需连接 CARLA（仍需安装 CARLA Python 客户端与采集依赖）：

```powershell
python scripts/collect_all_routes.py --route-file routes/Town05_Opt.yaml --weather all --disturbance --plan-only --output-root "D:\UAVDatasets\v3_plan_preview"
```

计划预览使用独立目录；它会写计划文件，因此不能再向同一目录正式录制。采集器拒绝覆盖非空 run，批量入口在启动前统一检查全部目标路径。已有路线修改后，显式指定 `--run-id run_0002`；run 编号不自动递增，不按天气或扰动状态自动改变。

## 数据与计时约定

```text
dataset_v3/
└─ Town05_Opt_clear/
   └─ run_0001/
      ├─ metadata.json
      ├─ calibration.json
      ├─ route.yaml
      ├─ config.yaml
      ├─ motion_plan.json
      ├─ attitude_schedule.json
      ├─ scenes.json
      ├─ rgb/
      ├─ lidar/
      ├─ pose/
      └─ labels/
```

同一路线、相同帧数与扰动设置，在不同天气下使用同一运动与扰动计划。开始采集前计划已保存，采集循环只读取对应帧；交通设置和种子相同不代表车辆实际轨迹逐帧一致。保存实际路线、配置和天气参数，完成后 `metadata.recording_complete` 为 true；中断录制不能作为完整训练数据转换。

四阶段循环：低匀速 4 m/s、余弦加速、高匀速 12 m/s、余弦减速，各 20 s。主巡航累计 30 s 后进入 3 s 停车、3 s 净悬停、3 s 恢复；三段期间冻结主速度循环与下次悬停计时。高度周期累计实际前进时间，包括停车和恢复中的移动，净悬停冻结。到达路线终点结束，保留精确终点帧。

姿态事件：空闲整秒时间格 5% 概率触发，3–6 s，结束后至少 2 s 间隔；横滚/俯仰上限 ±2°，航向上限 ±3°。事件外为零；不足以完整执行的尾部事件不生成。曲线平滑回零。相机、实例分割相机、LiDAR 共同随平台转动，以矩阵组合保持刚性外参。悬停时名义位置和高度冻结，姿态事件继续执行。

scene 采用半开帧范围 `[start_frame, end_frame_exclusive)`，每 200 帧一段，尾段保留。划分只建立索引，不重启交通或重置运动。原始点云为 float32 `[x,y,z,intensity]`，训练转换后五列，第五列由历史帧加载器填时间差。每 5 帧一个候选关键帧、前 9 帧历史点云，严格不跨 scene；完整 200 帧段通常保留 38 个训练样本。

## BEVFusion 转换与划分

以下在 BEVFusion 的训练环境和项目根目录执行（Linux / WSL 路径举例）。采集器不决定训练划分。

```bash
python tools/create_uavdataset.py \
  --root-path /mnt/d/UAVDatasets/dataset_v3 \
  --out-dir data/uavdataset_v3 \
  --split-file configs/uavdataset/splits/town_split_v3.yaml \
  --max-sweeps 9 --keyframe-stride 5 --image-mode copy
```

`town_split_v3.yaml` 仅提供初始整地图分配，用于转换出基础索引。它按 `metadata.route_map` 让同一地图的全部天气与 run 归入同一集合；不表示已经解决 Town05 高低路层分布不均。原始目录与转换目录请分开。

空间划分入口在 `tools/rebalance_uav_splits.py`。复制并编辑 `configs/uavdataset/splits/spatial_v3.example.yaml`：`spatial_maps` 的 key 是地图名，每个 `regions` 元素为 `split: train/val/test` 和 `bounds_xy: [xmin,ymin,xmax,ymax]`。坐标使用 CARLA 世界 XY，单位米，边界必须是实际数值。区域外使用 `default_split`。同一地图的所有天气、不同路线版本共享区域定义，区域不能重叠。

示例文件目前 regions 为空，等价于初始整地图分配；实际区域应根据试采统计与立交桥覆盖情况填写，不能照搬旧版本帧号。默认 110 m 缓冲带在区域边界两侧排除样本；跨区域或触及排除带的 20 s 片段整体排除，历史帧也检查归属。缓冲值需随训练视野范围调整。

```bash
python tools/rebalance_uav_splits.py \
  --dataset-root data/uavdataset_v3 \
  --plan configs/uavdataset/splits/spatial_v3.example.yaml \
  --output-dir data/uavdataset_v3 \
  --sweeps 0 3 6 9 --dry-run
```

审核 `split_audit_preview.json` 的样本、类别和低 Z 分布后，去掉 `--dry-run` 生成 `_s0/_s3/_s6/_s9.pkl`。训练时在原 D=2 或 D=7 的 `tools/train.py` 命令中加 `--dataset_root data/uavdataset_v3/` 指向新目录，沿用对应网络和权重方案。大范围变更前先验证一个训练批次，重新统计体素数量和 Z/深度覆盖。

转换使用逐帧完整姿态，把原始点云、框、相机投影、速度、历史点云统一到水平道路参考系：原点为实际 LiDAR 世界 XY 与当前道路 Z，X 朝向未加扰动的路线航向，Y 向左、Z 向上。网络仍预测 yaw 直立框，坡道目标自身倾斜属于近似，原始完整姿态保留。历史点云只补偿平台运动，移动目标可能保留拖影。

预测导出保留完整 run 身份与参考系到原始 LiDAR 的逆变换；回放使用 run 路径：

```powershell
python scripts/play_uavdataset.py --scene "D:\UAVDatasets\dataset_v3\Town05_Opt_clear\run_0001"
python scripts/play_uav_test_results.py --scene "D:\UAVDatasets\dataset_v3\Town05_Opt_clear\run_0001" --pred-dir "D:\UAVDatasets\predictions_v3" --mode both
```

天气主要改变 RGB，未额外实现雨天 LiDAR 物理退化。参数含义与灯光接口依据 [CARLA 0.9.16 官方 API](https://carla.readthedocs.io/en/0.9.16/python_api/#carla.WeatherParameters)。夜间曝光、立交桥高度范围及完整训练效果仍需实采验收。
