# HANDOFF

## 当前任务

- 用户目标：在 conda 的 `coop` 环境中，使用 `/home/qkb/Project/data/V2V4Real` 从训练阶段复现 CMP 的 V2V4Real 流程。
- 验收标准：依赖和原生算子可导入；train/test 的感知、跟踪、轨迹和融合特征数据可用；先完成 no-aggregation MTR 训练，再用其初始化 CMP 聚合模型训练；最终对自训 checkpoint 做完整评测。
- 当前状态（2026-08-07）：官方权重的完整测试已完成；训练数据准备和真实 train 样本 smoke 已开始，完整训练尚未启动。

## 背景与动机

- 原仓库配置把 V2V4Real 指向 `/data1/Datasets/V2V4Real`，缓存也包含 `/data3` 的旧绝对路径，不能直接用于本机数据目录。
- 目标数据位于 `/home/qkb/Project/data/V2V4Real`。实际检查到 train 有 14,210 个 YAML/PCD 帧对，test 有 3,986 个 YAML/PCD 帧对，test 分为 9 个场景。
- `docs/prepare_dataset_checkpoints.md` 指定了 V2V4Real 感知权重和 MTR 权重的官方 Google Drive 目录；仓库本身没有所需 checkpoint 或完整的 CoBEVT 融合特征。

## 已完成

### Git 与参考实现审计

- 修改前：`main` 分支，提交 `de0f839`，CMP 工作区干净。
- 已创建隔离任务分支：`codex/cmp-v2v4real-repro`。
- 已检查 `/home/qkb/Project/Reference/CMP`：与当前 CMP 同提交；其中没有可复用 checkpoint 或融合特征。该参考目录有一个用户未跟踪的论文 PDF，未作任何修改。
- 项目起初没有 `HANDOFF.md`；本文件由本任务创建。

### 环境审计与修复

- 实际环境：`coop`，Python 3.10.20，PyTorch 2.1.1+cu121，torchvision 0.16.1+cu121，torch-geometric 2.6.1，spconv 2.3.6，Open3D 0.19.0；机器有两张 RTX 3090。
- 补齐 CMP 直接导入所缺的 `tensorboardX==2.6.4`、`timm==0.9.16`、`Pympler==1.1`，并安装 `gdown==5.2.0` 以取得官方权重。
- 在当前 Python 3.10 / CUDA 12.4 工具链下重建 MTR 的 KNN 与 attention CUDA 扩展（架构 8.6），并重建 OpenCOOD `box_overlaps` 扩展；实际导入和 GPU 前向均已成功。
- `pip check` 仅留下环境中原有且与 CMP 无调用关系的冲突：`openxlab 0.1.3` 要求 `rich~=13.4.2`，当前为 `rich 15.0.0`。未擅自修改共享环境中的无关包。
- CMP 文档声称的 Python 3.7.11 / torch 1.13.1 与实际可用环境不同；当前运行结果确认这套 Python 3.10 / torch 2.1.1 组合可用。

### 权重与本地路径

- 按文档的官方 Drive 清单下载了以下文件（均被 Git 忽略）：
  - `pretrained/v2v4real/point_pillar_cobevt_multiego_256x/net_epoch75.pth`，739,231,572 bytes，Drive file id `1AkGmL4yWbUfnUviPDq8Wfrmm6z7K6cxn`。
  - `MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth`，2,540,101,497 bytes，Drive file id `1KEKbNa0tu3CTb35415lxDTKPVuRissnI`。
  - 同时下载了文档中提供的 no-aggregation 对照权重，未用于本次主评测。
- 四个 MTR V2V4Real 配置与四个 OpenCOOD V2V4Real 配置已改为本机数据根目录，并把 MTR cache 设为 `/home/qkb/Project/data/V2V4Real/derived/CMP/cache`，避免读写旧机器路径。

### 真实数据、数据流与最小验证

- 原始 YAML 包含 `gps`、4x4 `lidar_pose` 及车辆的角度、中心、尺寸、位置和类型字段；GT 与跟踪 pickle 的样本时间轴为 147 帧，使用末列有效 mask。
- 已用官方下载的 CoBEVT 配置构建实际 OpenCOOD test loader：共 3,986 个记录；单样本 `origin_lidar` 为 `(25731, 5)`，batch voxel 特征为 `(5530, 32, 4)`。
- 数据流：原始 V2V4Real YAML/PCD → OpenCOOD CoBEVT 感知网络 → 每帧融合特征 `.npy` → V2V4Real MTR dataset 的历史轨迹、future mask 与融合特征 → MTR 编码器/解码器 → minADE、minFDE、MR。
- 通过真实单样本 smoke cache 验证：`obj_trajs=(1,1,11,22)`、`obj_trajs_mask=(1,1,11)`、future 为 `(1,50,2)`，融合特征为 `(1,256,48,128)`，均无 NaN/Inf。
- 完整感知阶段于本次任务中实际完成：3,986 / 3,986 个特征文件，目录约 24 GB。首、中、末三个样本均为 `float32` 的 `(1,256,48,128)` 且所有值有限。
- 官方 c256 MTR checkpoint 已实际加载：250,669,208 个模型参数、1,059 个磁盘键、缺失键 0、意外键 0。
- 单 batch 的真实端到端 smoke 已完成（只含 1 样本，不代表完整数据集性能）：minADE 1/3/5s = 0.2515 / 0.3355 / 1.0657；minFDE 1/3/5s = 0.2361 / 1.0137 / 4.0619；MR@3.6 = 1。日志：`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_smoke_20260807/eval/log_eval_20260807-115323.txt`。

### 当前完整评测

- 实际命令：

  `PYTHONPATH=/home/qkb/Project/CMP:/home/qkb/Project/CMP/MTR CUDA_VISIBLE_DEVICES=0 conda run -n coop python MTR/tools/test_multiego.py --cfg_file MTR/tools/cfgs/v2v4real/v2v4real_multiego_cobevt_c256.yaml --batch_size 1 --workers 0 --launcher none --fix_random_seed --ckpt MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth --output_dir /home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_full_20260807`

- 数据验证阶段已于 2026-08-07 12:06:17 结束，生成 2,858 个可评估记录，分布为 2 CAV：2,604，1 CAV：254。
- 当前为单 GPU 执行：上游自定义评测没有跨进程指标汇总，且脚本中的分布式路径硬编码为 8 GPU；为避免重复/遗漏样本和错误指标，本次没有强行改为双卡。
- 完整运行的退出码为 0。GPU 推理进度为 2,858 / 2,858，耗时约 4 分 20 秒；日志的指标聚合包含 29,500 个预测目标、Filtered=0：minADE 1/3/5s = 1.1375 / 2.6684 / 3.9257，minFDE 1/3/5s = 2.0376 / 4.8467 / 7.0533，MR@3.6/3.0/2.0/1.8 = 0.3299 / 0.3393 / 0.3628 / 0.3673。
- 完整日志：`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_full_20260807/eval/log_eval_20260807-120255.txt`。

### 训练复现审计（进行中）

- 实际训练入口为 `MTR/tools/train_multiego.py`：训练循环执行 `loss.backward()` 和 `optimizer.step()`，并保存带 optimizer state 的 checkpoint。
- 配置表明作者采用两阶段初始化：`v2v4real_multiego_cobevt_c256_no_agg.yaml` 的 `PRETRAINED_MOTION_TRANSFORMER` 为空，应从随机初始化训练；聚合版 `v2v4real_multiego_cobevt_c256.yaml` 会加载 no-aggregation MTR，并对 MTR 使用 1/100 学习率、对 aggregation 模块使用完整学习率。为“从训练开始”复现，第二阶段将指向本机第一阶段的自训 checkpoint，而不是官方 no-aggregation 权重。
- 审计时发现 train 缺少 CoBEVT 融合特征和 AB3DMOT 跟踪轨迹；GT train 数据已存在（64 个 scenario-CAV pickle）。训练感知 loader 有 32 个场景、14,210 个样本。
- 已完成 11 帧真实 train 感知 smoke：官方 epoch 75 CoBEVT checkpoint 可加载并前向。为防止 smoke 污染后续训练，本任务修正了 `--max_frames`：它不再生成不完整的 tracking cache。该 smoke 未写 detection/tracking cache，留下的少量 train 融合特征可由完整预处理安全复用。
- train/test 目录有一个同名场景 `testoutput_CAV_data_2022-03-15-09-54-40_0`；比对结果为 294 个 YAML、294 个 PCD 和目录差异均为 0，即原始数据集本身的重复场景。融合特征的同名文件因此语义一致；为复现作者的原始切分，本任务不擅自移除该重复，但将其标记为数据泄露风险。
- 双卡训练可使用 `DistributedSampler` 和 DDP；但 `eval_one_epoch_custom` 没有跨进程聚合。`train_multiego.py` 的 `--not_eval_with_train` 原先仍会在训练后调用该分布式评测，现已修正为完整跳过它。双卡训练只生成 checkpoint，最终指标统一用单卡完整评测产生。

## 修改内容

- `.gitignore`：忽略本机解释器生成的扩展 `.so` 和 MTR 版本文件，避免将环境构建产物提交。
- `MTR/tools/cfgs/v2v4real/*.yaml`：改为本机数据及 cache 路径。
- `opencood/hypes_yaml/*v2v4real.yaml`：改为本机数据路径。
- `MTR/tools/eval_utils/eval_utils.py`：移除不参与 V2V4Real 自定义评测、却会强制导入缺失 Waymo/TensorFlow 依赖的顶层 Waymo evaluator import；数据集的原有 Waymo evaluation 路径保留其惰性导入。
- `MTR/mtr/models_v2v4real/context_encoder/mtr_encoder.py`：移除不参与上下文编码器的 `this` 与 transformers 导入，避免启动时打印 Zen of Python 和引入无关依赖。
- `opencood/tools/multi_ego_inference_v2v4real.py`：增加 `--refresh_perception_features` / `--max_frames`；修正融合特征实际 `.pkl.npy` 名称检查、缓存命中时丢失检测元数据、`None` 输出路径和跟踪缓存误读检测缓存的问题；默认行为仍可使用已有缓存。
- `MTR/tools/train_multiego.py`：使 `--not_eval_with_train` 同时跳过训练结束后的最终评测，避免 DDP 下未聚合的自定义指标被误当作完整结果。
- YAML 静态检查：MTR 的 4 个 V2V4Real 配置及 OpenCOOD 的 CoBEVT 1x、CoBEVT 256x、SinBEVT 配置均可由 PyYAML 解析；`point_pillar_v2vnet_multiego_v2v4real.yaml` 在当前和 `de0f839` 基线中都在 `preprocess.args.voxel_size` 存在同一处过度缩进，因而都不能解析。本次只替换了其数据路径，没有修复未使用的上游 V2VNet 配置语义。

## 未完成

- train 集完整 CoBEVT 融合特征、检测缓存和 AB3DMOT 跟踪轨迹尚未生成。
- no-aggregation 和 CMP 聚合模型的自训 checkpoint 尚未生成，因而尚未对自训模型进行完整测试。
- 未对 Waymo 数据集执行测试；它不属于本次 V2V4Real 任务，并且环境没有安装 Waymo/TensorFlow 栈。

## 已知问题与风险

- 上游 V2V4Real 自定义 evaluator 的评测逻辑沿用 OPV2V 的 prediction-dict 生成路径；它已在真实 V2V4Real smoke 与完整缓存构建中执行，但未改写其算法语义。
- 原始 train/test split 含一个 294 帧的完全重复场景；若研究目标是严格隔离的泛化评测，应另行设计去重 split，而不能把本任务的作者协议指标直接解释为无泄露泛化性能。
- `opencood/hypes_yaml/point_pillar_v2vnet_multiego_v2v4real.yaml` 是基线已有的无效 YAML，未用于本次 CoBEVT c256 复现；如需运行 V2VNet 变体，应先单独修复并验证其 `voxel_size` 缩进。
- 全量融合特征约 24 GB、MTR checkpoint 约 2.5 GB，均为可再下载/再生成的忽略文件，不随 Git 提交。
- 单样本 smoke 指标仅用于通路验证；完整指标才是本任务的复现结果，但它们是否与论文数值完全一致仍需在相同数据版本、评测协议和权重来源下另行逐项比对。
- 两阶段训练的完整时长取决于 train record 数、batch size 和双卡吞吐；在完成单 batch 反向和 DDP 采样审计前，不应估计为已完成或直接引用官方 checkpoint 代替自训结果。

## 下一步

1. 完成 train 集感知、检测和跟踪预处理，并核验文件数、形状、mask 与 train cache 记录数。
2. 以真实 batch 完成 no-aggregation MTR 的前向、损失、反向、梯度和 checkpoint smoke。
3. 核验双卡 DDP 的 sampler、batch size、checkpoint 与指标聚合后，启动两阶段完整训练；最终仅使用自训 checkpoint 评测。

## Git 状态

- 当前分支：`codex/cmp-v2v4real-repro`。
- 修改前基线：`main` 的 `de0f839`。
- 本次源码、配置与本文件作为一个独立任务提交，提交信息为 `fix: run CMP V2V4Real reproduction locally`；下载权重、缓存、特征和运行日志均在 Git 忽略路径中。
- 恢复方式（提交后）：`git revert <任务提交>`；或切回 `main` 查看原始版本。不得删除生成数据，除非用户另行授权。

## 更新时间

- 2026-08-07，本次 CMP V2V4Real 复现任务；完整 MTR 评测已以退出码 0 完成。
