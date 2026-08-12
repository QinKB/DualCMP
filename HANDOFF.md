# HANDOFF

## 当前任务

- 用户目标已转为：基于 CMP/V2V4Real 设计可投稿 T-ITS 的端到端协同轨迹预测模型；先完成研究设计和独立审查，不写模型代码、不直接启动完整训练。
- 当前候选模型为 CoJTP：融合 BEV/持续查询只负责形成因果在线车辆历史；预测侧保留 `observer × time` 的融合前 actor evidence；每个粗联合未来在 HCER 层内执行场景级来源路由、对象级证据相关、同世界多车交互和轨迹残差精修；最终输出共享场景概率的联合未来分布。
- 唯一整体主张：跨视角遮挡和多车交互不确定性下，预测证据不应在联合未来假设形成前被压成假设无关表示；应把来源—时间证据的最终聚合放入完整场景假设内部。
- 设计文档：`model_design.md`。已纳入 `/home/qkb/Project/CoLIF/docs/colif_failure_postmortem.md` 和 `/home/qkb/Project/CoLIF/HANDOFF.md` 的失败约束。
- 三类独立审查已完成两轮：原“persistent query + aligned fused evidence + actor×world”被判为 CoopTrack/PnPNet-MASAR/QCNeXt 式拼装风险，已废弃为核心主张；改版 HCER 的精确交集尚未发现完全等价工作，但必须通过 A/B/C/D、粗未来条件、来源/时间干预和联合场景指标，不能把普通 mode attention 或 source routing 称为首次。
- 理论审查要求已写入：历史/未来统一坐标、GT/mask 不进入 forward、`grid_sample` 连续采样、有界 Laplace/von-Mises 参数、scene-wise logsumexp、运行时 candidate mask 与 anti-leak 测试。
- 工程审查要求已写入：CAV0 `Car` canonical GT、strict split 重训前端、精简无 `detach` CoBEVT core、逐帧压缩 source token、持续查询 Stage-1 门禁，以及真实 11 帧 future-only BPTT/显存门禁。当前不能声称可直接完整训练。
- 当前没有实现新模型代码、没有运行长训练、没有使用 test 选择新模型。

## 历史复现任务

- 旧目标：在 conda 的 `coop` 环境中，使用 `/home/qkb/Project/data/V2V4Real` 与固定官方 CoBEVT 感知 checkpoint，从训练开始复现 CMP 的 no-aggregation MTR。
- 旧任务状态（2026-08-07 18:51）：no-aggregation MTR 双卡 30 epoch 训练已完成；epoch 5、28、29、30 均已在同一 test cache 上完成独立单卡评测。按 5 秒 minFDE，epoch 5 独立快照为已测试 checkpoint 中最优（7.5559）；末轮 epoch 30 为 10.3258。聚合器未启动。

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

### 训练复现审计（已完成：轨迹预测阶段）

- 实际训练入口为 `MTR/tools/train_multiego.py`：训练循环执行 `loss.backward()` 和 `optimizer.step()`，并保存带 optimizer state 的 checkpoint。
- 配置表明作者采用两阶段初始化：`v2v4real_multiego_cobevt_c256_no_agg.yaml` 的 `PRETRAINED_MOTION_TRANSFORMER` 为空，应从随机初始化训练；聚合版 `v2v4real_multiego_cobevt_c256.yaml` 会加载 no-aggregation MTR，并对 MTR 使用 1/100 学习率、对 aggregation 模块使用完整学习率。第二阶段将指向本机第一阶段的自训 checkpoint，而不是官方 no-aggregation 权重。
- 用户于 2026-08-07 明确选择固定官方 CoBEVT、只训练轨迹预测；官方 `net_epoch75.pth` 位于 `pretrained/v2v4real/point_pillar_cobevt_multiego_256x/`，大小 739,231,572 bytes。该模型生成的 17,902 个融合特征、64 个 train tracking pickle、18 个 test tracking pickle，以及对应 MTR cache 已实际核验存在。
- 已完成 11 帧真实 train 感知 smoke：官方 epoch 75 CoBEVT checkpoint 可加载并前向。为防止 smoke 污染后续训练，本任务修正了 `--max_frames`：它不再生成不完整的 tracking cache。该 smoke 未写 detection/tracking cache，留下的少量 train 融合特征可由完整预处理安全复用。
- train/test 目录有一个同名场景 `testoutput_CAV_data_2022-03-15-09-54-40_0`；比对结果为 294 个 YAML、294 个 PCD 和目录差异均为 0，即原始数据集本身的重复场景。融合特征的同名文件因此语义一致；为复现作者的原始切分，本任务不擅自移除该重复，但将其标记为数据泄露风险。
- 双卡训练可使用 `DistributedSampler` 和 DDP；但 `eval_one_epoch_custom` 没有跨进程聚合。`train_multiego.py` 的 `--not_eval_with_train` 原先仍会在训练后调用该分布式评测，现已修正为完整跳过它。双卡训练只生成 checkpoint，最终指标统一用单卡完整评测产生。
- 完整 train 预处理已以退出码 0 完成：17,902 个唯一融合特征（约 105 GB），检测 cache 为 25,789,678 bytes，64 个 tracking pickle 与 64 个 GT pickle 名称完全一致；多出的 `tracking/train/log.txt` 不是轨迹数据。MTR train cache 含 3,684 个记录，分布为 1 CAV：2,586，2 CAV：1,098。
- 发现并修正了上游 V2V4Real batching 根因：`collate_batch` 只使用 `batch_list[0]`，所以 batch_size>1 原先会静默丢弃其余场景。现在显式拒绝 batch_size>1；真实 4-item loader 已验证报出该错误。为在两张卡上保持作者 8 卡×每卡 1 场景的等效全局 batch=8，训练入口新增 `--grad_accum_steps`，每卡 1 个场景累积 4 步后更新一次。
- 训练信号验证：no-aggregation 模型的真实 batch 为 2 个 CAV、每 CAV `(1,256,48,128)` 融合特征，`obj_trajs=(2,1,11,22)`；loss=63.3177 且有限，456 个参数梯度张量均有限，完成 AdamW step。4 个真实 micro-batch 的累积 smoke 完成 1 次更新，参数绝对变化和为 0.53487045。
- 正式 no-aggregation MTR 训练已启动：`CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop torchrun --standalone --nproc_per_node=2 MTR/tools/train_multiego.py --cfg_file MTR/tools/cfgs/v2v4real/v2v4real_multiego_cobevt_c256_no_agg.yaml --batch_size 2 --workers 2 --launcher pytorch --fix_random_seed 42 --extra_tag trajectory_only_seed42 --not_eval_with_train --grad_accum_steps 4 --max_ckpt_save_num 3 --ckpt_save_interval 1 --logger_iter_interval 50`（另设项目内 `PYTHONPATH`）。日志确认从随机初始化开始，`PRETRAINED_MOTION_TRANSFORMER` 为空，加载 3,684 条 train cache 和 2,858 条 test cache。每 rank 有 1,842 个 micro-batch，累积 4 步后为 461 次 optimizer 更新/epoch，等效全局 batch=8。
- 第 1/30 epoch 于 14:32:46 完成，双卡显存约 1.98/1.85 GB、稳定吞吐约 0.55 秒/更新，训练 loss 均为有限值。`MTR/output/v2v4real_multiego_cobevt_c256_no_agg/trajectory_only_seed42/ckpt/checkpoint_epoch_1.pth` 为 699,853,318 bytes；在 CPU 上加载得到 epoch=1、iter=461、880 个模型张量及 optimizer state。第 2 个 epoch 已于 14:32:47 开始。
- 训练以 `--max_ckpt_save_num 3` 仅保留最近 3 个 checkpoint；因此 checkpoint epoch 1 在 epoch 4 保存后被自动清理。用户在其被清理后请求 epoch 1 FDE，无法再诚实评测该具体权重。随后尝试 epoch 2 测试时，epoch 5 已于 14:49:37 保存并清理 epoch 2；测试于 14:49:47 启动，实际报 `FileNotFoundError`，没有进入推理或产生指标。
- 为避免与训练清理并发，已将 epoch 5（epoch=5、iter=2305）checkpoint 复制为独立快照后单卡完整评测，配置为 `v2v4real_multiego_cobevt_c256_no_agg.yaml`，checkpoint 880 个键严格加载、missing/unexpected 均为 0。2026-08-07 14:50:49--14:57:36 对 2,858 条 test record、29,500 个预测目标完成评测（Filtered=0）：minADE 1/3/5s = 1.3007 / 3.0554 / 4.4001；minFDE 1/3/5s = 2.4689 / 5.4988 / 7.5559；MR@3.6/3.0/2.0/1.8 = 0.3327 / 0.3447 / 0.3775 / 0.3834。日志为 `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch5_test_20260807/eval/log_eval_20260807-145046.txt`。这是训练中途的 no-aggregation 基线，不是最终 CMP 聚合结果。
- 正式 no-aggregation MTR 训练在 2026-08-07 16:36:16 正常结束：日志最后记录为 `epoch 29/30, acc_iter=13830, cur_iter=460/461`，随后输出 `End training`。因正式命令带有 `--not_eval_with_train`，训练入口明确跳过了最终评测；这避免了该脚本不正确的分布式指标汇总，最终指标均由下述独立单卡 test 产生。
- 训练自动保留的 epoch 28/29/30 checkpoint 与 epoch 5 独立快照均已严格加载（880 个模型键、missing/unexpected 均为 0），并在相同的 2,858 条 test record、29,500 个预测目标上完成评测（Filtered=0）。其中 epoch 28、29 分别在 GPU 0、GPU 1 上独立执行，epoch 30 在 GPU 0 上独立执行；每次评测均为单进程，未混合各 GPU 的样本或指标。

  | checkpoint | iter | minADE 1/3/5s | minFDE 1/3/5s | MR@3.6/3.0/2.0/1.8 |
  | --- | ---: | --- | --- | --- |
  | epoch 5 snapshot | 2,305 | 1.3007 / 3.0554 / 4.4001 | 2.4689 / 5.4988 / **7.5559** | 0.3327 / 0.3447 / 0.3775 / 0.3834 |
  | epoch 28 | 12,908 | 1.1806 / 3.1383 / 4.9348 | 2.1798 / 5.9895 / 9.4701 | 0.6317 / 0.6729 / 0.7456 / 0.7580 |
  | epoch 29 | 13,369 | 1.1875 / 3.1478 / 5.0544 | 2.1683 / 6.1815 / 9.9534 | 0.6140 / 0.6447 / 0.7178 / 0.7325 |
  | epoch 30 | 13,830 | 1.2384 / 3.2854 / 5.2554 | 2.2775 / 6.3733 / 10.3258 | 0.6433 / 0.6621 / 0.7300 / 0.7519 |

- 训练日志：`MTR/output/v2v4real_multiego_cobevt_c256_no_agg/trajectory_only_seed42/log_train_20260807-142831.txt`。epoch 28/29/30 的评测日志分别为 `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch28_test_20260807/eval/log_eval_20260807-184238.txt`、`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch29_test_20260807/eval/log_eval_20260807-184238.txt`、`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch30_test_20260807/eval/log_eval_20260807-184746.txt`。
- 在已评测的 checkpoint 中，epoch 5 是 5 秒 minFDE、5 秒 minADE 和全部报告 MR 阈值的最佳者；epoch 28 的 1 秒 minADE（1.1806）及 epoch 29 的 1 秒 minFDE（2.1683）略低，但均未改善 5 秒终点误差。故若后续只按 5 秒 minFDE 使用本阶段模型，应使用独立快照 `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch2_test_20260807/checkpoint_epoch_5_snapshot.pth`，而非末轮 checkpoint。
- CoBEVT 从随机初始化的真实训练 smoke（batch=4）已通过：`record_len=(4,)=[1,1,1,1]`、voxel 特征 `(23676,32,4)`，输出 `psm=(4,2,48,128)`、`rm=(4,14,48,128)`、融合特征 `(4,256,48,128)`；loss=1314.1504 有限，159 个梯度张量均有限，Adam step 后参数绝对变化和为 230.2341，峰值显存 6036.7 MB。
- CoBEVT 双卡 DDP 真实 batch smoke 已以退出码 0 完成：每 rank batch=4、全局 batch=8，rank 0/1 loss 分别为 1819.8065/1566.5241，159 个梯度张量均有限，全局峰值显存为 6749.5 MB。PyTorch 仅报告一个卷积梯度 bucket stride 的性能警告，不影响数值或更新正确性。
- 首次 CoBEVT 80 epoch 训练在 epoch 0 第 3 个混合 CAV batch 以退出码 1 终止，未生成 epoch checkpoint。根因是 `collate_batch_train` 把只供 batch=1 tracking 推理使用、长度随 in-range CAV 数变化的 `ego_transformation_matrix` 列表做 `np.array`/stack；实际混合 batch 为 `[2,2,1,2]`，因而触发不等长数组错误。模型 forward 与检测 loss 对该字段均无读取路径。
- 修复后，`collate_batch_train` 仅在 `len(batch)==1` 时保留与原先相同的 tensor `transformation_matrix_list`，多样本训练 batch 不再携带该未消费字段。用原训练的 `DistributedSampler.set_epoch(0)` 顺序在双卡真实模型上执行 3 次更新：rank 1 的第 3 个 batch 正是 `[2,2,1,2]`，两 rank 均完成前向、loss、反向和 step；159 个梯度张量均有限，峰值显存为 8722.7 MB。单样本 test collate 保持矩阵形状 `(1,1,4,4)`。
- 修复后一次 CoBEVT 重新启动已正常越过第 38 个训练 step，GPU 利用率为 100%，显存约 18.7/19.4 GB，loss 有限；它在 epoch 0 尚未生成 checkpoint 时被有意停止，因为原 `opencood/tools/train.py` 没有对 Python、NumPy 和 PyTorch 设置训练随机种子。现已添加 `--seed`：每个 DDP process 使用 `base_seed + rank`（本次为 42/43），模型会在 DDP 建立时由 rank 0 同步参数，数据增强/worker 随机流则按 rank 分离。`--help` 与 `py_compile` 均已验证该入口。
- seed=42 的后续重启正常运行至第 337 step（loss 仍有限），但审计发现验证循环将 rank 0 的半个 test sampler 均值直接写为全量验证 loss，且两个 rank 会竞争创建 run-dir/config 与写同一 `net_epoch*.pth`。该运行也在 epoch 0、无 checkpoint 时有意停止。现在只有 rank 0 创建目录、写 config/TensorBoard/checkpoint，目录通过 DDP object broadcast 传给另一 rank；每个 epoch checkpoint 写完后 barrier；验证 loss 按 `sum(loss) / count` 的 float64 all-reduce 聚合。双卡最小测试实际验证了路径 broadcast 和 loss sum/count all-reduce，两个 rank 都得到路径一致和全局均值 1.5。
- 正式 CoBEVT 运行命令：`PYTHONPATH=/home/qkb/Project/CMP CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop torchrun --standalone --nproc_per_node=2 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_cobevt_multiego_256x_v2v4real.yaml --dataset_name v2v4real --seed 42`。输出目录为 `opencood/logs/point_pillar_cobevt_v2v4real_2026_08_07_13_44_41`，其 `config.yaml` 已实际记录 `train_params.seed: 42`。
- 正式 run 的 epoch 1 已完成：1,776 个每-rank train step 后仅由 rank 0 写出 `net_epoch1.pth`（739,231,489 bytes，2026-08-07 14:12:00）。跨两 rank、共 3,986 个 validation records 聚合的检测 validation loss 为 `2.0913016796`（2026-08-07 14:15:51）；不是 rank 0 半个 split 的局部均值。此后已进入第 2 个 epoch，实时 rank-0 的 regression/confidence loss 仍有限。
- `net_epoch1.pth` 的 checkpoint load 已在 CPU 上严格验证：state dict 含 259 个键，按当前 CoBEVT c256 配置构建的模型 `strict=True` 加载后 missing/unexpected keys 均为 0。

## 修改内容

- `.gitignore`：忽略本机解释器生成的扩展 `.so` 和 MTR 版本文件，避免将环境构建产物提交。
- `MTR/tools/cfgs/v2v4real/*.yaml`：改为本机数据及 cache 路径。
- `opencood/hypes_yaml/*v2v4real.yaml`：改为本机数据路径。
- `MTR/tools/eval_utils/eval_utils.py`：移除不参与 V2V4Real 自定义评测、却会强制导入缺失 Waymo/TensorFlow 依赖的顶层 Waymo evaluator import；数据集的原有 Waymo evaluation 路径保留其惰性导入。
- `MTR/mtr/models_v2v4real/context_encoder/mtr_encoder.py`：移除不参与上下文编码器的 `this` 与 transformers 导入，避免启动时打印 Zen of Python 和引入无关依赖。
- `opencood/tools/multi_ego_inference_v2v4real.py`：增加 `--refresh_perception_features` / `--max_frames`；修正融合特征实际 `.pkl.npy` 名称检查、缓存命中时丢失检测元数据、`None` 输出路径和跟踪缓存误读检测缓存的问题；默认行为仍可使用已有缓存。
- `MTR/tools/train_multiego.py`：使 `--not_eval_with_train` 同时跳过训练结束后的最终评测，避免 DDP 下未聚合的自定义指标被误当作完整结果。
- `MTR/mtr/datasets/v2v4real_multiego_dataset.py`：拒绝不受支持的 batch_size>1，避免静默丢弃场景。
- `opencood/data_utils/v2v4real/datasets/intermediate_fusion_dataset_multi_ego.py`：训练时不堆叠不等长且不参与模型/loss 的 tracking 变换矩阵；仍为 batch-size-one 推理保留原有 tensor 接口。
- `opencood/tools/train.py`：增加 `--seed`，在建立 dataset/model 前初始化 Python、NumPy 与 PyTorch 的每-rank 随机状态。
- `opencood/tools/train.py`：DDP 下只由 rank 0 建立 run 目录、写 TensorBoard/checkpoint，路径广播并在写 checkpoint 后同步；验证 loss 通过跨 rank 的 sum/count all-reduce 汇总，避免将半个 validation split 误报为完整指标。
- `MTR/tools/train_multiego.py` 与 `MTR/tools/train_utils/train_utils.py`：加入正确缩放 loss、只在每个累积组末尾更新参数/调度器、DDP 中间 micro-batch 不同步梯度的 `--grad_accum_steps` 支持。
- YAML 静态检查：MTR 的 4 个 V2V4Real 配置及 OpenCOOD 的 CoBEVT 1x、CoBEVT 256x、SinBEVT 配置均可由 PyYAML 解析；`point_pillar_v2vnet_multiego_v2v4real.yaml` 在当前和 `de0f839` 基线中都在 `preprocess.args.voxel_size` 存在同一处过度缩进，因而都不能解析。本次只替换了其数据路径，没有修复未使用的上游 V2VNet 配置语义。

## 未完成

- CMP 预测聚合器未启动；这是用户选择“只训练轨迹预测”后的有意范围限制，因此当前结果不能称为完整两阶段 CMP 聚合模型的结果。
- 未保存或评测 epoch 1--4、6--27；因此 epoch 5 只能称为“已评测 checkpoint 中按 5 秒 minFDE 最优”，不能称为 30 个 epoch 的全局最优。若需要无测试集选择偏差的模型选择，应另设验证划分或在下一次训练中保存/验证全部 checkpoint。
- 未对 Waymo 数据集执行测试；它不属于本次 V2V4Real 任务，并且环境没有安装 Waymo/TensorFlow 栈。

## 已知问题与风险

- 上游 V2V4Real 自定义 evaluator 的评测逻辑沿用 OPV2V 的 prediction-dict 生成路径；它已在真实 V2V4Real smoke 与完整缓存构建中执行，但未改写其算法语义。
- 原始 train/test split 含一个 294 帧的完全重复场景；若研究目标是严格隔离的泛化评测，应另行设计去重 split，而不能把本任务的作者协议指标直接解释为无泄露泛化性能。
- 项目文档中的 V2V4Real 评测 `--batch_size 8` 与该 dataset 的单场景 collate 实现不兼容；本任务的先前完整评测和后续训练都使用每进程 batch_size=1，避免遗漏数据。
- `opencood/hypes_yaml/point_pillar_v2vnet_multiego_v2v4real.yaml` 是基线已有的无效 YAML，未用于本次 CoBEVT c256 复现；如需运行 V2VNet 变体，应先单独修复并验证其 `voxel_size` 缩进。
- 全量融合特征约 24 GB、MTR checkpoint 约 2.5 GB，均为可再下载/再生成的忽略文件，不随 Git 提交。
- 单样本 smoke 指标仅用于通路验证；完整指标才是本任务的复现结果，但它们是否与论文数值完全一致仍需在相同数据版本、评测协议和权重来源下另行逐项比对。
- 两阶段训练的完整时长取决于 train record 数、batch size 和双卡吞吐；在完成单 batch 反向和 DDP 采样审计前，不应估计为已完成或直接引用官方 checkpoint 代替自训结果。
- 训练 checkpoint 当前只保留最近 3 个；若需在训练进行中评测特定 epoch，必须先复制其 checkpoint 到独立结果目录，避免与下一次 epoch 保存的自动清理发生竞态。
- 这次为了回答用户的 checkpoint 性能问题，使用 test cache 比较了 epoch 5、28、29、30；这可以证明末期 checkpoint 相对已测试 epoch 5 退化，但不应把该比较作为无偏的 test-set model-selection 协议或论文最终泛化结论。
- 固定的官方 CoBEVT checkpoint 与数据版本是本次轨迹预测复现的外部前提；因此最终结果只能支持“在官方感知输出上的轨迹预测复现”，不能声称感知模块从随机初始化已复现。

## 下一步

1. 当前“只训练轨迹预测”目标已完成；如需使用本阶段模型且以 5 秒 minFDE 为准，使用保存的 epoch 5 独立快照，不要默认使用 epoch 30。
2. 如需正式选择全局最优的 no-aggregation checkpoint，重新训练时应保存每轮 checkpoint，并在独立 validation split 上选择；test 仅用于一次最终报告。
3. 只有用户重新授权完整 CMP 两阶段复现时，才以选定的 no-aggregation checkpoint 初始化并训练预测聚合器，然后单卡完整评测。

## Git 状态

- 当前分支：`codex/cmp-v2v4real-repro`。
- 修改前基线：`main` 的 `de0f839`。
- 已提交的相关检查点：`9491046`（本机路径/缓存/推理修复）、`92997c6`（smoke cache 安全）、`3e84a44`（跳过不安全的分布式评测）、`4c5ffb0`（MTR batch 语义与梯度累积）、`889abe8`（CoBEVT 混合 CAV batch）、`d7ee7da`（CoBEVT seed）、`aff428b`（DDP run/checkpoint 单写与验证聚合）、`f86d0e1`（切换为官方 CoBEVT + 自训轨迹预测）、`667f9ba`（记录训练启动）、`44992e9`（记录 epoch 5 评测）。下载权重、缓存、特征和运行日志均在 Git 忽略路径中。
- 恢复方式（提交后）：`git revert <任务提交>`；或切回 `main` 查看原始版本。不得删除生成数据，除非用户另行授权。

## 更新时间

- 2026-08-07，本次 CMP V2V4Real 复现任务；用户已切换为官方 CoBEVT + 自训轨迹预测。no-aggregation MTR 的 30 epoch 训练及 epoch 5/28/29/30 完整单卡评测均已完成，聚合器未训练。
