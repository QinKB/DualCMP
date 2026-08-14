# CoLIF 输入—输出与执行链审计

本文件对应 `opencood/hypes_yaml/colif/colif_v2v4real.yaml` 的当前 fused-BEV 主路径。形状中的 `B` 是每卡 batch size，`S` 是该 batch 内有效 CAV 总数，`Q=128` 是 query slot 数，`A<=24` 是当前配置允许的 active query 数，`K=6` 是公开预测模态数，`K_raw=64` 是 CMP MTR 的训练 intention 数，`T=50` 是未来帧数。

## 1. 数据与监督边界

训练样本包含 11 个连续 runtime frames：前 7 帧仅建立因果状态，后 4 帧计算 loss。每个 runtime frame 只包含当前时刻可用的多车点云、receiver/source pose、离散时钟和来源可用性；GT identity、current boxes、future trajectories 与 anchor labels 位于独立的 `target_frames`，不会进入 `model.forward`。

每辆 source 的原始点云先由 `inv(receiver_pose) @ source_pose` 投影到当前 receiver LiDAR 坐标，再 voxelize。collate 后的核心输入为：

- `voxel_features: [N_voxel, 32, 4]`
- `voxel_coords: [N_voxel, 4]`，第 0 维是展平后的 source batch index
- `voxel_num_points: [N_voxel]`
- `record_len: [B]`，记录每个 receiver 的有效 source 数
- `receiver_pose: [B,4,4]`
- `receiver_time_s: [B]`

监督帧包含：

- `current_boxes: [B,128,7]`，顺序为 `[x,y,z,h,w,l,yaw]`
- `current_velocity: [B,128,2]`
- `current_mask/actor_ids: [B,128]`
- `future_xy: [B,128,50,2]`，是在当前 receiver frame 中相对当前 GT 中心的位移
- `future_mask: [B,128,50]`
- CMP anchor `fused_label_dict`

## 2. CMP fused-BEV encoder

`PointPillarCoBEVT.encode_fused_bev` 是唯一的 cooperative perception encoder；CoLIF 没有再串接第二个检测模型。

```text
receiver-aligned voxels
  -> PillarVFE                 [S,64,192,512]
  -> BEV Backbone              [S,384,96,256]
  -> shrink + compression      [S,256,48,128]
  -> regroup + CAV mask        [B,2,256,48,128], [B,2]
  -> SwapFusionEncoder         [B,256,48,128]
```

同一 fused BEV 同时进入：

- CMP auxiliary heads：`psm [B,2,48,128]`、`rm [B,14,48,128]`
- persistent query decoder：不先 decode/NMS 成检测框，不经过 AB3DMOT

## 3. Persistent query state 与 decoder

跨帧状态为：

- `query_feat: [B,128,256]`
- `ref_box: [B,128,7]`
- `velocity: [B,128,2]`
- `active_mask/age/miss_count/runtime_id: [B,128]`
- `history_feat: [B,128,11,256]`
- `history_box: [B,128,11,7]`
- `history_mask: [B,128,11]`
- 可选 future cue：`[B,128,256]`

进入新帧前，所有 reference/history boxes 都由上一 receiver frame 刚体变换到当前 receiver frame；reference 再只用当前 velocity 和时间差作一步传播。tracker 的 temporal GRU 保持 Phase-1 的最后 4 帧输入合同，MTR 单独读取完整 11 槽历史。

`PersistentQueryDecoder` 的输入/输出为：

```text
输入：fused BEV [B,256,48,128]
     query prior [B,128,256]
     propagated reference boxes [B,128,7]
  -> MOTR deformable transformer (3 decoder layers)
输出：query feature [B,128,256]
     existence logits [B,128]
     current boxes [B,128,7]
     velocity [B,128,2]
     2层 auxiliary query outputs
```

query existence logit在 query center连续采样同一 CoBEVT `psm`，二者相加后进入 lifecycle。该路径没有检测后处理、NMS、GT matching 或第二套 actor identity。lifecycle 生成 birth/active/missing/dead 状态，并将 active slot 上限设为 24；物理 actor 与 CAV 数不是同一概念。

## 4. Variable-history CMP MTR

每个 active query 从第一帧起即可预测。adapter 对每个 center actor 构造 CMP 原始的 22 属性轨迹输入：center-relative `xyz`、`lwh`、vehicle/center flags、11维 time one-hot、CMP checkpoint 使用的 timestamp scalar、相对 heading 的 sin/cos。无效历史槽严格置零。

若 batch 内共有 `M` 个 active center actor，每个 receiver 最多有 `N` 个 active context actors：

```text
history boxes/masks
  -> packed actor polylines     [M,N,11,22], mask [M,N,11]
  -> CMP MTREncoder             object [M,N,256], center [M,256]
  -> CMP MTRDecoder             raw logits [M,64]
                                raw GMM [M,64,50,5]
  -> endpoint NMS               logits [M,6]
                                public GMM [M,6,50,5]
```

GMM 五个参数为 `[dx,dy,log_sigma_x,log_sigma_y,rho]`。训练 loss 在全部 64 个 intention modes 上计算，正模态按 CMP 的“最后有效 GT goal 到 intention point 的距离”选择；6 模态 NMS 仅用于公开输出/评测。actor-local 均值和完整二维协方差在公开输出时旋回 receiver axes。

当前诊断主配置 `future_adapter.mode=none`：先证明正确历史合同下的 predictor 本身有效。bounded future cue、gate 和 recurrent BPTT 代码仍保留，但不参与本轮无反馈结果；空间 reference 无论开关 feedback 都不会由 5 s trajectory 直接移动。

## 5. Loss、梯度与分布式边界

Phase 1 优化 fused anchor auxiliary detection、query existence/box/velocity 与 sequential persistent matching。Phase 2 冻结全部 CoBEVT/query tracker，只优化 CMP MTR 的 GMM NLL 与 mode classification；无反馈配置不把常数 tracking loss混入优化目标。

Phase 2 使用 CMP 的预训练 BatchNorm running statistics，BN affine及其余 MTR 参数仍可训练。这保证每卡 batch=1、仅一个 age=1 actor/一个有效历史点时仍可前向反向。所有 loss 的分子在本 rank 保持 autograd，分母以 raw count 跨 rank all-reduce；无有效 future 的 actor不制造伪 count。每个 epoch 保存/评测前还会显式广播所有 model buffers，避免最后一个 local batch 后各 rank 的 BN buffer不一致。

FutureAdapter 固定在 FP32 小岛中执行。原因不是性能微调，而是正确性：同一外层 AMP context 中，前7帧 burn-in 使用 `no_grad`、后4帧开启梯度；若其 Linear 层参与 autocast，PyTorch 2.1 会复用 burn-in 阶段缓存的无梯度半精度权重，使 gate/cue MLP 在有反馈的监督帧静默无梯度。现在该路径有专门回归测试，trainer 也记录首 batch 的 kept feedback query-frames、最大 gate/delta 和未覆盖参数。

## 6. 公开输出

每帧模型输出：

- tracking：`track_boxes [B,128,7]`、`track_velocity [B,128,2]`、existence、active mask、runtime IDs、age/miss
- forecast：`pred_scores [B,128,6]`、`pred_trajs [B,128,6,50,5]`、forecast-valid mask、实际历史步数
- training only：64-mode raw score/GMM/intention tensors、query auxiliary outputs、CMP anchor outputs

strict-online evaluator只在 supervision-free forward 完成后，用独立 target branch 做当前帧 Hungarian 计分，并聚合 tracking、coverage、age-bucket、minADE/minFDE/MR。训练和推理均不读取 `ass_id`、GT box、future target 或预生成 tracking pickle。

## 7. 已执行的工程验证

- 全部 880 个 CMP MTR state tensors严格加载：context encoder 166、motion decoder 714，missing/unexpected 均为 0。
- age=1 单 actor、单有效历史点 GPU forward/backward通过；36 个 MTR BN 层均使用冻结统计，产生的梯度有限。
- 真实 V2V4Real 11-frame sample得到 7 runtime burn-in + 4 target frames；runtime GT guard通过。
- 双卡真实 10-batch smoke：首反向未覆盖可训练参数为 0，AMP overflow=0，无信号跳步=0，参数与buffers跨卡 checksum delta=0。
- 冻结 tracker 的400个 state tensors在 Phase-2 optimizer step 后逐 tensor 保持不变。
- `tests/colif`、相关模块 compileall 与 Git whitespace检查结果以 `HANDOFF.md` 顶部最新记录为准。

## 8. 2026-08-12 最终实跑结果

使用两张 RTX 3090、每卡 batch size 1，在真实 V2V4Real train split 上完成 Phase 2 一个完整 epoch：

- 1710 个 global clips，855 个 optimizer updates；
- 每个 clip 为 7 帧 causal burn-in + 4 帧 supervised unroll；
- AMP overflow 0 次、无有效监督跳步 0 次；
- epoch 末参数 checksum 跨卡最大差异 0，显式同步后的 buffer checksum 最大差异 0；
- checkpoint 的 1294 个 model state tensors 中无 NaN/Inf；
- Phase 1 冻结的 CoBEVT、query decoder、temporal memory 和 memory scale 共 400 个 tensors 与初始化 checkpoint 逐 tensor 完全相同；
- 随后自动完成双卡、整场景分片、1993 帧 official test，`strict_online=true` 且 `runtime_uses_ground_truth=false`。

该 epoch 的关键 test 指标为：

- `minADE@5s = 5.3127206163 m`
- `minFDE@5s = 10.7843668489 m`
- `MR@3.6m,5s = 0.7866859027`
- tracking recall / precision = `0.7465033897 / 0.7590436268`
- `MOTA = 0.4581017529`，identity switches = `1684`
- forecast coverage = `0.7483727674`

相对旧 fused Phase-2 已完成 epoch 中 5 s minFDE 最好值 11.7361256393 m，本次降低 0.9517587904 m（8.11%）；5 s minADE 从 5.9129101300 m 降至 5.3127206163 m（10.15%）。这证明本轮输入合同修正产生了可测改善，但仍未达到旧 source-local/no-fuse 或 CMP no-aggregation 的数值，因此不能据此声称论文性能已经成立。

完整产物：

- `opencood/logs/colif_fused_v3_phase2_epoch1_final_20260812/checkpoint_epoch_000.pth`
- `opencood/logs/colif_fused_v3_phase2_epoch1_final_20260812/train_log.jsonl`
- `opencood/logs/colif_fused_v3_phase2_epoch1_final_20260812/metrics_test.jsonl`
- `opencood/logs/colif_fused_v3_phase2_epoch1_final_20260812/runtime_test_2frames.pth`

最后一遍回归结果为 `38 passed`，相关模块 `compileall` 成功，`git diff --check` 成功。独立 GT-free 推理从上述 checkpoint 输出 2 个 chronological records；每帧均只含匿名 runtime state、tracking 与 forecast 结果，所有浮点 tensor 有限。

完整 recurrent-BPTT 分支另做了最终代码双卡2-batch验证。修复前，同一首 batch 虽有64个 query-frame 实际打开feedback、最大gate为0.00450，但cue projection与gate共8个参数没有梯度且10步后权重仍未变化；修复后首batch未覆盖参数为0，第二步后cue MLP、gate首层和末层均发生有限非零更新，0 AMP overflow，parameter/buffer checksum delta均为0。该 smoke 只证明闭环梯度与DDP执行合同，不是性能结果。

## 9. 当前能够与不能够下的结论

已由代码和实跑确认的是：数据坐标、fused-BEV、persistent-query、variable-history MTR、loss、梯度、checkpoint、双卡同步、strict-online evaluation 与无 GT inference 的执行合同已经连通，并通过上述有限范围的回归与完整 epoch 验证。

不能给出数学意义上的“任何输入、任何硬件、任意训练时长均无错误”保证；也不能把一个 epoch 的改善等同于模型优于基线。当前最明显的研究风险仍是 age=1 与 age=2--3 的 5 s FDE 分别为 22.91 m 和 18.49 m，说明“第一证据即可输出预测”已经实现，但短历史预测质量尚未达到论文主张要求。当前 `future_adapter.mode=none`，本结果只验证无反馈 predictor，不验证 future cue 改善后续 tracking 的闭环主张。
