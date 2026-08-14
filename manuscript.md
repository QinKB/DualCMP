# CoLIF: Task-Adaptive Fusion for Persistent Cooperative Motion Forecasting

> **Working manuscript, 2026-08-13.** This draft follows the verified implementation and frozen one-epoch ablations in `/home/qkb/Project/CoLIF`. BibTeX keys are temporary. Claims about the optional trajectory-source refiner are deliberately separated from claims supported by the default model.

## I. Introduction

Accurate motion forecasting is required for an automated vehicle to plan safely around interacting traffic. The difficulty is not limited to extrapolating an actor's kinematics: the predictor must estimate several plausible futures from partial observations while the sensing platform, neighboring actors, and visibility conditions continue to change. Cooperative sensing expands the evidence available to the receiver by combining LiDAR observations from spatially separated connected vehicles, thereby extending the effective field of view beyond that of a single vehicle \cite{xu2023v2v4real}. The central question for cooperative motion prediction is therefore not only whether communication improves current perception, but also whether the shared evidence is represented in a form that is useful for predicting how each actor will move.

Existing systems answer this question at different levels of the autonomy stack. CMP first fuses multi-vehicle bird's-eye-view (BEV) features for detection, constructs histories with a multi-object tracker, predicts trajectories at each vehicle, and then aggregates the resulting predictions \cite{wang2025cmp}. End-to-end systems such as V2XPnP instead perform spatio-temporal feature fusion and jointly decode perception and prediction outputs \cite{zhou2025v2xpnp}. More recent multi-level cooperative architectures further argue that perception, prediction, and planning impose heterogeneous demands on a shared representation and introduce task-specialized or prediction-level fusion \cite{souig2026unimm}. These developments show that cooperative prediction does not universally require a finalized tracking interface. They also expose a more precise design issue: the representation that is effective for estimating the current scene need not be the most informative representation for forecasting individual actors.

Intermediate cooperative perception commonly aligns source features and compresses them into one receiver-centric BEV. This operation is well suited to current object localization because it aggregates complementary evidence into a unified spatial measurement. Once fused, however, the downstream forecaster is not explicitly given the evidence supplied by each source around a particular actor. That distinction can matter because detection and forecasting ask different questions. Detection estimates whether an object occupies a location now, whereas forecasting also uses actor-centered motion history, interaction context, viewing geometry, and local scene evidence relevant to alternative motions. Applying one global fusion result to both tasks therefore couples two different information-selection problems. The limitation is not that fused BEV is intrinsically unsuitable for prediction; rather, a single task-shared fusion operator provides no mechanism for the forecaster to reweight aligned source evidence separately for each actor.

This observation motivates a task-adaptive cooperative interface. The perception path should retain dense multi-view fusion because it provides reliable receiver-centric measurements for detection and tracking. In parallel, the forecasting path should retain the aligned source BEVs long enough for each persistent actor state to query the evidence most relevant to its own geometry and motion. Such a design preserves a single receiver-side actor identity and avoids constructing independent per-vehicle tracks, while allowing perception and prediction to aggregate the same encoded LiDAR features differently. It also follows the broader motion-prediction lesson that local evidence, temporal evolution, and actor interaction should be modeled at their appropriate scales \cite{zhou2022hivt}, and that candidate trajectories can serve as queries for targeted context retrieval \cite{zhou2023qcnet}.

We instantiate this principle in **CoLIF**, a strict-online cooperative motion forecasting framework. Receiver-aligned source point clouds are encoded once by the CMP PointPillar backbone. The original FuseBEVT branch produces the fused BEV and dense detection heatmap used by a persistent query tracker. A parallel prediction branch samples a small patch from every valid source BEV around each active actor and uses actor-to-sensor relative geometry to aggregate these patches without relying on a fixed source index. The resulting cooperative actor context is combined with an ordered, variable-length history and neighboring actor states to produce six Gaussian trajectory modes over five seconds. Because one position observation cannot identify velocity, a query is created at the first supported measurement but becomes forecast-eligible after two causal observations. Forecast outputs do not feed back into tracking: experiments with future-conditioned association, propagation, and query updates did not improve the current implementation, so identity and current-state estimation depend only on current fused measurements and a causal constant-velocity state.

CoLIF retains two forecast outputs for different accuracy--capacity choices. The default output is an actor-source motion proposal whose modes are residuals around robust causal velocity anchors. A higher-capacity trajectory-source refiner additionally treats every mode--time pair as a state query, samples current aligned source features along each proposed trajectory, and predicts a bounded residual to the proposal. The refiner is initialized as an exact identity mapping so that enabling it cannot randomly replace a valid proposal before learning. This branch is retained as a controlled research variant, not assumed to be superior: on the frozen V2V4Real test protocol it improves over removing source evidence, but its overall displacement errors remain above those of the simpler actor-source proposal.

The current experiments isolate the proposed information paths by loading the same persistent-query tracker, freezing it during forecasting training, and using identical data order and actor sets for every ablation. On 1,993 strict-online V2V4Real test frames, the default model attains minADE/minFDE of 0.9997/1.3708 m at 1 s, 1.8472/2.9177 m at 3 s, and 2.5870/4.3804 m at 5 s, with forecast coverage of 0.7894. Relative to the otherwise identical model without actor-conditioned source fusion, it reduces minFDE by 12.96% at 1 s and 5.56% at 3 s, while the 5-s endpoint improvement is only 0.40%. The evidence therefore supports improved short- and mid-horizon cooperative forecasting and lower average error over the full horizon; it does not support a claim of substantial 5-s endpoint improvement from the new fusion module alone.

The main contributions are as follows:

1. We formulate cooperative perception and cooperative forecasting as related but non-identical fusion problems, and introduce a dual-path interface that preserves CMP-style dense fused-BEV measurements for persistent tracking while exposing aligned per-source BEV evidence to the actor-level predictor.

2. We develop a geometry-conditioned actor-source forecasting module that combines source-invariant local evidence, ordered variable-length actor history, and multi-actor interaction with multimodal motion anchors. Controlled ablations with identical tracker outputs demonstrate the separate contributions of source evidence and temporal ordering.

3. We provide an optional identity-initialized trajectory-source state refiner and a strict-online evaluation protocol covering displacement error, miss rate, forecast coverage, actor age, and source-support subsets. The results establish where the additional refiner helps and, equally importantly, where its added capacity does not outperform the simpler proposal.

The remainder of this paper is organized as follows. Section II positions CoLIF relative to cooperative fusion and motion-prediction models. Section III defines the online task and presents the method. Section IV records the current experimental evidence and its claim boundaries.

## II. Related Work

### A. Cooperative Perception and Prediction

Cooperative perception methods exchange raw observations, object outputs, or intermediate features to improve receiver-side scene understanding. Intermediate fusion balances spatial context and communication cost and is the perception backbone used by CMP \cite{wang2025cmp}. CMP extends cooperative perception to motion prediction through a modular detection--tracking--prediction pipeline and a second exchange of predicted trajectories. V2XPnP studies one-step spatio-temporal fusion for end-to-end multi-agent perception and prediction \cite{zhou2025v2xpnp}, while UniMM-V2X uses both perception-level and prediction-level cooperation and task-specialized representations for downstream tasks \cite{souig2026unimm}. CoLIF is narrower than an end-to-end planning stack: it studies how one LiDAR encoding can support dense current-state fusion and actor-conditioned forecasting without forcing both tasks to consume the same fused representation.

### B. Actor-Centric Motion Forecasting

Motion predictors commonly combine temporal actor states, local scene context, and social interaction. HiVT separates local context extraction from global actor interaction and uses relative geometric representations \cite{zhou2022hivt}. QCNet uses query-centric encoding and a proposal--refinement decoder in which proposed trajectories provide adaptive anchors for context retrieval \cite{zhou2023qcnet}. CoLIF adopts neither model directly: V2V4Real provides no HD map in the current setup, and CoLIF's inputs are online persistent-query states rather than ground-truth trajectory histories. We instead use their architectural principles to organize causal actor history, relative source geometry, actor interaction, and the optional proposal-conditioned source sampler.

### C. Joint Tracking and Forecasting

Persistent object queries have been used to connect detection, tracking, and forecasting in single-vehicle systems. Predicted motion can also support occlusion handling or subsequent perception in methods such as PF-Track and ForeSight \cite{pang2023pftrack,papais2025foresight}. These works prevent us from claiming that future-assisted tracking is new. More importantly, our own controlled tests did not validate that mechanism in the present fused-LiDAR setting. The submitted CoLIF design therefore uses persistent queries to maintain actor identity but keeps forecasting as a read-only consumer of the causal tracking state.

## III. Methodology

### A. Online Cooperative Forecasting Problem

At receiver frame (t), let \(\mathcal{S}_t\) be the set of available CAV sources and let \(\mathcal{P}_{s}^{t}\) be the LiDAR point cloud of source \(s\). V2V4Real provides discrete 10-Hz frame sequences rather than measured message-arrival timestamps; the main protocol is therefore synchronous, and any delay or source dropout experiment must be described as a simulated perturbation. Calibration and pose information define a transform \(\mathbf{T}_{s\rightarrow r}^{t}\) from source \(s\) to receiver \(r\). The runtime input is

\[
\mathcal{X}_{r}^{t}=\left\{
\mathcal{P}_{s}^{t},\mathbf{T}_{s\rightarrow r}^{t}
\right\}_{s\in\mathcal{S}_t}.
\tag{1}
\]

Ground-truth boxes, actor identities, source-support labels, and future trajectories are excluded from \(\mathcal{X}_{r}^{t}\). They enter only the training criterion and evaluator. The model maintains a receiver-side state for at most \(Q\) query slots,

\[
\mathcal{B}^{t}=\left\{
\mathbf{q}_{i}^{t},\mathbf{b}_{i}^{t},\mathbf{v}_{i}^{t},
e_i^t,a_i^t,m_i^t,\mathcal{H}_{i}^{t}
\right\}_{i=1}^{Q},
\tag{2}
\]

where \(\mathbf{q}\) is query content, \(\mathbf{b}\in\mathbb{R}^{7}\) is a 3-D box, \(\mathbf{v}\in\mathbb{R}^{2}\) is planar velocity, \(e\) is existence, \(a\) and \(m\) are age and missed-measurement counters, and \(\mathcal{H}\) is an ordered causal state history. Every forecast-eligible active query outputs \(K=6\) modes for \(H=50\) future steps.

### B. Shared LiDAR Encoding and Dual Cooperative Paths

Following CMP, every source point cloud is first transformed into the current receiver coordinate frame and then voxelized. A shared PointPillar encoder \(E_\theta\), comprising PillarVFE, BEV scatter, backbone, shrink, and compression layers, produces aligned source features

\[
\mathbf{F}_{s}^{t}=E_\theta
\left(\mathbf{T}_{s\rightarrow r}^{t}\mathcal{P}_{s}^{t}\right),
\qquad
\mathbf{F}^{t}\in\mathbb{R}^{B\times L\times C\times H_b\times W_b}.
\tag{3}
\]

In the implemented V2V4Real configuration, \(C=256\), \(H_b=48\), \(W_b=128\), and at most \(L=2\) sources are present. These features enter two task-specific paths.

The perception path preserves CMP's SwapFusionEncoder:

\[
\mathbf{F}_{\mathrm{fuse}}^{t}=
\operatorname{FuseBEVT}\left(\{\mathbf{F}_{s}^{t}\},\mathbf{M}_{\mathrm{cav}}^{t}\right),
\tag{4}
\]

where \(\mathbf{M}_{\mathrm{cav}}\) masks absent sources. Dense classification and box-regression heads produce \(\mathbf{P}^{t}\) and \(\mathbf{R}^{t}\). The forecasting path does not add a second detector. It retains \(\{\mathbf{F}_{s}^{t}\}\) so that active actor queries can extract source-conditioned local context from the same encoded features.

### C. Fused-BEV Persistent Query Tracking

At the beginning of frame \(t\), the previous receiver pose and planar velocity propagate active reference boxes with a constant-velocity model. The dense CMP heads decode a bounded set of current measurement proposals. A one-to-one assignment with private null alternatives associates active query slots to proposals within a metric gate; unmatched proposals initialize inactive slots. This association uses only propagated current state, proposal geometry, and proposal confidence. Forecast distributions are not used for matching.

The propagated query content and reference box are decoded against \(\mathbf{F}_{\mathrm{fuse}}^{t}\) by a persistent deformable-attention query decoder,

\[
(\widetilde{\mathbf{q}}_i^t,\widehat{\mathbf{b}}_i^t,
\widehat{\mathbf{v}}_i^t,\widehat e_i^t)
=D_\phi(\mathbf{F}_{\mathrm{fuse}}^t,
\mathbf{q}_i^{t,-},\mathbf{b}_i^{t,-}).
\tag{5}
\]

The existence logit reads both the query classifier and the dense CMP heatmap sampled at the decoded center. The state update combines propagated and measured boxes through a learned, bounded measurement gain. Lifecycle logic marks queries as newborn, observed, temporarily missed, or expired. A missing query coasts with the causal current-state model for at most a fixed time-to-live; it is not propagated with a five-second predicted trajectory. Query slots correspond to physical actors in one receiver scene, not to CAV sources.

### D. Geometry-Conditioned Actor-Source Fusion

For every active query, a \(3\times3\) metric patch centered on \(\mathbf{b}_{i,xy}^{t}\) is bilinearly sampled from each aligned source BEV. Let \(\mathbf{f}_{isn}^{t}\) denote patch token \(n\) from source \(s\). To avoid making source order a semantic identity, the source token is conditioned on the actor's geometry relative to the source sensor. We encode

\[
\mathbf{g}_{is}^{t}=
\left[
\frac{\Delta x_{is}}{R_x},
\frac{\Delta y_{is}}{R_y},
\frac{\|\Delta\mathbf{p}_{is}\|_2}{R_d},
\sin\Delta\beta_{is},\cos\Delta\beta_{is},
\sin\Delta\psi_{is},\cos\Delta\psi_{is}
\right],
\tag{6}
\]

where \(\Delta\beta\) is the actor-relative source bearing and \(\Delta\psi\) is the source-to-actor heading difference. Patch offsets are also embedded. The actor query attends to the valid source--patch tokens,

\[
\mathbf{c}_{i}^{t}=
\operatorname{MHA}\left(
\operatorname{LN}(\mathbf{q}_{i}^{t}),
\left\{
W_f\mathbf{f}_{isn}^{t}+G(\mathbf{g}_{is}^{t})+P(n)
\right\}_{s,n}
\right).
\tag{7}
\]

Absent CAVs are hard-masked before the softmax. With fixed source-index embeddings disabled, permuting valid source slots and their transforms leaves \(\mathbf{c}_{i}^{t}\) unchanged up to floating-point tolerance. This path makes cooperation actor-conditioned while the dense perception path remains scene-conditioned.

### E. Ordered Causal History and Actor Interaction

CoLIF stores up to eleven online state observations for every active query. The history is transformed only as required by the receiver coordinate contract and is right-aligned with a validity mask. Each history token encodes relative position, one-step displacement, acceleration, box geometry, sine/cosine heading, relative time, and speed. A two-layer temporal Transformer produces an ordered history feature \(\mathbf{h}_{i}^{t}\). In parallel, the median of all valid pairwise velocities provides a robust velocity estimate \(\overline{\mathbf{v}}_i^t\), and the median-deviation statistic describes motion dispersion.

The current query, cooperative source context, temporal feature, motion descriptor, and normalized current box form an actor feature

\[
\mathbf{u}_{i}^{t}=
\operatorname{LN}(\mathbf{q}_{i}^{t})
+\alpha_c\operatorname{LN}(\mathbf{c}_{i}^{t})
+\alpha_h\mathbf{h}_{i}^{t}
+M(\overline{\mathbf{v}}_i^t,\sigma_{v,i}^t,a_i^t)
+B(\mathbf{b}_i^t),
\tag{8}
\]

where \(\alpha_c\) and \(\alpha_h\) are learned bounded scales. Self-attention among forecast-eligible active actors yields interaction-aware features. A newborn query is maintained immediately, but Eq. (8) is forecast-eligible only after two valid causal observations.

### F. Actor-Source Motion Proposal

Six scale factors \(\gamma_k\in\{0,0.5,0.75,1,1.25,1.5\}\) form physical motion anchors

\[
\mathbf{A}_{ikh}=\gamma_k\overline{\mathbf{v}}_i^t\,h\Delta t,
\qquad \Delta t=0.1\ \mathrm{s}.
\tag{9}
\]

Actor-conditioned mode embeddings produce \(\mathbf{z}_{ik}^{t}\). Prediction heads estimate a bounded residual, Gaussian scale and correlation, and a mode logit:

\[
\boldsymbol{\mu}_{ikh}=
\mathbf{A}_{ikh}+r_{\max}\tanh
\left(\frac{R_h(\mathbf{z}_{ik}^{t})}{r_{\max}}\right),
\tag{10}
\]

\[
(\log\boldsymbol{\sigma}_{ikh},\rho_{ikh},a_{ik})
=U(\mathbf{z}_{ik}^{t}),
\qquad \pi_{ik}=\operatorname{softmax}_k(a_{ik}).
\tag{11}
\]

The mean is relative to the current actor center. The residual heads are zero-initialized, so the untrained predictor begins from the causal physical anchors. The default deployed output is this proposal. A cross-mode layer remains as training-time deep supervision, but its refined output is not served because the frozen full-test comparison showed worse accuracy.

### G. Optional Trajectory-Source State Refinement

The optional high-capacity branch uses each proposal state as a spatial query. Its absolute receiver-frame location is

\[
\mathbf{x}_{ikh}^{t}=\mathbf{b}_{i,xy}^{t}+\boldsymbol{\mu}_{ikh}.
\tag{12}
\]

At \(\mathbf{x}_{ikh}^{t}\), the refiner samples the **current** source BEVs; it does not claim access to future measurements. A mode token plus a learned horizon embedding queries the sampled source tokens, with the same hard CAV mask as Eq. (7). A shared temporal state encoder then processes the \(H\) states of each mode. The branch predicts bounded corrections to all proposal parameters:

\[
\boldsymbol{\mu}_{ikh}^{\mathrm{ref}}
=\boldsymbol{\mu}_{ikh}+\Delta\boldsymbol{\mu}_{ikh},
\quad
\log\boldsymbol{\sigma}_{ikh}^{\mathrm{ref}}
=\log\boldsymbol{\sigma}_{ikh}+\Delta\mathbf{s}_{ikh},
\quad
a_{ik}^{\mathrm{ref}}=a_{ik}+\Delta a_{ik}.
\tag{13}
\]

Every correction head is initialized to zero. Consequently, proposal and refined outputs are exactly equal before optimization, including mean, uncertainty, correlation, and mode logits. This identity contract is essential: an earlier non-identity implementation degraded training because it replaced calibrated proposal statistics with random refined statistics.

The two outputs are deliberately exposed by configuration. `cross_mode + proposal` is the default model supported by the best aggregate results. `trajectory_source + refined` retains the full candidate-conditioned branch for controlled capacity and difficult-target experiments.

### H. Learning Objectives and Two-Stage Optimization

Training uses causal clips and separates current-state learning from prediction learning.

**Phase 1** initializes the shared encoder and FuseBEVT from the released CMP CoBEVT checkpoint and optimizes dense anchor classification/regression, query existence, current box, velocity, and lifecycle supervision. Training-only Hungarian matching associates predictions with current actors. Annotation identities never enter the runtime forward path.

**Phase 2** freezes the encoder, FuseBEVT, measurement generator, and persistent query tracker. It trains only actor-source fusion, temporal and interaction encoders, the motion proposal, and the selected refinement branch. Freezing the current-state path ensures that forecast-architecture ablations use the same actor set, boxes, coverage, and tracking metrics.

For a matched actor with valid future positions \(\mathbf{y}_{ih}\), the closest mode under masked average displacement is the positive mode \(k^*\). Each supervised prediction layer uses correlated Gaussian negative log-likelihood and mode classification:

\[
\mathcal{L}_{\mathrm{pred}}^{(l)}=
\frac{1}{|\mathcal{V}|}
\sum_{(i,h)\in\mathcal{V}}
-\log\mathcal{N}
\left(
\mathbf{y}_{ih};
\boldsymbol{\mu}_{ik^*h}^{(l)},
\boldsymbol{\Sigma}_{ik^*h}^{(l)}
\right)
+\lambda_{\mathrm{mode}}\operatorname{CE}(a_i^{(l)},k^*).
\tag{14}
\]

The Phase-2 objective sums proposal and auxiliary/refined layers. Distributed training aggregates raw numerators and valid counts across ranks before normalization, so a rank with no valid actor contributes neither fabricated samples nor an inconsistent denominator.

### I. Strict-Online Inference

Inference processes each `(scene, receiver)` timeline chronologically and resets state at scene boundaries. The model receives only runtime point clouds, poses, source masks, and timestamps. It does not read `ass_id`, ground-truth object IDs, future targets, or precomputed AB3DMOT/MTR trajectory pickles. Every active query with two valid observations produces six five-second trajectories. Evaluation matches runtime queries to annotations only after inference and reports tracking quality, forecast coverage, minADE, minFDE, miss rate, actor age, and receiver/shared/partner-support subsets.

## IV. Current Experimental Evidence

### A. Controlled Ablation Protocol

All four CoLIF variants below load the same Phase-1 epoch-1 checkpoint, use seed 20, train Phase 2 for exactly one epoch on the official V2V4Real train split, and evaluate the fixed final checkpoint on the official test split. The data-loader generator and worker seeds are explicit; previously, model-construction RNG could alter point-shuffle order between ablations. Each run contains 855 optimizer steps with two GPUs, zero AMP overflow, zero no-signal skip, and zero cross-rank parameter-checksum delta. All variants produce identical MOTA, forecast coverage, and actor counts.

| Forecast configuration | minADE at 1/3/5 s (m) | minFDE at 1/3/5 s (m) | MOTA | Coverage |
| --- | --- | --- | ---: | ---: |
| No actor-source fusion | 1.1189 / 2.0404 / 2.7437 | 1.5749 / 3.0895 / 4.3980 | 0.6333 | 0.7894 |
| Actor-source, no ordered history | 1.0504 / 1.9313 / 2.6535 | 1.4608 / 2.9931 / 4.3827 | 0.6333 | 0.7894 |
| **Actor-source + ordered history (default)** | **0.9997 / 1.8472 / 2.5870** | **1.3708 / 2.9177 / 4.3804** | **0.6333** | **0.7894** |
| Trajectory-source refined (optional) | 1.0728 / 1.9512 / 2.6714 | 1.4703 / 2.9896 / 4.3836 | 0.6333 | 0.7894 |

The default model improves every reported ADE/FDE over the no-source variant. Its strongest effects occur at 1 and 3 s: minFDE decreases by 12.96% and 5.56%, respectively. Ordered history provides a further 6.16% and 2.52% minFDE reduction at 1 and 3 s relative to actor-source fusion without temporal ordering. The 5-s endpoint differences are small, although minADE over the complete 5-s trajectory improves by 5.71% over no-source fusion and 2.50% over no ordered history.

The optional trajectory-source refiner also improves every ADE/FDE over the no-source variant, including 3-s minFDE by 3.23%, but it does not outperform the default model overall at 3 or 5 s. It yields small lower 5-s FDE values for the partner-support subset (6.4371 versus 6.4486 m) and age-2--3 subset (12.4346 versus 12.4862 m). These subset differences motivate further analysis; they are not sufficient to claim aggregate superiority.

### B. CMP Reference Boundary

The locally reproduced CMP no-aggregation checkpoint reports minADE/minFDE of 4.4001/7.5559 m at 5 s, whereas default CoLIF reports 2.5870/4.3804 m. These values are numerically lower by 41.21% and 42.03%, respectively, but the protocols are not directly equivalent. The audited CMP cache filters detections through ground-truth IoU matching, carries ground-truth identity into tracking, and evaluates a different fixed-history target population. Accordingly, the current manuscript reports this comparison as a reproduced numerical reference, not as a protocol-matched state-of-the-art claim.

### C. Supported and Unsupported Claims

The completed ablations support the following statement: retaining aligned source features for actor-conditioned forecasting improves short- and mid-horizon trajectory accuracy beyond using only the persistent actor state, and ordered causal history provides an additional gain while tracking output remains fixed. They do not support three stronger statements: that a trajectory-source refiner is always better than actor-source fusion, that one observation is sufficient for reliable long-horizon forecasting, or that predicted futures improve fused-BEV tracking. Those statements are excluded from the final method claim.

## Working References

[1] R. Xu *et al*., “V2V4Real: A Real-World Large-Scale Dataset for Vehicle-to-Vehicle Cooperative Perception,” in *Proc. IEEE/CVF CVPR*, 2023.

[2] Z. Wang *et al*., “CMP: Cooperative Motion Prediction With Multi-Agent Communication,” *IEEE Robotics and Automation Letters*, vol. 10, no. 4, pp. 3876–3883, 2025.

[3] Z. Zhou *et al*., “V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction,” in *Proc. IEEE/CVF ICCV*, 2025.

[4] S. Souig *et al*., “UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving,” in *Proc. AAAI*, 2026.

[5] Z. Zhou, J. Wang, Y.-H. Li, and Y.-K. Huang, “Query-Centric Trajectory Prediction,” in *Proc. IEEE/CVF CVPR*, 2023.

[6] Z. Zhou *et al*., “HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction,” in *Proc. IEEE/CVF CVPR*, 2022.

[7] Z. Pang *et al*., “Standing Between Past and Future: Spatio-Temporal Modeling for Multi-Camera 3D Multi-Object Tracking,” in *Proc. IEEE/CVF CVPR*, 2023.

[8] S. Papais *et al*., “ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting,” in *Proc. IEEE/CVF ICCV*, 2025.

## Author Evidence and Citation Ledger

| BibTeX key | Supported statement | Verified evidence |
| --- | --- | --- |
| `xu2023v2v4real` | V2V4Real supplies real multi-vehicle LiDAR viewpoints and cooperative perception sequences. | V2V4Real paper and local dataset; CVPR 2023. |
| `wang2025cmp` | CMP uses cooperative BEV detection, AB3DMOT histories, MTR, and prediction aggregation. | `/home/qkb/Project/Reference/Paper/TXT/2025-RAL-CMP.txt`, Introduction and Sec. IV-A/B; local code audit. |
| `zhou2025v2xpnp` | V2XPnP performs one-step spatio-temporal intermediate fusion for end-to-end cooperative perception and prediction. | `/home/qkb/Project/Reference/Paper/TXT/2025-ICCV-V2XPnP.txt`, Introduction and method overview. |
| `souig2026unimm` | UniMM-V2X argues for multi-level/task-specialized cooperative representations across perception and prediction. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2026-AAAI-UniMM-V2X.md`, Introduction and multi-level fusion sections. |
| `zhou2022hivt` | HiVT separates local context and global interaction and uses relative geometric invariance. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2022-CVPR-HiVT.md`, Secs. 1 and 3.1. |
| `zhou2023qcnet` | QCNet uses query-centric encoding and proposal-conditioned refinement. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2023-CVPR-QCNet.md`, Secs. 1, 3.3, and 4.3. |
| `pang2023pftrack` | PF-Track uses predicted future motion in persistent tracking/re-association. | CVPR 2023 paper, Secs. 3.1 and 3.3. |
| `papais2025foresight` | ForeSight maintains multiple-hypothesis forecast memory for subsequent streaming perception/forecasting. | `/home/qkb/Project/Reference/Paper/TXT/2025-ICCV-ForeSight.txt`, Secs. 1 and 3. |

Before submission, the working references must be converted to a verified BibTeX database. The current official-test monitoring policy must be disclosed: no validation split is used, test metrics are recorded after each fixed epoch, and checkpoints are not automatically selected by test score.
