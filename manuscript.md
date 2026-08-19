# DualCMP: Task-Adaptive Cooperative Motion Prediction with Fused and Source-Specific BEV Representations

> **Working manuscript, 2026-08-18.** This draft follows the verified implementation and controlled ablations in `/home/qkb/Project/CoLIF`. BibTeX keys are temporary. The manuscript distinguishes the default agent-specific feature aggregation from the optional trajectory-conditioned refinement branch.

## Abstract

Predicting the future motion of surrounding agents is essential for safe autonomous driving. Vehicle-to-everything communication improves scene understanding by sharing observations from spatially separated connected and automated vehicles (CAVs). Existing cooperative predictors mainly fuse BEV features for perception or associate tracked trajectories before forecasting. These pipelines improve scene understanding and historical motion representation, respectively. Motion prediction, however, often relies on agent-specific spatial context, and the informative viewpoint may vary across agents because of sensing range and occlusion. We propose **DualCMP**, a task-adaptive cooperative motion prediction framework with fused and source-specific BEV representations. DualCMP uses the same LiDAR encoding in two ways: fused BEV features update persistent agent queries, while each query aggregates local features from the aligned CAV feature maps. A motion-anchored decoder integrates the cooperative context with motion history and agent interactions to generate multimodal trajectories. Experiments on the real-world V2V4Real dataset demonstrate that DualCMP achieves state-of-the-art performance in cooperative motion prediction.

## I. Introduction

Autonomous vehicles rely on motion prediction to anticipate surrounding traffic and plan safe maneuvers. Motion predictors commonly represent multiple plausible futures using observed agent motion, interactions, and local scene context \cite{zhou2022hivt,zhou2023qcnet}. However, occlusion, sparse LiDAR returns, and finite sensing range can make both agent histories and local interaction context incomplete. Vehicle-to-everything communication provides spatially separated observations from nearby connected and automated vehicles (CAVs), extending the scene information available to perception and forecasting \cite{xu2023v2v4real,wang2020v2vnet}. Cooperative observations can reveal target agents and surrounding traffic that are occluded or only partially observed from the ego vehicle.

The same cooperative observations support current-state estimation and motion forecasting in different ways. For detection and tracking, multiple CAVs observe the same traffic scene from different viewpoints \cite{wang2020v2vnet,wang2025cmp,zhou2025v2xpnp}. After spatial alignment, intermediate fusion combines complementary observations to compensate for occlusion and sparse observations and extend the receiver's field of view \cite{wang2020v2vnet,zhou2025v2xpnp}. The fused representation is then used for detection and tracking. Processing each CAV separately retains information from individual views, but every stream remains limited by its own sensing range and occlusion. Detections or tracks belonging to the same agent must also be associated across views \cite{ruan2024v2xgraph,wu2025copad,madjid2026latefusion}. Scene-level fusion therefore supports detecting surrounding agents and estimating their current states in a common receiver frame.

Motion forecasting uses the cooperative observations differently. Once a target agent has been established, prediction is conditioned on its motion history, neighboring agents, and local scene context \cite{zhou2022hivt,zhou2023qcnet}. The CAV that provides the clearest view of the target or surrounding traffic may differ across target agents \cite{wang2020v2vnet,wang2025cmp,ruan2024v2xgraph}. In the same scene, an agent occluded from the receiver may be clearly visible to a partner CAV, whereas another agent may be better observed by the receiver. The same variation applies to nearby agents that influence the target's future motion. If the aligned CAV maps are fused before the target state is available, an agent-centric predictor can still extract different spatial regions for different agents from the fused BEV. However, it can no longer access the features from each CAV separately or change their relative contribution to a particular forecast. At this point, the features from different CAVs have already been combined.

Existing approaches use either feature-level fusion before detection and tracking or trajectory- and prediction-level fusion after perception. Intermediate-fusion methods aggregate aligned CAV features \cite{wang2020v2vnet,wang2025cmp,zhou2025v2xpnp}, so the forecasting module receives the fused representation rather than aligned features from individual CAVs. Trajectory- and prediction-level methods retain histories, motion queries, or forecasts from individual views \cite{yu2023v2xseq,ruan2024v2xgraph,souig2026unimm,madjid2026latefusion}, but dense spatial features have already been converted into detections, tracks, motion representations, or trajectory hypotheses. As shown in Fig. 1(a) and (b), feature fusion is applied before agent-state estimation, whereas trajectory fusion begins from histories produced by separate perception streams. Feature fusion supports a common receiver-side state but removes explicit access to features from individual CAVs before forecasting, whereas later fusion preserves motion information from individual views after dense CAV observations have been converted into agent states or predictions.

![DualCMP introduction figure](figures/dualcmp_introduction_fig1_v3.png)

*Fig. 1. Cooperative information fusion for motion prediction. (a) Intermediate fusion aggregates aligned CAV BEV features before agent-state estimation and forecasting. (b) Trajectory fusion associates and fuses histories generated by separate detection and tracking pipelines before forecasting. (c) DualCMP uses fused BEV features to update persistent agent queries and aggregates aligned CAV feature maps for agent-centric motion prediction.*

We propose **DualCMP**, a cooperative motion prediction framework that retains a fused BEV together with the aligned CAV feature maps produced by a shared LiDAR encoder. FuseBEVT aggregates the aligned maps into the fused BEV, which is used to update one persistent query for each traffic agent. A single query bank avoids separate tracks for individual CAVs. The resulting query and its current box provide a common reference for retrieving forecasting context from the aligned CAV maps. For each active query, local features are combined with relative position, distance, bearing, and heading to represent the relationship between the agent and each viewpoint. By separating these roles, DualCMP maintains one receiver-side state for each agent while constructing multi-view spatial context separately for each forecast. Persistent queries may accumulate histories of different lengths and inherit noise from state estimation. The motion decoder therefore derives trajectory anchors from the observed history and uses the encoded context to predict residual trajectories around those anchors. We evaluate DualCMP on the real-world V2V4Real dataset. To isolate the effect of the forecasting representation, the controlled ablations use fixed tracking results, agent sets, and forecast coverage. Results show that aligned CAV feature maps and ordered motion history yield lower trajectory errors than the corresponding variants.

The main contributions are summarized as follows:

1. We present a receiver-side LiDAR cooperative motion prediction framework that maintains persistent agent queries from fused BEV measurements and uses the resulting online states for forecasting rather than taking precomputed tracking trajectories as input.

2. We develop agent-specific CAV feature aggregation that conditions local BEV features on the current agent query and relative CAV-agent geometry. This design allows fused state estimation and multi-view forecasting context to share one LiDAR encoder without constructing separate tracks for each CAV.

3. We design a motion-anchored trajectory decoder that combines cooperative spatial context with ordered motion history and agent interactions. Fixed-tracking ablations on V2V4Real isolate the contributions of aligned CAV features and temporal encoding while jointly reporting tracking, forecast coverage, and trajectory accuracy.

The remainder of this paper is organized as follows. Section II reviews related work. Section III defines the cooperative motion prediction problem, Section IV presents DualCMP, and Section V reports the experimental results.

## II. Related Work

### A. Cooperative Perception and Prediction

Cooperative perception methods exchange raw observations, object outputs, or intermediate features to improve receiver-side scene understanding. Intermediate fusion balances spatial context and communication cost and is the perception backbone used by CMP \cite{wang2025cmp}. CMP extends cooperative perception to motion prediction through a modular detection--tracking--prediction pipeline and a second exchange of predicted trajectories. V2XPnP studies one-step spatio-temporal fusion for end-to-end multi-agent perception and prediction \cite{zhou2025v2xpnp}, while UniMM-V2X uses perception-level and prediction-level cooperation with task-specialized representations \cite{souig2026unimm}.

Trajectory-level methods instead start from historical trajectories produced by upstream perception and tracking. PP-VIC fuses vehicle and infrastructure trajectories with CBMOT before applying a trajectory predictor \cite{yu2023v2xseq}. V2X-Graph learns cross-view associations and fuses motion and interaction features \cite{ruan2024v2xgraph}; CoPAD combines Hungarian matching with Kalman filtering for multi-source trajectory fusion \cite{wu2025copad}; and Co-MTP performs trajectory fusion over history and future domains \cite{zhang2025comtp}. ViTraj further learns vehicle-side and infrastructure-side trajectory representations \cite{you2025vitraj}. Prediction aggregation provides a later form of cooperation in CMP and in recent late-fusion work \cite{wang2025cmp,madjid2026latefusion}. DualCMP retains intermediate fusion for persistent state estimation and reuses the aligned CAV feature maps for agent-centric motion prediction.

### B. Agent-Centric Motion Forecasting

Motion predictors commonly combine temporal agent states, local scene context, and social interaction. HiVT separates local context extraction from global agent interaction and uses relative geometric representations \cite{zhou2022hivt}. QCNet uses query-centric encoding and a proposal--refinement decoder in which proposed trajectories provide adaptive anchors for context retrieval \cite{zhou2023qcnet}. DualCMP adopts neither model directly: V2V4Real provides no HD map in the current setup, and DualCMP inputs are online persistent-query states rather than ground-truth trajectory histories. We instead use their architectural principles to organize observed motion history, relative CAV geometry, agent interaction, and the optional proposal-conditioned feature sampler.

### C. Joint Tracking and Forecasting

Persistent object queries have been used to connect detection, tracking, and forecasting in single-vehicle systems. Predicted motion can also support occlusion handling or subsequent perception in methods such as PF-Track and ForeSight \cite{pang2023pftrack,papais2025foresight}. These works prevent us from claiming that future-assisted tracking is new. More importantly, our own controlled tests did not validate that mechanism in the present fused-LiDAR setting. DualCMP therefore uses persistent queries to maintain actor identity but keeps forecasting as a read-only consumer of the online tracking state.

## III. Problem Formulation

Consider a scene with \(N_t\) available CAVs at time step \(t\), and let \(r\in\{1,\ldots,N_t\}\) denote the receiver. We denote the LiDAR point cloud and pose of CAV \(j\) by \(\mathbf{L}_t^j\) and \(\boldsymbol{\xi}_t^j\), respectively. The relative transformation from CAV \(j\) to receiver \(r\) and the runtime input are

\[
\mathbf{T}_{j\rightarrow r}^t
=\left(\boldsymbol{\xi}_t^r\right)^{-1}\boldsymbol{\xi}_t^j,
\qquad
\mathcal{X}_r^t=
\left(
\left\{\mathbf{L}_t^j,\mathbf{T}_{j\rightarrow r}^t\right\}_{j=1}^{N_t},
\boldsymbol{\xi}_t^r,\tau_t^r
\right),
\tag{1}
\]

where \(\tau_t^r\) is the receiver timestamp. Given \(\mathcal{X}_r^t\) and the state from the previous time step, the task is to estimate the current states of surrounding agents and predict their future trajectories in the receiver coordinate frame. Ground-truth boxes, identities, and future trajectories are used only for training and evaluation and are not included in \(\mathcal{X}_r^t\).

## IV. Method

### A. Method Overview

DualCMP processes the receiver sequence chronologically and maintains a fixed set of persistent agent queries,

\[
\mathcal{Q}_r^t=
\left\{
\mathbf{q}_i^t,\mathbf{b}_i^t,\mathbf{v}_i^t,
e_i^t,\eta_i^t,d_i^t,\mathcal{H}_i^t
\right\}_{i=1}^{Q},
\tag{2}
\]

where \(\mathbf{q}_i^t\) is the query feature, \(\mathbf{b}_i^t\) and \(\mathbf{v}_i^t\) are the current 3-D box and planar velocity, \(e_i^t\) is the existence score, \(\eta_i^t\) and \(d_i^t\) record the query age and consecutive misses, and \(\mathcal{H}_i^t\) contains the available state history. The recurrent state also stores the receiver pose and timestamp used to transform the queries between consecutive receiver frames. Each active query represents one traffic agent rather than one CAV. Once an agent has two valid observations, the model predicts a multimodal distribution over its future positions.

DualCMP follows a dual-representation design. For each receiver frame, the LiDAR encoder first computes aligned CAV BEV feature maps. FuseBEVT aggregates these features into a fused BEV representation, which is used to update the current state and history of the persistent agent queries. The motion predictor also retrieves local features for each agent from the aligned CAV feature maps. The resulting cooperative context is combined with the query history and inter-agent interactions to predict multimodal future trajectories. The persistent query state connects tracking and forecasting, while forecast outputs do not update the tracking state.

### B. Fused and Aligned CAV BEV Encoding

Following CMP \cite{wang2025cmp}, each CAV point cloud is transformed to the receiver frame before voxelization. A weight-shared PointPillar encoder \(E_\theta\) maps the receiver-aligned point clouds to aligned CAV BEV features, which are regrouped with an availability mask and fused by FuseBEVT:

\[
\mathbf{F}_j^t=E_\theta
\left(\mathbf{T}_{j\rightarrow r}^t\mathbf{L}_t^j\right),
\qquad
\mathbf{F}_{\mathrm{fuse}}^t=
\operatorname{FuseBEVT}
\left(\{\mathbf{F}_j^t\}_{j=1}^{N_t},\mathbf{m}^t\right).
\tag{3}
\]

The mask \(\mathbf{m}^t\) marks valid CAV features when samples are padded to a common batch shape. The model retains both outputs of (3). The fused feature \(\mathbf{F}_{\mathrm{fuse}}^t\) provides a receiver-centric scene representation for current-state estimation. The aligned features \(\{\mathbf{F}_j^t\}\) preserve the CAV dimension and are reused by the motion predictor after the agent queries have been updated. Both paths use the same PointPillar encoder; DualCMP does not introduce a second perception network. Dense classification and box-regression heads attached to \(\mathbf{F}_{\mathrm{fuse}}^t\) also provide current-frame measurement proposals for query tracking.

### C. Persistent Query-Based Agent Tracking

Before processing a new frame, the previous boxes are transformed to the current receiver frame and propagated with their estimated planar velocities. The dense heads decode a bounded set of current measurement proposals. Active queries are matched to these proposals by a distance-gated one-to-one assignment, and unmatched proposals initialize inactive query slots.

The query features and reference boxes are updated from the fused BEV by a deformable-attention decoder following the persistent-query design used in online tracking \cite{gu2023vip3d}:

\[
(\widetilde{\mathbf{q}}_i^t,
\widehat{\mathbf{b}}_i^t,
\widehat{\mathbf{v}}_i^t,
\widehat e_i^t)
=D_\phi
\left(
\mathbf{F}_{\mathrm{fuse}}^t,
\mathbf{q}_i^{t,-},
\mathbf{b}_i^{t,-}
\right),
\tag{4}
\]

where the superscript \((- )\) denotes the propagated state. The existence score combines the query prediction with the dense classification response sampled at the decoded center. A learned measurement update combines the propagated and decoded states, after which the lifecycle module creates new queries, retains temporarily missed queries, and releases expired slots. Runtime identities are assigned internally and have no correspondence to annotation IDs. Motion forecasts are not used for proposal matching, state propagation, or lifecycle decisions in the default model.

### D. Agent-Specific CAV Feature Aggregation

The fused BEV supplies a unified state estimate, while motion prediction additionally uses the aligned CAV features. For each active agent \(i\), DualCMP samples a local feature patch around \(\mathbf{b}_{i,xy}^t\) from every available \(\mathbf{F}_j^t\). Let \(\mathbf{f}_{ijn}^t\) denote the \(n\)-th sampled feature from CAV \(j\). The source token combines this feature with its patch position and the relative geometry between the agent and the CAV. Agent-source attention is then computed as

\[
\begin{aligned}
\mathbf{z}_{ijn}^t
&=W_f\mathbf{f}_{ijn}^t+G(\mathbf{g}_{ij}^t)+P(\mathbf{p}_n),\\
\mathbf{c}_i^t
&=\operatorname{MHA}
\left(
\operatorname{LN}(\mathbf{q}_i^t),
\{\mathbf{z}_{ijn}^t\}_{j,n},
\{\mathbf{z}_{ijn}^t\}_{j,n}
\right),
\end{aligned}
\tag{5}
\]

where \(\mathbf{g}_{ij}^t\) contains relative position, distance, bearing, and heading, and \(\mathbf{p}_n\) is the patch offset. Features from unavailable CAVs are masked before attention. The model does not assign a learned identity embedding to a fixed CAV slot; relative CAV geometry is used as positional information. The resulting context \(\mathbf{c}_i^t\) is used only by the motion predictor, so agent tracking continues to depend on the fused-BEV path.

### E. Motion Context Encoding

For every active query, DualCMP stores an ordered sequence of estimated agent states together with a validity mask. A temporal Transformer encodes relative position, motion increments, box state, heading, and relative time. The model also computes a robust velocity \(\overline{\mathbf{v}}_i^t\) from the median of all valid pairwise velocities and uses their dispersion to describe motion variation. Let \(n_i^t\) denote the number of valid history states. The query feature, cooperative context, temporal feature, motion descriptor, and current box are combined as

\[
\mathbf{u}_i^t=
\operatorname{LN}(\mathbf{q}_i^t)
+\alpha_c\operatorname{LN}(\mathbf{c}_i^t)
+\alpha_h\mathbf{h}_i^t
+M(\overline{\mathbf{v}}_i^t,\sigma_{v,i}^t,n_i^t)
+B(\mathbf{b}_i^t),
\tag{6}
\]

where \(\mathbf{h}_i^t\) is the history feature and \(\alpha_c\) and \(\alpha_h\) are learned scales. Self-attention among forecast-eligible agents updates \(\mathbf{u}_i^t\) with inter-agent context.

### F. Motion-Anchored Trajectory Decoding

The decoder uses speed-scaled motion anchors to provide a stable initialization for multimodal prediction. Given scale \(\gamma_k\) for mode \(k\), the anchor at future step \(h\) is

\[
\mathbf{A}_{ikh}=\gamma_k\overline{\mathbf{v}}_i^t h\Delta t.
\tag{7}
\]

Combining \(\mathbf{u}_i^t\) with a learned embedding for mode \(k\) gives the mode feature \(\mathbf{z}_{ik}^t\). Prediction heads estimate a bounded residual around each anchor, a mode probability, and the parameters of a correlated bivariate Gaussian. Let \(\mathcal{V}_i\) be the valid future steps. The resulting distribution is

\[
\begin{aligned}
\boldsymbol{\mu}_{ikh}
&=\mathbf{A}_{ikh}+r_{\max}\tanh
\left(\frac{R_h(\mathbf{z}_{ik}^t)}{r_{\max}}\right),\\
\ell_{ik}
&=U_\pi(\mathbf{z}_{ik}^t),
\qquad
\boldsymbol{\Sigma}_{ikh}
&=U_{\Sigma,h}(\mathbf{z}_{ik}^t),
\qquad
\boldsymbol{\pi}_i=\operatorname{softmax}(\boldsymbol{\ell}_i),\\
p(\mathbf{Y}_i\mid\mathbf{u}_i^t)
&=\sum_{k=1}^{K}\pi_{ik}
\prod_{h\in\mathcal{V}_i}
\mathcal{N}
\left(\mathbf{y}_{ih};
\boldsymbol{\mu}_{ikh},\boldsymbol{\Sigma}_{ikh}\right).
\end{aligned}
\tag{8}
\]

The residual heads are initialized to zero, so the initial mean trajectories follow the history-derived anchors. The default model returns this motion proposal at inference and uses a cross-mode layer only for auxiliary supervision. We also implement an optional candidate-conditioned refinement variant: it samples the aligned CAV BEV features at the proposed trajectory locations and predicts residual corrections to the proposal. The optional variant is evaluated separately and does not alter the default prediction path.

### G. Training Objectives

Training uses chronological clips and separates current-state estimation from motion prediction. In Phase 1, the PointPillar encoder, FuseBEVT, dense heads, and persistent query tracker are optimized with

\[
\mathcal{L}_{\mathrm{state}}=
\lambda_{\mathrm{det}}(\mathcal{L}_{\mathrm{cls}}+2\mathcal{L}_{\mathrm{reg}})
+\lambda_{\mathrm{exist}}\mathcal{L}_{\mathrm{exist}}
+\lambda_{\mathrm{box}}\mathcal{L}_{\mathrm{box}}
+\lambda_{\mathrm{vel}}\mathcal{L}_{\mathrm{vel}}
+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}.
\tag{9}
\]

Training-only sequential matching assigns annotations to query slots. A matched query keeps its annotation identity while it remains active, and Hungarian matching is applied only to unmatched targets and available query slots. Annotation identities supervise the loss but are not passed to the model state.

In Phase 2, the encoder, fused-BEV measurement path, and query tracker are frozen. The agent-source attention, history encoder, interaction module, and motion decoder are trained with correlated Gaussian negative log-likelihood and mode classification. For each supervised prediction layer \(l\), the positive mode \(k_i^{*(l)}\) is selected by the distance between its predicted endpoint and the final valid target position. Let \(L\) be the number of supervised prediction layers. The prediction objective is

\[
\mathcal{L}_{\mathrm{pred}}=
\frac{1}{L}\sum_{l=1}^{L}
\left[
\lambda_{\mathrm{gmm}}
\sum_{i}\sum_{h\in\mathcal{V}_i}
-\log\mathcal{N}
\left(
\mathbf{y}_{ih};
\boldsymbol{\mu}_{ik_i^{*(l)}h}^{(l)},
\boldsymbol{\Sigma}_{ik_i^{*(l)}h}^{(l)}
\right)
+\lambda_{\mathrm{mode}}
\sum_i\operatorname{CE}
\left(\boldsymbol{\ell}_i^{(l)},k_i^{*(l)}\right)
\right].
\tag{10}
\]

Freezing the current-state path gives all forecast variants the same agent queries, boxes, and forecast coverage. At inference, DualCMP processes each receiver sequence in temporal order, resets the query state at scene boundaries, and produces trajectories without ground-truth annotations or precomputed tracking trajectories.

## V. Current Experimental Evidence

### A. Controlled Ablation Protocol

All four DualCMP variants below load the same Phase-1 epoch-1 checkpoint, use seed 20, train Phase 2 for exactly one epoch on the official V2V4Real train split, and evaluate the fixed final checkpoint on the official test split. The data-loader generator and worker seeds are explicit; previously, model-construction RNG could alter point-shuffle order between ablations. Each run contains 855 optimizer steps with two GPUs, zero AMP overflow, zero no-signal skip, and zero cross-rank parameter-checksum delta. All variants produce identical MOTA, forecast coverage, and actor counts.

| Forecast configuration | minADE at 1/3/5 s (m) | minFDE at 1/3/5 s (m) | MOTA | Coverage |
| --- | --- | --- | ---: | ---: |
| No actor-source fusion | 1.1189 / 2.0404 / 2.7437 | 1.5749 / 3.0895 / 4.3980 | 0.6333 | 0.7894 |
| Actor-source, no ordered history | 1.0504 / 1.9313 / 2.6535 | 1.4608 / 2.9931 / 4.3827 | 0.6333 | 0.7894 |
| **Actor-source + ordered history (default)** | **0.9997 / 1.8472 / 2.5870** | **1.3708 / 2.9177 / 4.3804** | **0.6333** | **0.7894** |
| Trajectory-source refined (optional) | 1.0728 / 1.9512 / 2.6714 | 1.4703 / 2.9896 / 4.3836 | 0.6333 | 0.7894 |

The default model improves every reported ADE/FDE over the no-source variant. Its strongest effects occur at 1 and 3 s: minFDE decreases by 12.96% and 5.56%, respectively. Ordered history provides a further 6.16% and 2.52% minFDE reduction at 1 and 3 s relative to actor-source fusion without temporal ordering. The 5-s endpoint differences are small, although minADE over the complete 5-s trajectory improves by 5.71% over no-source fusion and 2.50% over no ordered history.

The optional trajectory-source refiner also improves every ADE/FDE over the no-source variant, including 3-s minFDE by 3.23%, but it does not outperform the default model overall at 3 or 5 s. It yields small lower 5-s FDE values for the partner-support subset (6.4371 versus 6.4486 m) and age-2--3 subset (12.4346 versus 12.4862 m). These subset differences motivate further analysis; they are not sufficient to claim aggregate superiority.

### B. CMP Reference Boundary

The locally reproduced CMP no-aggregation checkpoint reports minADE/minFDE of 4.4001/7.5559 m at 5 s, whereas default DualCMP reports 2.5870/4.3804 m. These values are numerically lower by 41.21% and 42.03%, respectively, but the protocols are not directly equivalent. The audited CMP cache filters detections through ground-truth IoU matching, carries ground-truth identity into tracking, and evaluates a different fixed-history target population. Accordingly, the current manuscript reports this comparison as a reproduced numerical reference, not as a protocol-matched state-of-the-art claim.

### C. Supported and Unsupported Claims

The completed ablations support the following statement: retaining aligned CAV features for agent-specific forecasting improves short- and mid-horizon trajectory accuracy beyond using only the persistent agent state, and ordered motion history provides an additional gain while tracking output remains fixed. They do not support three stronger statements: that a trajectory-conditioned refiner is always better than agent-specific feature aggregation, that one observation is sufficient for reliable long-horizon forecasting, or that predicted futures improve fused-BEV tracking. Those statements are excluded from the final method claim.

## Working References

[1] R. Xu *et al*., “V2V4Real: A Real-World Large-Scale Dataset for Vehicle-to-Vehicle Cooperative Perception,” in *Proc. IEEE/CVF CVPR*, 2023.

[2] Z. Wang *et al*., “CMP: Cooperative Motion Prediction With Multi-Agent Communication,” *IEEE Robotics and Automation Letters*, vol. 10, no. 4, pp. 3876–3883, 2025.

[3] Z. Zhou *et al*., “V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction,” in *Proc. IEEE/CVF ICCV*, 2025.

[4] S. Souig *et al*., “UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving,” in *Proc. AAAI*, 2026.

[5] Z. Zhou, J. Wang, Y.-H. Li, and Y.-K. Huang, “Query-Centric Trajectory Prediction,” in *Proc. IEEE/CVF CVPR*, 2023.

[6] Z. Zhou *et al*., “HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction,” in *Proc. IEEE/CVF CVPR*, 2022.

[7] Z. Pang *et al*., “Standing Between Past and Future: Spatio-Temporal Modeling for Multi-Camera 3D Multi-Object Tracking,” in *Proc. IEEE/CVF CVPR*, 2023.

[8] S. Papais *et al*., “ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting,” in *Proc. IEEE/CVF ICCV*, 2025.

[9] T.-H. Wang *et al*., “V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Prediction,” in *Proc. ECCV*, 2020.

[10] J. Gu *et al*., “ViP3D: End-to-End Visual Trajectory Prediction via 3D Agent Queries,” in *Proc. IEEE/CVF CVPR*, 2023.

[11] J. Zhong *et al*., “CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception,” in *Proc. IEEE/CVF ICCV*, 2025.

[12] H. Ruan, H. Yu, W. Yang, S. Fan, and Z. Nie, “Learning Cooperative Trajectory Representations for Motion Forecasting,” in *Advances in Neural Information Processing Systems*, 2024.

[13] S. You, L. Weng, and F. Gao, “ViTraj: Learning Dual-Side Representations for Vehicle-Infrastructure Cooperative Trajectory Prediction,” in *Proc. ACM Multimedia*, 2025.

[14] N. A. Madjid *et al*., “Collaborative Trajectory Prediction via Late Fusion,” arXiv:2604.22973, 2026.

[15] H. Yu *et al*., “V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting,” in *Proc. IEEE/CVF CVPR*, 2023.

[16] K. Wu, J. Qiao, and Y. Zhang, “CoPAD: Multi-Source Trajectory Fusion and Cooperative Trajectory Prediction With Anchor-Oriented Decoder in V2X Scenarios,” in *Proc. IEEE/RSJ IROS*, 2025.

[17] X. Zhang *et al*., “Co-MTP: A Cooperative Trajectory Prediction Framework With Multi-Temporal Fusion for Autonomous Driving,” in *Proc. IEEE ICRA*, 2025.

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
| `wang2020v2vnet` | V2VNet uses spatially aware cross-vehicle aggregation for joint perception and motion forecasting. | `/home/qkb/Project/Reference/Paper/TXT/2020-ECCV-V2VNet.txt`, Abstract and Secs. 1 and 3. |
| `gu2023vip3d` | ViP3D propagates 3-D agent queries through detection, tracking, and trajectory prediction. | `/home/qkb/Project/Reference/Paper/TXT/2023-CVPR-ViP3D.txt`, Abstract and Secs. 1 and 3. |
| `zhong2025cooptrack` | CoopTrack uses persistent instance queries and learnable cross-agent association for cooperative sequential perception. | `/home/qkb/Project/Reference/Paper/TXT/2025-CoopTrack.txt`, Abstract and Secs. 1 and 3. |
| `ruan2024v2xgraph` | V2X-Graph constructs forecasting-oriented motion and interaction representations from multi-view cooperative trajectories. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2024-NeurIPS-V2X-Graph.md`, Introduction and Secs. 4.1--4.3. |
| `you2025vitraj` | ViTraj retains vehicle-side and infrastructure-side trajectory features and performs feature-side selection and cooperative interaction. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2025-ACM-MM-ViTraj.md`, Introduction and Secs. 4.1--4.2. |
| `madjid2026latefusion` | The late-fusion framework retains observer-specific forecasts and combines them after local prediction with temporal alignment and association. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2026-arxiv-Collaborative-Trajectory-Prediction.md`, Introduction and Secs. 3.1--3.2. |
| `yu2023v2xseq` | The PP-VIC baseline fuses vehicle and infrastructure trajectories with CBMOT before TNT or HiVT forecasting. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2023-CVPR-V2X-Seq.md`, Sec. 6.2.1. |
| `wu2025copad` | CoPAD matches multi-source trajectories with the Hungarian algorithm and fuses matched trajectories with Kalman filtering before prediction. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2025-IROS-CoPAD.md`, Secs. III-A--III-B. |
| `zhang2025comtp` | Co-MTP preprocesses vehicle and infrastructure histories and performs multi-temporal trajectory fusion in a heterogeneous graph. | `/home/qkb/Project/Reference/Paper/Markdown/ALL_MARKDOWN/2025-ICRA-Co-MTP.md`, Secs. III-B--III-D. |

Before submission, the working references must be converted to a verified BibTeX database. The current official-test monitoring policy must be disclosed: no validation split is used, test metrics are recorded after each fixed epoch, and checkpoints are not automatically selected by test score.
