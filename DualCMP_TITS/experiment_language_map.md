# Experiments Language Map

本文件用于约束 DualCMP 论文实验部分的英文标题、句式和术语。标题统计来自本地 20 篇同领域论文的 Experiments/Evaluation 正文，共检查 90 个下级标题。句式来自 CMP、ObjectFusion、IS-FUSION、LoGoNet、MTR 和 V2V4Real 的实验部分。论文正文不得用本文件中的功能归类替代实际原文标题。

## 1. Section Titles

### Accepted structure

```latex
\section{Experiments}

\subsection{Experimental Setup}

\subsection{Main Results}

\subsection{Ablation Study}

\subsection{Qualitative Results}
```

### Exact title counts

| Original title | Count |
| --- | ---: |
| `Experimental Setup` | 6 |
| `Implementation Details` | 5 |
| `Ablation Study` | 5 |
| `Ablation studies` | 3 |
| `Dataset` | 2 |
| `Datasets` | 2 |
| `Experiment Setup` | 2 |
| `Results` | 2 |
| `Main Results` | 2 |
| `Quantitative Evaluation` | 2 |
| `Quantitative evaluation` | 2 |
| `Qualitative Results` | 2 |
| `Qualitative evaluation` | 2 |

### Terms used inside Experimental Setup

```text
Dataset
Evaluation Metrics
Implementation Details
```

### Titles not used

```text
Overall Performance
Effect of Source-Specific BEV Features
Source-Aware Analysis
Further Analysis
Efficiency and Limitations
```

## 2. Experimental Setup

### Dataset

Common sentence patterns:

```text
We conduct experiments on the ... dataset.
We use the ... dataset to evaluate our method.
Following the official split, ...
The training and test sets contain ...
We use ... seconds of historical observations and predict ... seconds into the future.
```

Terms:

```text
dataset
training set
validation set
test set
official split
traffic scenarios
historical observations
prediction horizon
receiver
LiDAR observations
```

### Evaluation Metrics

Common sentence patterns:

```text
We use the standard evaluation metrics, including ...
We report ... as the evaluation metrics.
Lower values indicate better performance.
All models are evaluated on the same ...
```

Terms:

```text
evaluation metrics
minADE
minFDE
miss rate
MOTA
MOTP
precision
recall
forecast coverage
endpoint threshold
```

### Implementation Details

Common sentence patterns:

```text
We train the model using the AdamW optimizer.
The learning rate is set to ...
The batch size is set to ...
We train the model for ... epochs.
All models use the same experimental setting.
All experiments are conducted on ... GPUs.
```

Terms:

```text
implementation details
AdamW optimizer
learning rate
weight decay
batch size
training epochs
mixed precision
checkpoint
random seed
```

## 3. Main Results

Common sentence patterns:

```text
Table ... shows the main results.
Table ... shows the motion prediction results.
Compared with ..., our method reduces ... from ... to ... .
Our method achieves ... on the test set.
The results show that ...
```

Terms:

```text
main results
motion prediction results
performance comparison
baseline
test set
prediction performance
compared with
achieves
reduces
improves
```

Writing rules:

- Report the table, metric, value, comparator, and prediction horizon in the same paragraph.
- Use `shows`, `achieves`, `reduces`, and `improves` for measured results.
- Do not attribute a result to a component unless the corresponding ablation result is available.
- Do not insert unfinished MTR or CMP results into the main table.

## 4. Ablation Study

Common sentence patterns:

```text
We conduct ablation studies to evaluate the effectiveness of ...
We study the effect of ...
All models use the same ...
Table ... shows the ablation results.
Compared with ..., ...
```

Terms:

```text
ablation study
effectiveness
component
feature aggregation
fused BEV
per-CAV BEV
mean aggregation
attention weights
same tracker
same training schedule
same evaluation agent set
```

Required comparison:

```text
Fused BEV + agent patch
Per-CAV BEV + mean
Per-CAV BEV + query and geometry
```

Writing rules:

- State the factor changed by the ablation.
- State the components and evaluation conditions shared by all models.
- Use attention weights only to describe feature aggregation.
- Do not use attention weights as motion prediction performance results.
- Do not claim that per-CAV BEV improves forecasting until the fused-BEV and mean results are available.

## 5. Qualitative Results

Common sentence patterns:

```text
We provide qualitative results in Fig. ...
Fig. ... shows the qualitative results.
The first column shows ...
The second and third columns show ...
The last column shows ...
```

Terms:

```text
qualitative results
traffic scenario
observed trajectory
ground-truth trajectory
predicted trajectories
trajectory probability
feature map
attention weight
tracked agent
```

Writing rules:

- Define every column, color, line style, marker, and opacity used in the figure.
- State the example selection rule in the text.
- Use the figure to show model outputs and attention weights, not to claim an improvement in minADE or minFDE.

## 6. Source Sections

- CMP: `/home/qkb/Project/Reference/Thesis/Markdown/2025-RAL-Cmp-2.md`, Section V.
- ObjectFusion: `/home/qkb/Project/Reference/Thesis/Markdown/2023-ICCV-ObjectFusion.md`, Section 4.
- IS-FUSION: `/home/qkb/Project/Reference/Thesis/Markdown/2024-CVPR-Is-fusion.md`, Section 4.
- LoGoNet: `/home/qkb/Project/Reference/Thesis/Markdown/2023-CVPR-LoGoNet.md`, Section 4.
- MTR: `/home/qkb/Project/Reference/Thesis/Markdown/2022-NeurIPS-Motion-transformer-with-global-intention-localization-and-local-movement-refinem.md`, Section 4.
- V2V4Real: `/home/qkb/Project/Reference/Thesis/Markdown/2023-CVPR-V2V4Real.md`, Section 5 and appendix.

## 7. Evidence Boundary

- Verified DualCMP seed-20 results may be reported as measured results.
- Fused-BEV, mean aggregation, MTR, CMP, additional seeds, latency, and memory results must not be reported before evaluation is complete.
- Attention weights show changes in feature aggregation with observation support and relative distance. They do not establish an improvement in prediction accuracy.
