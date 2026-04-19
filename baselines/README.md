# Baselines

This directory contains auxiliary baseline implementations used for comparison with `SAVA-X`.

## Included Baselines

| Baseline | Upstream repository | Upstream license |
| --- | --- | --- |
| PDVC | [ttengwang/PDVC](https://github.com/ttengwang/PDVC) | MIT |
| Exo2EgoDVC | [ut-vision/Exo2EgoDVC](https://github.com/ut-vision/Exo2EgoDVC) | MIT |
| ActionFormer | [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) | MIT |
| TriDet | [dingfengshi/TriDet](https://github.com/dingfengshi/TriDet) | MIT |

## Unified Adaptation Protocol

Baseline results reported with this repository follow a unified ego/exo imitation error detection protocol:

- paired frozen ego and exo visual features are used as input
- ego and exo features are concatenated along the channel dimension
- an imitation-error prediction head is added for the target task
- the remaining architecture and hyperparameter choices follow the corresponding upstream baseline

