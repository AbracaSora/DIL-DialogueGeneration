# Ablation And Comparison Report (Smoke Run)

## Scope

本报告基于当前最小可运行骨架的 smoke run，目标是验证“训练链路可跑通 + 指标接口可产出”，不是最终科研结论。

## Completed Matrix

- SEQ2SEQ: done (`outputs/seq2seq_baseline/metrics.json`)
- MMI-PG: done (`outputs/mmi_pg/metrics.json`)
- Li2016-RL: done (`outputs/li2016_rl/metrics.json`)
- AIRL reward: done (`outputs/airl_reward/metrics.json`)
- Hybrid(alpha=0.5): done (`outputs/hybrid_policy/metrics.json`)

## Key Numbers

- SEQ2SEQ valid loss (best): ~0.1568
- MMI-PG reward trend: -0.74 -> -10.47 (unstable, needs reward normalization)
- Li2016-RL:
  - avg_turns: 5.0
  - distinct1: 0.0004
  - distinct2: 0.0
- Dialogue metrics sample file:
  - distinct1: 0.8261
  - distinct2: 0.9444
- Trajectory alignment:
  - discriminability acc: 1.0 (越低越好，当前差距很大)
  - embedding MMD: 1.2713
  - question rate: expert=1.0, policy=0.0

## Preliminary Findings

1. 端到端入口脚本和输出路径已全部打通，可支持后续真实数据实验。
2. 当前策略生成仍严重塌缩（`...` 与低多样性），说明奖励与采样设计还需强化。
3. IRL 判别器能分开 expert/policy，但这说明“像人类”目标尚未达成。

## Main Risks

- 奖励尺度失衡导致训练不稳定。
- 简化版 backward/coherence 近似偏粗糙，可能偏离 Li2016 原始目标。
- synthetic 数据对真实对话分布覆盖不足。

## Next Concrete Upgrades

- 接入真实 OpenSubtitles 预处理和前后向独立模型。
- 在 Hybrid 阶段引入真实 policy log-prob 的 AIRL 目标。
- 增加多 seed 统计与置信区间，避免单次运行偏差。
