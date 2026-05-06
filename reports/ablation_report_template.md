# Ablation And Comparison Report

## Experiment Matrix

- SEQ2SEQ
- MMI-PG
- Li2016-RL
- IRL-only
- Hybrid(alpha=0.2/0.5/0.8)

## Baseline Metrics

- Avg simulated turns
- Distinct-1
- Distinct-2

## Trajectory Alignment Metrics

- Trajectory discriminability accuracy (lower is better)
- Embedding MMD (lower is better)
- Question rate gap

## Findings

1. Baseline trend check:
2. IRL improvement over Li2016-RL:
3. Failure cases:

## Risks

- Reward hacking / mode collapse
- Weak alignment between proxy metric and human preference

## Next Actions

- Improve expert trajectory quality
- Add multi-seed confidence intervals
- Add human pairwise multi-turn evaluation
