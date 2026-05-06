# DIL Dialogue Generation

本仓库用于复现 Li et al. 2016（Deep Reinforcement Learning for Dialogue Generation）并扩展到 IRL 轨迹对齐实验。

## 1. 环境

```bash
pip install -r requirements.txt
```

## 2. Baseline 三阶段

```bash
python scripts/train_seq2seq.py --config configs/seq2seq.yaml
python scripts/train_mmi_pg.py --config configs/mmi_pg.yaml --init_ckpt outputs/seq2seq_baseline/best.pt
python scripts/train_li2016_rl.py --config configs/li2016_rl.yaml --init_ckpt outputs/mmi_pg/last.pt
```

## 3. IRL + Hybrid

```bash
python scripts/train_irl_reward.py --config configs/irl_airl.yaml --policy_ckpt outputs/li2016_rl/last.pt
python scripts/train_policy_with_irl.py --config configs/irl_airl.yaml --policy_ckpt outputs/li2016_rl/last.pt --reward_ckpt outputs/airl_reward/reward_model.pt
```

## 4. 评估

```bash
python scripts/eval_dialogue_metrics.py --pred_file outputs/generated.txt --out_file outputs/dialogue_metrics.json
python scripts/eval_trajectory_alignment.py --expert_file outputs/expert_traj.txt --policy_file outputs/policy_traj.txt --out_file outputs/trajectory_alignment.json
```

## 5. 当前实现说明

- 已提供端到端最小可运行骨架，支持在无真实数据时使用 synthetic 数据快速 smoke test。
- 论文完整严谨复现仍需：接入真实 OpenSubtitles 数据处理、前后向独立模型、严格的策略梯度估计与人评流程。
