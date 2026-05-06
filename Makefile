PY=python

.PHONY: train-seq2seq train-mmi train-li-rl train-irl train-hybrid eval-dialogue eval-traj

train-seq2seq:
	$(PY) scripts/train_seq2seq.py --config configs/seq2seq.yaml

train-mmi:
	$(PY) scripts/train_mmi_pg.py --config configs/mmi_pg.yaml --init_ckpt outputs/seq2seq_baseline/best.pt

train-li-rl:
	$(PY) scripts/train_li2016_rl.py --config configs/li2016_rl.yaml --init_ckpt outputs/mmi_pg/last.pt

train-irl:
	$(PY) scripts/train_irl_reward.py --config configs/irl_airl.yaml --policy_ckpt outputs/li2016_rl/last.pt

train-hybrid:
	$(PY) scripts/train_policy_with_irl.py --config configs/irl_airl.yaml --policy_ckpt outputs/li2016_rl/last.pt --reward_ckpt outputs/airl_reward/reward_model.pt

eval-dialogue:
	$(PY) scripts/eval_dialogue_metrics.py --pred_file outputs/generated.txt --out_file outputs/dialogue_metrics.json

eval-traj:
	$(PY) scripts/eval_trajectory_alignment.py --expert_file outputs/expert_traj.txt --policy_file outputs/policy_traj.txt --out_file outputs/trajectory_alignment.json
