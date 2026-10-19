# Reinforcement Learning Assignment 1 - Yujun Sun

Source of requirements: four-page `mlds_assignment1.pdf`. Actual execution host: Apple M4 Pro, 24 GB, macOS 15.6.1, native arm64 Python 3.12.14. All compute is local. See `CODEX_TASK.md`, `USER_CONTEXT.md`, `REQUIREMENTS.md` and `PROGRESS.md` for scope and current status.

Each of the four original assignment models trained for 4,096 complete episodes and was frozen before its 500 held-out evaluation episodes. Actual results (raw undiscounted reward, sample SD with `ddof=1`):

| Model | Evaluation episodes | Mean | Sample SD |
|---|---:|---:|---:|
| CartPole basic | 500 | 143.678 | 125.834 |
| CartPole baseline | 500 | 423.592 | 83.414 |
| Pong basic | 500 | -20.534 | 0.706 |
| Pong baseline | 500 | -19.806 | 1.046 |

Both Pong policies remain losing policies. A separate paired 32-episode diagnostic found a 46.16% reduction in Pong gradient covariance trace with the saved baseline, while final reward SD increased. Gradient-estimator variance and reward variability are different quantities. These are single-training-seed comparisons; full evidence and limitations are in the report.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python src/rl.py inspect
```

The tested ALE package includes the Pong ROM; no alternate version or RAM input is used. On this Codex desktop session, its process sandbox hides MPS even though native MPS works; the recorded device test and benchmark were run outside that sandbox on the same Mac. Normal macOS Terminal can access MPS. The experiment scripts never use a remote host.

## Shuttle solution and quantitative validation

```sh
.venv/bin/python src/shuttle.py all
.venv/bin/python src/shuttle.py plot
```

`docs/shuttle_method.md` defines the event order and the assumptions. Value functions represent positive costs. Raw values and independent checks are in `data/shuttle/`.

## REINFORCE correctness and device measurement

```sh
.venv/bin/python src/validate_rl.py
.venv/bin/python src/rl.py benchmark --seconds 300 --batch 8
```

The benchmark runs CPU and MPS in separate fresh processes, each for at least 300 seconds including complete rollouts and policy updates, and chooses the device with greater measured environment steps/second. It can overshoot slightly to finish the current full batch. `data/benchmark/decision.json` records the choice and `docs/experiment_budget.md` explains the formal budget.

## Formal training and complete evaluation

The complete pipeline resumes unfinished runs and skips frozen completed runs:

```sh
caffeinate -i .venv/bin/python -u scripts/run_experiments.py
```

The measured formal budget is 4,096 complete training episodes per model (512 updates of 8). Pong uses CPU (4,203.09 steps/s versus 3,217.39 on MPS); CartPole LR 0.01, Pong LR 0.001. The preregistered plan and rationale are in `data/experiment_plan.json` and `docs/experiment_budget.md`. Individual training/resume commands:

```sh
.venv/bin/python src/rl.py train --run runs/cartpole_basic --game cartpole --episodes 4096 --seed 101 --lr 0.01 --batch 8 --device cpu
.venv/bin/python src/rl.py train --run runs/cartpole_baseline --game cartpole --episodes 4096 --seed 101 --lr 0.01 --batch 8 --device cpu --baseline
.venv/bin/python src/rl.py train --run runs/pong_basic --game pong --episodes 4096 --seed 202 --lr 0.001 --batch 8 --device cpu
.venv/bin/python src/rl.py train --run runs/pong_baseline --game pong --episodes 4096 --seed 202 --lr 0.001 --batch 8 --device cpu --baseline
```

Evaluation uses the frozen final checkpoint and 500 independent complete episodes per model:

```sh
.venv/bin/python src/rl.py evaluate --run runs/cartpole_basic --episodes 500 --batch 8 --device cpu
.venv/bin/python src/rl.py evaluate --run runs/cartpole_baseline --episodes 500 --batch 8 --device cpu
.venv/bin/python src/rl.py evaluate --run runs/pong_basic --episodes 500 --batch 8 --device cpu
.venv/bin/python src/rl.py evaluate --run runs/pong_baseline --episodes 500 --batch 8 --device cpu
```

Use the device recorded in the configuration if the benchmark chooses MPS. `evaluation.csv` stores all complete episodes, original total rewards, lengths, termination flags, seeds and checkpoint SHA-256. Reported standard deviations use sample `ddof=1`; population values are also saved. Both methods sample the action distribution. No final evaluation reward is used for checkpoint selection.

Training resumes automatically from `checkpoint.pt` at a complete policy-update boundary. An interrupted uncommitted batch is regenerated using the same per-episode seeds and unchanged checkpoint policy. Already committed train/evaluation episodes are not repeated. Do not start duplicate processes: inspect `logs/experiment_runner.json`, `runs/*/process.json`, `logs/` and `PROGRESS.md` first.

Training `sampling_policy_update` identifies the policy version that generated a trajectory; `update` identifies the subsequent optimizer update consuming that trajectory. Training `checkpoint` is a logical sampling-version tag, not the name of a retained file. Only the latest resumable checkpoint and the frozen final checkpoint are retained. Evaluation rows identify the exact frozen file by SHA-256.

## Figures, report and package

```sh
.venv/bin/python src/plot_results.py
.venv/bin/python -m src.gradient_diagnostic --run runs/cartpole_baseline --run runs/pong_baseline --episodes 32 --device cpu
.venv/bin/python src/build_report.py
.venv/bin/python scripts/check_delivery.py
.venv/bin/python scripts/package.py
```

Replotting and report generation use saved data; no retraining is required. Report source is `src/build_report.py` plus generated editable `docs/report.md`. Chinese learning notes are `docs/LEARNING_ZH.md`. Final report is `output/pdf/assignment1_report.pdf`. Final archive is `output/assignment1_Yujun_Sun.zip`; it excludes environments, caches, redundant checkpoint copies and smoke-run model files.

## Repository and submission

Delivered repository: [SAMSUNYUJUN/mlds-rl-assignment1](https://github.com/SAMSUNYUJUN/mlds-rl-assignment1), branch `main`. The user explicitly authorized the push, and the source commit was verified against the GitHub API. Identity comes from the actual origin and authenticated `gh api user`, not `git user.name`. Initial sandboxed `gh auth status` misleadingly reported an invalid token; outside the network sandbox authentication succeeded. Authorization and delivery evidence are in `data/repository.json` and `ACCEPTANCE.md`. The user performs the final Canvas submission.

## CartPole PPO extension

The `codex/cartpole-ppo-comparison` branch adds a self-written PPO-Clip experiment
alongside the frozen assignment REINFORCE runs. It uses the same actor architecture,
raw observations, gamma 0.95 and CPU. Its fixed 1,384,029 environment-step budget
matches the original no-baseline REINFORCE run. PPO additionally uses a state-value
critic, GAE, normalized advantages and multiple minibatch epochs; this is an
algorithm/configuration comparison, not an isolated clipping ablation. The original
submission PDF remains unchanged. See [comparison](docs/PPO_COMPARISON.md) and
the preregistered configuration in `data/ppo_plan.json`.

Fresh paired evaluation (500 full sampled episodes each, seeds 84000000–84000499, sample SD):

| Method | Mean reward | Sample SD |
|---|---:|---:|
| REINFORCE, no baseline |142.464|120.310|
| REINFORCE + EMA baseline |419.182|83.477|
| PPO + state critic / GAE |500.000|0.000|

PPO reached the 500-step cap in all 500 fresh evaluation episodes. This is one
training seed; PPO also uses more optimization work and different estimator
conventions. See the comparison document for the full limits and raw evidence.

```sh
# Quantitative estimator, clipping, critic and full checkpoint/collector checks
.venv/bin/python -m src.validate_ppo
# Train or resume; final-only freeze at the fixed interaction budget
caffeinate -i .venv/bin/python -u -m src.ppo train --run runs/cartpole_ppo
# Historical seed protocol, 500 complete episodes
.venv/bin/python -m src.ppo evaluate --run runs/cartpole_ppo --episodes 500
# New paired seed protocol, 500 complete episodes for each frozen policy
for run in cartpole_basic cartpole_baseline cartpole_ppo; do
  .venv/bin/python -m src.ppo evaluate --run runs/$run \
    --output runs/ppo_comparison_eval/$run --episodes 500 --eval-seed-base 84000000
done
# Regenerate comparison plots/statistics/discussion from saved data only
.venv/bin/python -m src.compare_ppo
```

## One-step TD critic target ablation

This controlled PPO variant changes only the critic regression target from the
GAE lambda-return to `r + gamma * (1-terminated) * V_old(next_state)`. The actor
still uses normalized GAE with lambda 0.95. All networks, initialization, learning
rates, rollout/minibatch sizes and the 1,384,029-step budget match the frozen
GAE-target PPO run. See [training comparison](docs/TD1_CRITIC_COMPARISON.md).

Observed results for training seed 101 (each model uses 1,384,029 interactions):

| Critic target | First full 100-episode training mean of 500: interactions | Fresh 500-episode evaluation mean / sample SD |
|---|---:|---:|
| GAE lambda-return |228,168|500 / 0|
| One-step TD |331,544|500 / 0|

The GAE-target run reaches the perfect training window earlier in this seed;
both final policies reach the official cap in every fresh evaluation episode.
Training-window crossings are descriptive, not frozen-policy evaluations or a
general performance ranking. The comparison includes full and enlarged training
curves, own-target critic MSE, evaluation histograms and initial-state critic errors.

```sh
.venv/bin/python -m src.validate_td1_critic
caffeinate -i .venv/bin/python -u -m src.ppo train \
  --run runs/cartpole_ppo_td1_critic --critic-target td1
.venv/bin/python -m src.ppo evaluate --run runs/cartpole_ppo \
  --output runs/td1_critic_eval/gae --episodes 500 --eval-seed-base 88000000
.venv/bin/python -m src.ppo evaluate --run runs/cartpole_ppo_td1_critic \
  --output runs/td1_critic_eval/td1 --episodes 500 --eval-seed-base 88000000
.venv/bin/python -m src.diagnose_td1_critic
.venv/bin/python -m src.compare_td1_critic
```

The new complete-episode evaluation seeds are separate from the earlier results.
Initial-state critic diagnostics reuse those recorded returns and seeded resets;
they do not add training or evaluation episodes. Own-target training losses use
different regression labels and cannot by themselves rank value accuracy.
