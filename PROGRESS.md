# Progress and recovery

Updated: 2026-10-19, America/Chicago.

## Completed local scientific work

- Read all four pages of mlds_assignment1.pdf. No other course materials were supplied. CODEX_TASK.md preserves the full original request; USER_CONTEXT.md records verified host and identity.
- Actual host: native arm64 Apple M4 Pro MacBook Pro, 24 GB, macOS 15.6.1. No remote compute; new compute budget $0. Dependencies pinned in requirements-lock.txt.
- Shuttle enumeration T=500, VI and PI complete; three plots and six quantitative validations. Optimal dispatch threshold 13, VI/PI max value difference 1.865e-9. Methods, assumptions and evidence in docs/shuttle_method.md and data/shuttle/.
- Self-written REINFORCE and detached prior-batch EMA baseline complete. Independent mathematical, production-update, chunking, preprocessing and exact checkpoint-resume checks pass (data/validation/rl.json).
- Full local CPU/MPS benchmark complete: CPU 4,203.09 steps/s versus MPS 3,217.39 steps/s, approximately five minutes each in separate native processes. CPU selected. Raw records and peak memory in data/benchmark/.
- Tuning pilots and learning-rate decision retained; formal plan fixed before formal runs and final evaluation in data/experiment_plan.json and docs/experiment_budget.md.
- ALL FOUR formal runs complete: 4,096 full training episodes each, 16,384 total; 10,689,593 formal training steps. Final checkpoints frozen before evaluation. No final-score checkpoint reselection.
- ALL FOUR final evaluations complete: 500 full episodes each, 2,000 total, 1,290,169 steps. Raw records and hashes in runs/{cartpole,pong}_{basic,baseline}/.
- Final mean/sample SD: CartPole basic 143.678/125.833885; CartPole baseline 423.592/83.414108; Pong basic -20.534/0.705578; Pong baseline -19.806/1.046209. All 1,000 Pong evaluations naturally terminated; no point/rally pseudo-episodes.
- Independent learned-policy diagnostics: 32 full episodes per baseline policy plus 32 uniform-reference episodes per game. Pong saved baseline reduces gradient covariance trace by approximately 46.16%; reward SD nevertheless increases. Actual ten-step image/action/return trace preserved. These episodes are excluded from final 2,000 evaluations.
- Eleven required figure contents and comparison panels generated from saved raw data; optimizer-update audit confirms zero clipped updates in every run.
- English report complete: output/pdf/assignment1_report.pdf, 16 pages; editable sources docs/report.md and src/build_report.py. All current pages visually inspected, all numbers/formulas/axes/legends/tables/pagination/links checked. SHA-256 a05f49e547e6e4d280b3b9d2679f1393b25aec979b894880b77c05b64f741474. Matching QA in data/report_visual_qa.json.
- Chinese learning guide explains algorithms, code, tests and actual conclusions for every figure. README contains actual setup, training, evaluation, resume, replot and report commands.
- Final local scientific/report acceptance passes with no failed required local checks; data/delivery_check.json and ACCEPTANCE.md retain the evidence.

## Process state

No training, evaluation or diagnostic process remains active. Main runner PID 29746 (tool session 27060) ended successfully. Historical PIDs in process records are not evidence of a live run. Logs are retained under logs/; per-run status and frozen files are under runs/.

## Delivery completed

- Archive created at output/assignment1_Yujun_Sun.zip (approximately 14 MB); all manifest payload hashes and CRC pass. It excludes virtual environments, caches and redundant model copies. Extracting it into a clean temporary folder without .git successfully rebuilt the 16-page report using the project interpreter and saved data, without training. Evidence: data/archive_rebuild_check.json. Final audit/repackage refreshes this packet after status changes.
- User explicitly authorized the final push. Source commit 74251b40ca7e59ea7108d2ac3ad4afd1230901a7 is delivered to existing public repository https://github.com/SAMSUNYUJUN/mlds-rl-assignment1 on main. Native GitHub API SHA matches the local commit; no new repository was created.
- Remote delivery is recorded in data/repository.json. Required-remote acceptance and the archive are refreshed with current delivery records; final status metadata is synced without retraining. The recorded experiments and repository delivery are complete; both Pong policies lost every final evaluation game, so this is not a claim that all learning objectives were achieved.
- Canvas submission is the user's action.

## Recovery and reproduction

Do not restart completed experiments. If investigating an interrupted environment, inspect actual processes, logs/experiments.log, runs/*/FROZEN.json and evaluation summaries first. All four final summaries must remain n=500.

Rebuild from saved data without retraining:

```sh
.venv/bin/python src/plot_results.py --panels
.venv/bin/python src/build_report.py
```

If rebuilding the PDF, re-inspect the current rendered pages and refresh data/report_visual_qa.json with its actual new hash before acceptance. To repeat only scientific validation, use src/shuttle.py validate and src/validate_rl.py. Existing gradient diagnostics resume from their own saved states without repeating completed episodes.

Audit/package commands:

```sh
.venv/bin/python scripts/check_delivery.py
.venv/bin/python scripts/package.py
```

For an actually unfinished future training run, the driver supports update-boundary checkpoint recovery via caffeinate -i .venv/bin/python -u scripts/run_experiments.py; it skips the current four frozen completed models. Never remove frozen manifests to restart these models accidentally.

## Report editorial revision (2026-10-17)

Removed task-material inspection, deadlines/timezone, authentication/API checks, permission and delivery narration from the academic report. Preserved methods, results, quantitative validation, reproducibility, required repository identity/link and explicit performance limitations. Rebuilt the 16-page PDF from saved data and visually inspected every page. No experiments or checkpoints changed. Current PDF SHA-256: a05f49e547e6e4d280b3b9d2679f1393b25aec979b894880b77c05b64f741474.

## PPO extension (2026-10-18, complete)

- Branch: `codex/cartpole-ppo-comparison`, based on main `5113b94`. Self-written PPO-Clip in `src/ppo.py`, state critic/GAE and update-boundary exact environment/optimizer/RNG recovery. Plan/config fixed before formal training in `data/ppo_plan.json`.
- Pilot8192 steps excluded from formal training/evaluation. Formal PPO CPU budget completed exactly1,384,029 interactions, matching original basic REINFORCE. Completed3,324 training episodes account for1,382,225 steps; eight unfinished episodes account for1,804 steps. Rollout slices are not logged as completed episodes.676 rollout updates,54,070 actor and54,070 critic optimizer steps. Logged elapsed48.426s, peakRSS310,460,416 bytes.
- Final-only checkpoint `runs/cartpole_ppo/final.pt`, SHA256 `8bf35c1067e785e910faddb1d8ab1f92718de55aa4b39618ff1715fd67d641eb`. Frozen at budget before evaluation; no evaluation-based selection.
- Historical protocol: PPO500 complete sampled episodes at seeds80m, mean500/sampleSD0. Primary new paired protocol: allthree frozen policies500 complete sampled episodes each at seeds84000000..84000499. Basic142.464/120.3097568; EMA419.182/83.4772651; PPO500/0. PPO reaches500-step cap in all500 new evaluation episodes. Totalnew evaluation episodes2,000.
- Independent validation: sixGAE cases, six signedclip derivative cases, exactinitialactor matching, exactfirstAdam actor update, critic regression direction, and both snapshot and actualtrain-loop interruption/resume equivalence. Actor/critic parametererrors0; crash-window log trimming prevents duplicateepisodes. Evidence `data/validation/ppo.json`.
- Integrity check recomputes rawmeans/SDs, exactbudget/remainingrollout/optimizercounts and confirms all20 original scientific artifacts unchanged: `data/ppo_integrity_check.json`, `data/ppo_original_artifacts.json`. Original assignmentPDF SHA remains `a05f49e547e6e4d280b3b9d2679f1393b25aec979b894880b77c05b64f741474`.
- Comparison outputs: `docs/PPO_COMPARISON.md`, `data/ppo_comparison_summary.json`, three `figures/ppo_comparison_*.png`. Allthree plots visually checked. Original report, models and rawlogs remain unchanged. Main assignment package remains its original delivery packet; PPO is documented separately on this branch.
- Limits: one training seed; baseline REINFORCE used1,290,047 steps (not exactlymatched). PPO simultaneously changes critic/GAE, learningrates, normalization, minibatchreuse, uniformtransition weighting and TimeLimit bootstrap. Results do not isolate clipping or timeweighting as solecause. Equalenvironmentsteps do not mean equalcompute.

No PPO training/evaluation process remains active. Native training PID21544 completed; historical process.json records are not live-process evidence. PPO logs: `logs/cartpole_ppo_train.log`, `logs/cartpole_ppo_evaluate_historical.log`, `logs/ppo_compare_*_evaluate.log`.

Reproduction/recovery commands (completed runs resume/skip safely; do not delete frozen manifests):

```sh
.venv/bin/python -m src.validate_ppo
caffeinate -i .venv/bin/python -u -m src.ppo train --run runs/cartpole_ppo
.venv/bin/python -m src.ppo evaluate --run runs/cartpole_ppo --episodes 500
for run in cartpole_basic cartpole_baseline cartpole_ppo; do
  .venv/bin/python -m src.ppo evaluate --run runs/$run --output runs/ppo_comparison_eval/$run --episodes 500 --eval-seed-base 84000000
done
.venv/bin/python -m src.compare_ppo
```

## One-step TD critic ablation (2026-10-19, complete)

- Branch `codex/cartpole-ppo-comparison`, starting at `5edb3fc`. Plan fixed in `data/td1_critic_plan.json`. Only the critic label changes to `reward + gamma * (1-terminated) * V_old(next)`; actor GAE (lambda 0.95), networks, initialization, optimizers and interaction budget are unchanged. The existing frozen GAE run is reused without modification.
- Local CPU formal run `runs/cartpole_ppo_td1_critic` completed exactly 1,384,029 environment interactions with seed 101. Its 3,554 completed episodes account for 1,381,881 steps; eight ongoing episodes account for 2,148 steps. There are 676 rollout updates, 54,070 actor steps and 54,070 critic steps, matching the control. Logged elapsed time 47.600 s; peak RSS 312,377,344 bytes. The 8,192-step timing pilot is excluded from formal results.
- Final-only frozen checkpoint `runs/cartpole_ppo_td1_critic/final.pt`, SHA-256 `6f00a5f3d4ff855e6651c381e9ca1ab0ca98d11966095ea498081005e8e8580d`. No checkpoint selection or tuning used final evaluation results.
- Fresh paired evaluation is complete: 500 full sampled episodes per model, seeds 88000000–88000499, saved under `runs/td1_critic_eval/{gae,td1}`. Both means are 500 and sample SDs are 0; all 1,000 episodes reach the official cap. No shortened evaluation episodes.
- Training-window result: the first full 100-episode mean of 500 occurs at 228,168 interactions for GAE targets and 331,544 for one-step TD targets. All subsequently completed training episodes also score 500 (2,311 and 2,104 respectively). These are descriptive single-seed training statistics, not intermediate frozen-policy evaluations or a general ranking.
- Frozen initial-state critic diagnostics reuse recorded evaluation returns and matched seeded resets. RMSE against each policy's sampled discounted return is 0.0025003 (GAE) versus 0.0063523 (TD1), both close to value 20. This is not measurement of true conditional expected value. Own-target training MSEs have different labels and cannot rank value accuracy.
- Quantitative validation passes: independent hand targets and physical/TimeLimit/slice masks; first rollout and actor GAE match exactly; first-batch actor parameter error 0 while critic parameters differ by 0.00037194; frozen labels; bitwise actual train-loop interruption/resume. Evidence `data/validation/td1_critic.json` and `data/validation/ppo_compat_td1.json`.
- Raw-log integrity checks recompute counters/statistics and verify 30 preserved original artifact hashes: `data/td1_critic_integrity.json`, `data/td1_critic_original_artifacts.json`. Original assignment PDF remains unchanged.
- Comparison document `docs/TD1_CRITIC_COMPARISON.md`, summary `data/td1_critic_comparison.json` and four `figures/td1_critic_*.png` plots are generated entirely from saved data.

No training, evaluation or diagnostic process remains active. Logs: `logs/td1_critic_train.log`, `logs/td1_critic_eval_{gae,td1}.log`, `logs/td1_critic_value_diagnostic.log`. Historical PIDs in saved process records are not live-process evidence. No experiment needs restarting.

Reproduce or safely resume (completed frozen runs are skipped):

```sh
.venv/bin/python -m src.validate_td1_critic
caffeinate -i .venv/bin/python -u -m src.ppo train --run runs/cartpole_ppo_td1_critic --critic-target td1
.venv/bin/python -m src.ppo evaluate --run runs/cartpole_ppo --output runs/td1_critic_eval/gae --episodes 500 --eval-seed-base 88000000
.venv/bin/python -m src.ppo evaluate --run runs/cartpole_ppo_td1_critic --output runs/td1_critic_eval/td1 --episodes 500 --eval-seed-base 88000000
.venv/bin/python -m src.diagnose_td1_critic
.venv/bin/python -m src.compare_td1_critic
```
