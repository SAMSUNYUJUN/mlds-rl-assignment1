# Progress and recovery

Updated: 2026-10-17, America/Chicago.

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
