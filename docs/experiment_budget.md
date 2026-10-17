# Measured device choice and formal experiment budget

The budget below was fixed after the full local device benchmark and tuning pilots, before any formal training or held-out evaluation. All runs use the actual M4 Pro Mac (24 GB), new compute spend $0.

## Benchmark

Command: `.venv/bin/python src/rl.py benchmark --seconds 300 --batch 8`, executed outside the Codex process sandbox so native MPS was available. CPU and MPS ran sequentially in separate fresh processes with the same 6400-64-2 policy, image pipeline, environment, initial weights, seed sequence, eight complete episodes per update, Adam 0.001 and identical no-baseline loss. No formal training ran concurrently.

| Device | Seconds | Environment steps | Steps/second | Mean update seconds | Peak process RSS bytes |
|---|---:|---:|---:|---:|---:|
| CPU | 300.919967 | 1,264,795 | 4,203.0943 | 0.042052 | 609,402,880 |
| MPS | 301.276259 | 969,323 | 3,217.3893 | 0.093687 | 946,143,232 |

CPU is approximately 30.64% faster in end-to-end throughput, uses less process memory, and is selected for formal Pong runs. CPU also runs shuttle and CartPole. Each timed trial completed its last full batch rather than cutting an episode. MPS allocation snapshots are separately saved; they are not claimed to be a sampled device-allocation peak. The process RSS peak includes all CPU-visible process allocations in each independent process. The benchmark includes environment stepping, preprocessing, inference, CPU/device transfers, returns and backpropagation, but excludes process import/setup before the timer. No game display was opened.

A preliminary CPU run of roughly one minute was stopped to correct the original shared-process high-water-memory design. Its raw records are retained in `data/benchmark_preliminary/`, excluded from final comparisons. The corrected comparison is in `data/benchmark/`; no results were fabricated or reused as formal training.

## Pilot attempts and learning-rate choice

- CartPole, seed 91, 80 episodes, Adam 0.01: mean raw reward 34.025; final batch 41.875. This checked execution only and is not a formal model.
- Pong, seed 92, 128 episodes, Adam 0.01: mean raw training reward -20.8046875. Independent eight-episode tuning diagnostic mean -21, mean action entropy 0.5402572 nats; the fraction of steps with max action probability >0.999 was 0.
- Pong, same seed 92, 128 episodes, Adam 0.001: mean raw training reward -20.4453125. Same independent tuning seeds gave mean -20.875, entropy 0.6878748, saturation fraction 0.

Both learning rates produced losing short-run policies; neither pilot solved Pong. The larger rate reduced entropy much more quickly. I conservatively choose 0.001 for both formal Pong methods to preserve exploration and reduce abrupt changes in sparse-reward gradient updates. The pilot score difference is weak, single-seed evidence and is not a reliable estimate of final superiority. The larger rate did not numerically diverge or fully saturate. CartPole retains the assignment default 0.01.

Pilot config/CSV/diagnostic records and logs are retained. The official final evaluation seed range was not used in tuning. Tuning diagnostic seeds start at 91,000,000, separate from final evaluation at 80,000,000 and all training seeds.

## Formal budget and stop rule

| Model | Complete train episodes | Updates (batch 8) | Seed | Adam LR | Device |
|---|---:|---:|---:|---:|---|
| CartPole basic | 4096 | 512 | 101 | 0.01 | CPU |
| CartPole baseline | 4096 | 512 | 101 | 0.01 | CPU |
| Pong basic | 4096 | 512 | 202 | 0.001 | CPU |
| Pong baseline | 4096 | 512 | 202 | 0.001 | CPU |

Each Pong run contains 32 times its 128-episode pilot budget and 512 Monte Carlo batch updates. At measured throughput and roughly 1,000 decision steps per losing game, the initial runtime estimate is about 16-17 minutes per Pong model, with longer games increasing runtime. This estimate does not impose an episode timeout. The full matched baseline/no-baseline budget is retained even if one method learns faster. CartPole is much cheaper locally; 4,096 episodes provide a long observed learning curve under the same 512-update comparison.

Stop solely on the fixed complete-episode count. Freeze the final model after the last update, hash it, and then run 500 complete held-out environment episodes. Do not select among intermediate checkpoints using final results. Record poor or unstable outcomes honestly; the budget does not guarantee a winning Pong agent. Retain all official environment episode limits, all evaluation episodes, and full original task definitions. Memory use is controlled by graph-free collection and chunked log-probability recomputation.

One training seed per method/environment is the explicit evidence limit. The report discusses this limitation and further three-month experiments. There is no wall-time cancellation rule; interruption uses the recorded resume command.

## Actual commands

```sh
.venv/bin/python src/rl.py train --run runs/cartpole_basic --game cartpole --episodes 4096 --seed 101 --lr 0.01 --batch 8 --device cpu
.venv/bin/python src/rl.py train --run runs/cartpole_baseline --game cartpole --episodes 4096 --seed 101 --lr 0.01 --batch 8 --device cpu --baseline
.venv/bin/python src/rl.py train --run runs/pong_basic --game pong --episodes 4096 --seed 202 --lr 0.001 --batch 8 --device cpu
.venv/bin/python src/rl.py train --run runs/pong_baseline --game pong --episodes 4096 --seed 202 --lr 0.001 --batch 8 --device cpu --baseline
```

The runner executes these exact settings and 500 full evaluations after each frozen run: `caffeinate -i .venv/bin/python -u scripts/run_experiments.py`. It resumes automatically and skips frozen/complete runs. Loss uses a fixed positive scale 100 and gradient norm bound 10, identical across methods; it does not center returns. No further hyperparameter tuning is scheduled from final evaluation data.
