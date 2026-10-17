# Frozen-policy gradient and random-policy diagnostic

This command is prepared for execution **after final checkpoints are frozen**:

```sh
.venv/bin/python -m src.gradient_diagnostic \
  --run runs/cartpole_baseline --run runs/pong_baseline \
  --episodes 32 --device cpu
```

It produces `data/validation/gradient_diagnostic.json`, a generated quantitative
discussion with the same basename and `.md` extension, and raw CSV records in
`data/validation/gradient_diagnostic_records/`. Paths can be changed using
`--run` and `--output`. The final checkpoint hashes are verified before use.
No policy or optimizer update occurs.

For each baseline-trained final policy, 32 complete diagnostic episodes use
seed base 92000000. On the **same actual trajectories**, the program calculates
two per-episode gradient vectors:

\[
g_i(b)=\sum_t \gamma^t\nabla\log\pi(a_t|s_t)(G_t-b)/100,
\]

with b=0 and b equal to the final saved scalar EMA. That saved scalar is frozen
before every diagnostic action, preserving its action independence. The vectors
are reward-gradient estimators; negating them gives the production per-episode
policy-loss gradients. Production float32 weights, policy network, environment,
preprocessing, and bounded chunking are used. Gradient clipping and Adam are
excluded because the target here is estimator variance before the aggregated
batch gradient is transformed by an optimizer.

The covariance trace is

\[
\operatorname{tr}(\widehat{\operatorname{Cov}}(g))
=\frac{1}{n-1}\sum_i\|g_i-\overline g\|^2.
\]

Float64 vector Welford updates store two mean vectors and scalar sums of centered
squared norms. The program never stores a 32-by-parameter gradient matrix.
One complete episode's observations and one bounded activation chunk are
retained at a time. Per-episode CSVs preserve raw reward, length, termination,
truncation, gradient squared norms, paired dot products and squared distances,
paired cosines, and projections onto one fixed unit Rademacher direction.
Summary data also preserve sample mean gradient squared norms and the variance
ratio with versus without the saved baseline. A ratio above one is a valid
possible outcome and is reported directly.

Resume the exact same command after interruption. An atomic compressed state
file commits online moments after every complete episode. Any uncommitted CSV
row is trimmed on resume, and parameters/configuration/seed metadata must match.
The state stores only the two online vector means, not all episodic gradients.

By default, the same command also runs a uniform random policy for 32 complete
episodes per game, using separate seed base 93000000. It uses zero policy logits,
the identical environment and permitted action set, and graph-free collection.
Its CSV preserves every raw reward, length, and terminal/truncation flag; its
summary reports mean and sample standard deviation (`ddof=1`). Use
`--random-episodes` to change its count or `--skip-random-reference` to omit it.
These are held-out diagnostic references, excluded from the four required
500-episode trained-policy evaluations and from checkpoint selection.

The paired gradient comparison concerns one fixed learned policy in each game.
It does not compare independent training seeds or prove that using a baseline
improves final score. Thirty-two episodes have limited precision, especially
for sparse Pong rewards. A finite-sample difference between the two mean
gradients does not establish bias, since the empirical mean baseline score need
not be exactly zero. The random-policy reference supports descriptive task-level
comparisons; it is not a formal test of superiority across training runs.

For Pong, its per-episode CSV also records the zero-based decision index of the
first nonzero reward, that reward, its first contribution `gamma**t * reward`
to the episode-start objective, and the complete discounted episode return.
The summary reports the range and mean of the first-reward indices across the
32 actual diagnostic games. These timing fields are added only to the Pong
schema, preserving existing CartPole diagnostic records and resume state.

The first actual complete Pong diagnostic episode also produces
`data/validation/pong_trace.json` and `pong_trace.npz`. The JSON states the
purpose, theory, method, actual results, and interpretation, and lists the first
ten decision inputs' foreground/motion pixel counts, policy/environment actions,
rewards, complete-game return-to-go, and separate `gamma**t` weights. The small
compressed archive preserves the first ten signed 80x80 inputs, corresponding
binary foreground images, their action indices, and the full sparse episode
reward vector so the first-ten returns can be regenerated. Foreground images
are obtained by a deterministic replay of those decision states with the same
reset seed and actions; their differences are asserted equal to the collected
policy inputs. This short trace replay is excluded from all episode counts.
The first reset input is zero by the production frame-difference convention.
Later point rewards affect earlier return-to-go values; they remain reward
events within one full game rather than being counted as complete episodes.
