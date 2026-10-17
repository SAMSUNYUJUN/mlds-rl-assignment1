# Independent policy-gradient and baseline validation

Run from the repository root:

```sh
.venv/bin/python -m src.validate_rl --output data/validation/rl.json
```

The saved JSON contains each test's purpose, theoretical expectation, method,
actual quantitative result, and interpretation. All numbers below were obtained
by running this command on the local Mac. The exact-enumeration examples are
small mathematical experiments; they are not claims about final CartPole or
Pong scores.

## Objective and estimator being checked

For an episode of length L, define the training objective and reward-to-go as

\[
J(\theta)=\mathbb E_{\pi_\theta}[\sum_{t=0}^{L-1}\gamma^t r_t],\qquad
G_t=\sum_{u=t}^{L-1}\gamma^{u-t}r_u.
\]

The episodic score-function estimator is

\[
\widehat {\nabla J}=\sum_{t=0}^{L-1}\gamma^t
\nabla\log\pi_\theta(a_t|s_t)(G_t-b_t).
\]

For a batch of B episodes the policy loss averages these sums over episodes.
A positive fixed loss scale multiplies this expression without changing the
gradient's sign or objective. It does change optimization magnitude and must be
recorded. `G_t` already contains discounts *relative to t*; the additional
`gamma**t` is needed for the explicitly stated episode-start discounted objective.
The basic version sets b=0, without centering or z-score normalization.

The baseline version uses a scalar frozen from earlier batches. Conditional on
the current pre-action history it cannot depend on the current action. In the
policy loss the advantage is detached. Its estimate is updated only after the
current batch has been sampled and used for its policy update. A moving average
is an estimator choice, not a guarantee of lower final episode-score variance.

## 1. Hand trajectory and discount boundary cases

**Purpose.** Detect return indexing and terminal-reward errors.

**Theory.** Rewards [1,2,3], gamma=.95, and zero continuation after the final
reward give G=[5.6075,4.85,3]. At gamma=0 the returns equal immediate rewards;
at gamma=1 they equal [6,5,3]. Empty and single-step trajectories should also
be well defined.

**Method and result.** An independent direct weighted sum is compared with a
backward recurrence. Maximum absolute error against the hand values was
8.88e-16. The boundary cases passed. The separate objective time weights were
[1,.95,.9025].

**Interpretation.** The last reward is included once and the terminal
continuation is zero. The test also prevents conflating RTG's internal discount
with the episode objective's absolute time discount.

## 2. Exact estimator expectation and baseline variance

**Purpose.** Check estimator unbiasedness and measure gradient variance without
sampling error.

**Theory.** Use three independent Bernoulli actions with
p=sigmoid(theta)=.35 and rewards [1+2a0,-.5+3a1,2-a2]. The analytic derivative is
p(1-p)(2+3 gamma-gamma^2)=0.89805625 for gamma=.95. A fixed scalar baseline
has zero expected score contribution.

**Method.** Enumerate all eight action trajectories, calculate their exact
probabilities, and compute the estimator's expectation and population variance.

| Fixed baseline b | Expected gradient | Gradient estimator variance |
|---:|---:|---:|
| 0 | 0.89805625 | 6.94115144 |
| 2 | 0.89805625 | 2.01123013 |
| 8 | 0.89805625 | 16.89117447 |

**Interpretation.** b=2 reduced variance by 71.0245%; b=8 increased it. All
three expected gradients agreed with the analytic derivative within floating
point precision. Thus an action-independent baseline preserves the expected
gradient, while a poorly chosen baseline can increase variance. No conclusion
about final game score follows from this toy experiment alone.

Two deliberate invalid estimators make the test sensitive to common mistakes.
Omitting `gamma**t` gave expected gradient .91, a bias of +.01194375 against the
declared objective. Subtracting the immediate action-dependent reward as a
baseline gave expected gradient approximately zero, a bias of -.89805625.

## 3. Why the baseline comes from earlier batches

**Purpose.** Quantify the self-dependence from subtracting a current-batch mean.

**Theory.** For N independent one-step samples, using each sample in its own
sample-mean return baseline scales the expected gradient by (1-1/N).

**Method and result.** Exact enumeration of all 16 Bernoulli action batches for
N=4 and reward r=1+2a gave .34125, versus the uncentered expected gradient .455.
The measured ratio was .75, as predicted.

**Interpretation.** An EMA frozen from completed earlier batches avoids this
specific action dependence. This is also why mean-centered returns cannot be
called a baseline-free implementation. The result is about the gradient
estimator, not an argument that every centered-return experiment must perform
poorly.

## 4. Loss sign, detach, and an actual parameter update

**Purpose.** Check whether minimizing the stated loss increases the objective
and excludes the baseline from policy differentiation.

**Theory.** For the exact experiment, the loss derivative should be
-.89805625, and a sufficiently small gradient step should increase J.

**Method and result.** A float64 PyTorch loss used frozen trajectory
probabilities and detached baseline b=2. Its derivative was -.89805625;
the central finite difference of analytic J was .89805624981. The detached
baseline's gradient was `None`. One gradient-descent step of size .01 increased
J from 3.711625 to 3.719700875, an increase of .008075875.

**Interpretation.** Loss sign, advantage detach, and analytic expectation agree.
This small SGD check isolates the mathematics. Actual training uses Adam, so it
does not establish convergence or tune Adam's learning rate.

## 5. Actual CartPole first-ten-step trace

**Purpose.** Connect the return equations to actual environment observations,
actions, rewards, and termination.

**Theory.** For an L-step CartPole episode with +1 per step, the complete
trajectory return is G_t=(1-.95^(L-t))/.05. The final reward belongs to the return.

**Method.** `CartPole-v1` reset seed 74219; an independent uniform policy uses
Python action seed 74220. The environment runs until its own natural terminal
or built-in TimeLimit. This diagnostic did not truncate an episode to ten steps.
The JSON saves the entire episode; the table shows only the first ten steps.
No autograd graph is created by the pure Python diagnostic policy.

| t | Observation [position, velocity, angle, angular velocity] | a | r | G_t |
|---:|---|---:|---:|---:|
| 0 | [-.0352,-.0335,.0192,-.0136] | 1 | 1 | 18.011195 |
| 1 | [-.0359,.1613,.0189,-.3001] | 1 | 1 | 17.906521 |
| 2 | [-.0327,.3562,.0129,-.5868] | 0 | 1 | 17.796338 |
| 3 | [-.0255,.1609,.0012,-.2900] | 0 | 1 | 17.680356 |
| 4 | [-.0223,-.0343,-.0046,.0030] | 1 | 1 | 17.558269 |
| 5 | [-.0230,.1609,-.0045,-.2911] | 1 | 1 | 17.429757 |
| 6 | [-.0198,.3561,-.0104,-.5852] | 1 | 1 | 17.294481 |
| 7 | [-.0127,.5514,-.0221,-.8812] | 0 | 1 | 17.152085 |
| 8 | [-.0016,.3565,-.0397,-.5955] | 0 | 1 | 17.002195 |
| 9 | [.0055,.1620,-.0516,-.3156] | 0 | 1 | 16.844416 |

**Result and interpretation.** The episode ended naturally after 45 steps,
reward 45, `terminated=True`, `truncated=False`. Backward returns and direct
sums differed by at most 3.55e-15; the geometric formula differed by at most
1.78e-14. These are real random-policy environment observations and are excluded
from formal trained-policy evaluation.

## 6. Production trainer cross-checks and checkpoint resume

**Purpose.** Verify that the actual training implementation follows the checked
equations, bounds activation memory without changing the update, preserves
signed image differences, and can resume an update boundary correctly.

**Theory.** Repeat the exhaustive three-step experiment at p=.5, where all
eight trajectories have equal probability. The exact objective derivative is
.986875. With loss scale 100 and SGD learning rate .01, the production update
must move theta from zero to .0000986875 for both b=0 and b=2. Production's
next-batch baseline must equal .9b+.1 times the weighted mean G. A single
backward expression and the chunked expression must yield the same Adam update,
up to float32 rounding. Serialized parameters, Adam moments, EMA baseline, and
deterministic per-episode RNGs must preserve the next update.

**Method.** The validator imports `src.rl` and calls its actual
`returns_to_go`, `Policy`, `collect`, `update`, `preprocess`, and
`save_checkpoint`. It compares production returns to independent direct sums.
It runs production `update` on all eight synthetic trajectories with chunks of
two, then compares chunk-size-one Adam against an independently written
single-graph loss on three saved synthetic trajectories. Synthetic image
pixels exercise background values 144/109 and foreground values 255/42.
For resume, it trains one actual four-episode CartPole batch, saves and reloads
model/Adam/EMA, and compares the next four rollouts and update against the
uninterrupted path. Reset/action seed base is 73400000, disjoint from final
evaluation seeds.

| Check | Actual quantitative result |
|---|---:|
| Production RTG versus direct weighted sum, maximum error | 8.88e-16 |
| Production theta after b=0 update | .0000986874976661 |
| Production theta after b=2 update | .0000986875011586 |
| b=2 next-batch EMA, computed and expected | 2.079097283085013 |
| Chunked versus single-graph parameter maximum error | 2.98e-8 |
| Chunked versus single-graph loss absolute error | 2.87e-9 |
| Preprocessed test pixels | [0,0,1,1] |
| Difference image range in test | [-1,0] |
| Resumed next-batch observations/actions/rewards | Identical |
| Resumed next-update parameter maximum error | 0 |
| Resumed next EMA baseline | Exactly equal |

Actual CartPole episode lengths in the two batches were
[17,16,16,45,20,26,21,14]. Preprocessing left the raw observation unchanged.

**Interpretation.** These cross-checks connect independent analytic expectations
to the production loss and confirm that signed disappearance pixels are -1
rather than unsigned wraparound. Chunked recomputation preserves the intended
batch update, while avoiding a long retained activation graph. The checkpoint
roundtrip is bitwise reproducible on CPU at an update boundary. It does not
claim exact CPU/MPS equality or recovery of partially completed episodes.

## Evidence limits and additional three-month experiments

The exact examples verify estimator mathematics independently of the trainer,
but do not demonstrate training convergence in a high-dimensional game. One
trace checks return semantics on one observed episode, not all failure modes.
Single-seed training comparisons have limited power to isolate a baseline's
effect from training randomness.

With three additional months, run at least 10 matched training seeds per
environment and method, with independent held-out evaluation seeds. Report
confidence intervals for differences in mean reward and sample efficiency.
At fixed checkpoints, collect independent gradient batches to measure gradient
covariance directly, compare the implemented EMA against an exact small-MDP
control and a learned value baseline, and vary EMA rate and batch size.
Test terminal/truncated trajectories separately, verify policy resumption under
saved RNG states, and compare CPU/MPS gradients on an identical saved batch.
For Pong, inspect first-score timing, sparse-reward credit assignment, sticky
actions, and image-difference edge cases; compare compatible image feature
networks without changing the required game, action set, or evaluation horizon.
