# Paired gradient diagnostic under frozen learned policies

Purpose: measure whether the actual saved scalar baseline changes Monte Carlo policy-gradient variance under each learned policy.

Theory: a scalar baseline fixed before the diagnostic actions preserves the expected score-function gradient. It need not reduce its variance. The diagnostic measures a sample variance, not exact population variance or training-seed variability.

Method: collect complete held-out diagnostic episodes, then compute both estimators on the same actual observations/actions/rewards. Use gamma^t return-to-go /100 and production float32 chunked recomputation. Vectors are reward gradients before clipping or Adam. No parameter updates occur. The scalar is the final checkpoint's saved EMA, held fixed for all episodes.

For gradient vectors g_i, trace(sample covariance) = sum_i ||g_i - mean(g)||^2 / (n-1). Online float64 vector Welford updates avoid storing an episode-by-parameter matrix. Per-episode norms, paired inner products/distances, cosines, and one fixed unit Rademacher projection are preserved in CSV. Resume state preserves only two mean vectors and two scalar M2 accumulators.

| Learned policy | n | Saved b | Trace covariance b=0 | Trace covariance saved b | Ratio b / 0 |
|---|---:|---:|---:|---:|---:|
| runs/cartpole_baseline | 32 | 19.9998 | 1.59873 | 2.45244e-10 | 1.534e-10 |
| runs/pong_baseline | 32 | -0.900326 | 0.047282 | 0.0254585 | 0.538439 |

Interpretation: ratios below one indicate reduced sampled gradient variance for that policy and those episodes; ratios above one indicate increased sampled variance. Sample means may differ because the empirical average baseline score is not exactly zero. This finite-sample difference does not establish estimator bias. Saved policy parameters are checked unchanged.

Evidence limits: this is a paired fixed-policy diagnostic, not a repeated-training comparison. Thirty-two episodes provide limited precision, particularly with sparse Pong rewards. The diagnostic cannot establish that baseline training improves final reward or lowers the standard deviation of evaluation rewards. Diagnostic seed base 92000000 is distinct from final evaluation seed base 80000000; these rewards are excluded from final 500-episode evaluations.

## Actual Pong sparse-reward timing

Across 32 complete diagnostic games, 32 had a nonzero reward. The first nonzero reward occurred at mean zero-based decision 115.34375, range [63, 245]. Each decision uses configured frameskip=4. The mean first reward's episode-start discounted contribution was 0.00988364877640589.

The first complete episode's first ten actual image inputs/actions/rewards/returns are preserved in `data/validation/pong_trace.json` and its compressed array companion. Foreground counts come from a seeded trace replay whose signed inputs are verified identical. A point is a sparse reward within the full game; its gamma^t attenuation helps explain the credit available under the stated discounted objective. This observation does not itself identify the cause of any training failure.

## Uniform random-policy reference

Purpose: compare actual task-level behavior with an uninformed uniform policy under exactly the same environment, preprocessing, action set, native termination, and built-in limits. Method: zero logits, 32 complete episodes by default, no gradient recording, separate seed base 93000000. This reference is collected after checkpoints are frozen and cannot be used for checkpoint selection.

| Game | n | Mean raw episode reward | Sample SD (n-1) | Mean episode length |
|---|---:|---:|---:|---:|
| cartpole | 32 | 20.3438 | 8.29539 | 20.3438 |
| pong | 32 | -20.5938 | 0.712079 | 1041.81 |

Interpretation: compare this small reference with the separately reported 500-episode trained-policy scores descriptively. Differences include seed sampling error; these are not matched repeated training runs or a formal superiority test. Full raw random-policy rewards, lengths, and terminal/truncation flags are preserved in CSV.
