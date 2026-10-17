"""Paired gradient-variance diagnostics under frozen learned policies.

This is a post-training diagnostic, not another training run or final evaluation.
Each actual complete diagnostic episode is used for both b=0 and the frozen
prior-batch scalar baseline. No parameters or optimizer states are updated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import rl


FIELDS = [
    "episode", "seed", "reward", "length", "terminated", "truncated",
    "gradient0_norm_squared", "gradient_baseline_norm_squared",
    "gradient_pair_inner_product", "gradient_pair_squared_distance",
    "gradient_pair_cosine", "random_projection0", "random_projection_baseline",
    "elapsed_seconds",
]
REFERENCE_FIELDS = ["episode", "seed", "reward", "length", "terminated", "truncated", "elapsed_seconds"]
PONG_TIMING_FIELDS = ["first_nonzero_reward_step", "first_nonzero_reward",
                      "first_nonzero_reward_discounted_contribution", "episode_discounted_return"]


def gradients(policy, trajectory, cfg, device, baseline):
    """Two pre-clipping, unaveraged episodic reward-gradient estimators.

    Use exactly production float32 advantage weights and bounded chunks. The
    negative of each vector is the per-episode policy-loss gradient. A complete
    batch's loss divides their sum by its episode count; clipping is applied
    later to that batch sum and is excluded from this variance diagnostic.
    """
    parameters = tuple(policy.parameters())
    shapes = [parameter.numel() for parameter in parameters]
    total = sum(shapes)
    g0 = np.zeros(total, dtype=np.float64)
    gb = np.zeros(total, dtype=np.float64)
    returns = rl.returns_to_go(trajectory["rewards"], cfg["gamma"])
    weights = cfg["gamma"] ** np.arange(len(returns), dtype=np.float64)
    advantage0 = (weights * returns / cfg["loss_scale"]).astype(np.float32)
    advantageb = (weights * (returns - baseline) / cfg["loss_scale"]).astype(np.float32)
    for lo in range(0, len(returns), cfg["chunk_size"]):
        hi = lo + cfg["chunk_size"]
        observations = torch.as_tensor(np.asarray(trajectory["observations"][lo:hi]),
                                       dtype=torch.float32, device=device)
        actions = torch.as_tensor(trajectory["actions"][lo:hi], dtype=torch.int64, device=device)
        log_probs = policy(observations).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        score0 = (log_probs * torch.as_tensor(advantage0[lo:hi], device=device).detach()).sum()
        scoreb = (log_probs * torch.as_tensor(advantageb[lo:hi], device=device).detach()).sum()
        chunk0 = torch.autograd.grad(score0, parameters, retain_graph=True)
        chunkb = torch.autograd.grad(scoreb, parameters)
        offset = 0
        for a, b, count in zip(chunk0, chunkb, shapes):
            g0[offset:offset + count] += a.detach().cpu().numpy().reshape(-1)
            gb[offset:offset + count] += b.detach().cpu().numpy().reshape(-1)
            offset += count
        del chunk0, chunkb, log_probs, score0, scoreb
    return g0, gb


class OnlineMoments:
    """Store O(parameter count), never the episode-by-parameter matrix."""

    def __init__(self, size):
        self.n = 0
        self.mean0 = np.zeros(size, np.float64)
        self.meanb = np.zeros(size, np.float64)
        self.m20 = 0.0
        self.m2b = 0.0

    def add(self, g0, gb):
        self.n += 1
        delta0, deltab = g0 - self.mean0, gb - self.meanb
        self.mean0 += delta0 / self.n
        self.meanb += deltab / self.n
        self.m20 += float(delta0 @ (g0 - self.mean0))
        self.m2b += float(deltab @ (gb - self.meanb))

    def save(self, path, metadata):
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, n=self.n, mean0=self.mean0, meanb=self.meanb,
                            m20=self.m20, m2b=self.m2b,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
        temporary.replace(path)

    def load(self, path, metadata):
        with np.load(path, allow_pickle=False) as state:
            if json.loads(str(state["metadata"])) != metadata:
                raise ValueError(f"Diagnostic resume metadata mismatch: {path}")
            self.n = int(state["n"])
            self.mean0 = state["mean0"].copy()
            self.meanb = state["meanb"].copy()
            self.m20 = float(state["m20"])
            self.m2b = float(state["m2b"])


def nonzero_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def trim_raw_rows(path, count, fields=FIELDS):
    """Checkpoint commits state after raw CSV; discard an uncommitted last row."""
    if not path.exists():
        if count:
            raise ValueError(f"Missing raw diagnostic records for checkpoint: {path}")
        return []
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < count:
        raise ValueError(f"Raw diagnostic records shorter than checkpoint: {path}")
    rows = rows[:count]
    if [int(row["episode"]) for row in rows] != list(range(count)):
        raise ValueError(f"Noncontiguous raw diagnostic records: {path}")
    path.unlink()
    rl.append_csv(path, rows, fields)
    return rows


def pong_reward_timing(trajectory, gamma):
    """All step indices are zero-based policy-decision steps (frameskip=4)."""
    rewards = trajectory["rewards"]
    first = next((t for t, reward in enumerate(rewards) if reward != 0), None)
    return dict(first_nonzero_reward_step=first,
                first_nonzero_reward=float(rewards[first]) if first is not None else None,
                first_nonzero_reward_discounted_contribution=float(gamma ** first * rewards[first]) if first is not None else None,
                episode_discounted_return=float(rl.returns_to_go(rewards, gamma)[0]))


def save_pong_trace(trajectory, cfg, metadata, output_folder):
    """Preserve small, interpretable evidence from an actual full episode.

    Replay just the first ten decisions with the same ALE seed/actions to obtain
    binary foreground counts. Equality to the collected signed inputs is checked;
    this replay is not another complete diagnostic/evaluation episode.
    """
    length = min(10, len(trajectory["rewards"]))
    inputs = np.asarray(trajectory["observations"][:length], dtype=np.int8).reshape(length, 80, 80)
    if inputs.shape != (length, 80, 80):
        raise AssertionError("Unexpected Pong trace input shape")
    rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
    returns = rl.returns_to_go(rewards, cfg["gamma"])
    foregrounds = []
    replay_env = rl.make_env("pong")
    try:
        raw, _ = replay_env.reset(seed=trajectory["seed"])
        previous = rl.preprocess(raw)
        current = previous
        observed = np.zeros(6400, dtype=np.int8)
        for t in range(length):
            if not np.array_equal(observed, inputs[t].reshape(-1)):
                raise AssertionError(f"Seeded Pong first-ten trace replay differs at decision {t}")
            foregrounds.append(current.reshape(80, 80).copy())
            if t + 1 < length:
                raw, _, terminated, truncated, _ = replay_env.step([2, 3][trajectory["actions"][t]])
                if terminated or truncated:
                    raise AssertionError("Trace replay terminated before recorded trajectory")
                current = rl.preprocess(raw)
                observed = current - previous
                previous = current
    finally:
        replay_env.close()
    trace_rows = []
    for t, (frame, foreground) in enumerate(zip(inputs, foregrounds)):
        trace_rows.append(dict(step=t, policy_action_index=int(trajectory["actions"][t]),
                               environment_action=[2, 3][trajectory["actions"][t]],
                               reward=float(rewards[t]), return_to_go=float(returns[t]),
                               objective_time_weight=float(cfg["gamma"] ** t),
                               current_foreground_pixel_count=int(np.count_nonzero(foreground)),
                               nonzero_difference_pixel_count=int(np.count_nonzero(frame)),
                               positive_difference_pixel_count=int(np.count_nonzero(frame == 1)),
                               negative_difference_pixel_count=int(np.count_nonzero(frame == -1)),
                               difference_pixel_sum=int(frame.sum())))
    trace_npz = output_folder / "pong_trace.npz"
    temporary = trace_npz.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, first_ten_inputs=inputs,
                        first_ten_binary_foregrounds=np.asarray(foregrounds, dtype=np.int8),
                        first_ten_policy_actions=np.asarray(trajectory["actions"][:length], dtype=np.int8),
                        full_episode_rewards=rewards)
    temporary.replace(trace_npz)
    result = dict(
        purpose="Inspect actual early image states/actions/rewards/returns and quantify sparse Pong credit timing under the frozen learned policy.",
        theoretical_expectation="Reset input is an all-zero frame difference. A reward first seen at zero-based decision t contributes gamma^t*r_t to the episode-start discounted objective, while G_t includes all later rewards of the complete game.",
        method="Use the first complete held-out Pong gradient-diagnostic episode, without policy updates. Preserve the first ten 80x80 signed input images and the complete sparse reward vector. Deterministically replay only the first ten decisions with the identical seed/actions to count binary foreground pixels and verify the saved input differences exactly; this replay is excluded from episode counts.",
        checkpoint_sha256=metadata["checkpoint_sha256"], run=metadata["run"],
        seed=trajectory["seed"], gamma=cfg["gamma"], step_indexing="zero-based policy decisions, with native configured frameskip=4",
        episode_length=len(rewards), raw_episode_reward=float(rewards.sum()),
        terminated=trajectory["terminated"], truncated=trajectory["truncated"],
        results={**pong_reward_timing(trajectory, cfg["gamma"]), "first_ten_steps": trace_rows,
                 "input_difference_values": sorted(int(value) for value in np.unique(inputs)),
                 "reset_input_nonzero_pixels": int(np.count_nonzero(inputs[0])),
                 "seeded_trace_replay_inputs_identical": True},
        compressed_arrays=str(trace_npz.relative_to(rl.ROOT)) if trace_npz.is_relative_to(rl.ROOT) else str(trace_npz),
        explanation="Early zero rewards do not imply zero return-to-go: later point rewards enter G_t, attenuated by gamma. The episode-start objective additionally downweights late actions by gamma^t. A Pong point is a reward event, not an episode termination. This one trace illustrates a particular frozen-policy episode and does not establish general convergence or baseline benefit.",
    )
    rl.atomic_json(output_folder / "pong_trace.json", result)


def diagnose_run(run, args):
    run = Path(run)
    if not run.is_absolute():
        run = rl.ROOT / run
    frozen = json.loads((run / "FROZEN.json").read_text())
    checkpoint_path = run / frozen["file"]
    if rl.sha256(checkpoint_path) != frozen["sha256"]:
        raise ValueError(f"Frozen checkpoint hash mismatch: {run}")
    saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    cfg = saved["config"]
    if not cfg["baseline"]:
        raise ValueError(f"This paired diagnostic requires a baseline-trained model: {run}")
    if args.seed_base < cfg["train_seed_base"] + saved["episode"] and args.seed_base + args.episodes > cfg["train_seed_base"]:
        raise ValueError("Diagnostic seed range overlaps training seeds")
    if args.seed_base < cfg["eval_seed_base"] + 500 and args.seed_base + args.episodes > cfg["eval_seed_base"]:
        raise ValueError("Diagnostic seed range overlaps final evaluation seeds")
    policy = rl.Policy(cfg["game"], cfg["hidden"]).to(args.device)
    policy.load_state_dict(saved["model"])
    policy.eval()
    parameters_before = [parameter.detach().cpu().clone() for parameter in policy.parameters()]
    count = sum(parameter.numel() for parameter in policy.parameters())
    baseline = float(saved["baseline"])
    projection_rng = np.random.default_rng(args.projection_seed)
    projection = projection_rng.choice(np.array([-1.0, 1.0]), size=count) / np.sqrt(count)
    metadata = dict(run=str(run.relative_to(rl.ROOT)) if run.is_relative_to(rl.ROOT) else str(run),
                    game=cfg["game"], checkpoint_sha256=frozen["sha256"],
                    saved_baseline=baseline, parameter_count=count,
                    episodes=args.episodes, seed_base=args.seed_base, device=args.device,
                    projection_seed=args.projection_seed, gamma=cfg["gamma"],
                    loss_scale=cfg["loss_scale"], chunk_size=cfg["chunk_size"],
                    train_seed_base=cfg["train_seed_base"], trained_episode_count=saved["episode"],
                    eval_seed_base=cfg["eval_seed_base"])
    output_folder = args.output.parent / "gradient_diagnostic_records"
    output_folder.mkdir(parents=True, exist_ok=True)
    basename = run.name
    state_path = output_folder / f"{basename}_state.npz"
    raw_path = output_folder / f"{basename}.csv"
    moments = OnlineMoments(count)
    if state_path.exists():
        moments.load(state_path, metadata)
    fields = FIELDS + PONG_TIMING_FIELDS if cfg["game"] == "pong" else FIELDS
    rows = trim_raw_rows(raw_path, moments.n, fields)
    prior_elapsed = float(rows[-1]["elapsed_seconds"]) if rows else 0.0
    start = time.perf_counter()
    while moments.n < args.episodes:
        episode = moments.n
        trajectory = rl.collect(policy, cfg["game"], args.device, [episode], args.seed_base)[0]
        if cfg["game"] == "pong" and episode == 0:
            save_pong_trace(trajectory, cfg, metadata, args.output.parent)
        g0, gb = gradients(policy, trajectory, cfg, args.device, baseline)
        moments.add(g0, gb)
        norm0_squared, normb_squared = float(g0 @ g0), float(gb @ gb)
        dot = float(g0 @ gb)
        row = dict(episode=episode, seed=args.seed_base + episode,
                   reward=float(sum(trajectory["rewards"])), length=len(trajectory["rewards"]),
                   terminated=trajectory["terminated"], truncated=trajectory["truncated"],
                   gradient0_norm_squared=norm0_squared,
                   gradient_baseline_norm_squared=normb_squared,
                   gradient_pair_inner_product=dot,
                   gradient_pair_squared_distance=float((g0 - gb) @ (g0 - gb)),
                   gradient_pair_cosine=nonzero_ratio(dot, np.sqrt(norm0_squared * normb_squared)),
                   random_projection0=float(projection @ g0),
                   random_projection_baseline=float(projection @ gb),
                   elapsed_seconds=prior_elapsed + time.perf_counter() - start)
        if cfg["game"] == "pong":
            row.update(pong_reward_timing(trajectory, cfg["gamma"]))
        rl.append_csv(raw_path, [row], fields)
        moments.save(state_path, metadata)
        print(json.dumps(dict(run=basename, episode=moments.n, diagnostic_reward=row["reward"],
                              length=row["length"], elapsed_seconds=round(row["elapsed_seconds"], 2))), flush=True)
        del trajectory, g0, gb
    parameters_unchanged = all(torch.equal(before, after.detach().cpu())
                               for before, after in zip(parameters_before, policy.parameters()))
    if not parameters_unchanged:
        raise AssertionError("Diagnostic unexpectedly changed frozen parameters")
    trace0 = max(0.0, moments.m20 / (moments.n - 1))
    traceb = max(0.0, moments.m2b / (moments.n - 1))
    with raw_path.open() as stream:
        rows = list(csv.DictReader(stream))
    projections0 = np.array([float(row["random_projection0"]) for row in rows])
    projectionsb = np.array([float(row["random_projection_baseline"]) for row in rows])
    mean0norm2 = float(moments.mean0 @ moments.mean0)
    meanbnorm2 = float(moments.meanb @ moments.meanb)
    result = dict(**metadata, n=moments.n,
                  trace_sample_gradient_covariance_no_baseline=trace0,
                  trace_sample_gradient_covariance_saved_baseline=traceb,
                  variance_ratio_saved_baseline_over_no_baseline=nonzero_ratio(traceb, trace0),
                  sample_mean_gradient_squared_norm_no_baseline=mean0norm2,
                  sample_mean_gradient_squared_norm_saved_baseline=meanbnorm2,
                  paired_sample_mean_gradient_difference_squared_norm=float((moments.mean0 - moments.meanb) @ (moments.mean0 - moments.meanb)),
                  sample_mean_gradient_cosine=nonzero_ratio(float(moments.mean0 @ moments.meanb), np.sqrt(mean0norm2 * meanbnorm2)),
                  random_projection_sample_variance_no_baseline=float(projections0.var(ddof=1)),
                  random_projection_sample_variance_saved_baseline=float(projectionsb.var(ddof=1)),
                  random_projection_paired_difference_mean=float((projectionsb - projections0).mean()),
                  raw_episode_records=str(raw_path.relative_to(rl.ROOT)) if raw_path.is_relative_to(rl.ROOT) else str(raw_path),
                  online_resume_state=str(state_path.relative_to(rl.ROOT)) if state_path.is_relative_to(rl.ROOT) else str(state_path),
                  diagnostic_episode_reward_mean=float(np.mean([float(row["reward"]) for row in rows])),
                  diagnostic_episode_length_mean=float(np.mean([int(row["length"]) for row in rows])),
                  natural_terminations=sum(row["terminated"] == "True" for row in rows),
                  built_in_truncations=sum(row["truncated"] == "True" for row in rows),
                  parameters_unchanged=parameters_unchanged,
                  elapsed_seconds=float(rows[-1]["elapsed_seconds"]))
    if cfg["game"] == "pong":
        first_steps = [int(row["first_nonzero_reward_step"]) for row in rows if row["first_nonzero_reward_step"] != ""]
        contributions = [float(row["first_nonzero_reward_discounted_contribution"]) for row in rows
                         if row["first_nonzero_reward_discounted_contribution"] != ""]
        result["pong_sparse_reward_timing"] = dict(
            step_indexing="zero-based policy decisions with frameskip=4",
            episodes_with_nonzero_reward=len(first_steps),
            episodes_without_nonzero_reward=len(rows) - len(first_steps),
            first_nonzero_reward_step_mean=float(np.mean(first_steps)) if first_steps else None,
            first_nonzero_reward_step_min=min(first_steps) if first_steps else None,
            first_nonzero_reward_step_max=max(first_steps) if first_steps else None,
            first_nonzero_reward_discounted_contribution_mean=float(np.mean(contributions)) if contributions else None,
            first_episode_trace=str((args.output.parent / "pong_trace.json").relative_to(rl.ROOT))
            if (args.output.parent / "pong_trace.json").is_relative_to(rl.ROOT) else str(args.output.parent / "pong_trace.json"),
        )
    return result


def random_policy_reference(game, args):
    """Uniform action distribution, same complete environment and action set."""
    class UniformPolicy(torch.nn.Module):
        def forward(self, observations):
            return torch.zeros((len(observations), 2), device=observations.device)

    output_folder = args.output.parent / "gradient_diagnostic_records"
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / f"{game}_uniform_random_reference.csv"
    if path.exists():
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
    if len(rows) > args.random_episodes:
        raise ValueError(f"Reference file already contains more than requested episodes: {path}")
    if [int(row["seed"]) for row in rows] != list(range(args.random_seed_base, args.random_seed_base + len(rows))):
        raise ValueError(f"Random-reference seed mismatch: {path}")
    elapsed = float(rows[-1]["elapsed_seconds"]) if rows else 0.0
    start = time.perf_counter()
    policy = UniformPolicy().to(args.device)
    while len(rows) < args.random_episodes:
        episode = len(rows)
        tr = rl.collect(policy, game, args.device, [episode], args.random_seed_base, retain=False)[0]
        row = dict(episode=episode, seed=args.random_seed_base + episode,
                   reward=float(sum(tr["rewards"])), length=len(tr["rewards"]),
                   terminated=tr["terminated"], truncated=tr["truncated"],
                   elapsed_seconds=elapsed + time.perf_counter() - start)
        rl.append_csv(path, [row], REFERENCE_FIELDS)
        # Keep rows' types identical to a resumed CSV read.
        rows.append({key: str(value) for key, value in row.items()})
        print(json.dumps(dict(reference=game, episode=len(rows), reward=row["reward"],
                              length=row["length"], elapsed_seconds=round(row["elapsed_seconds"], 2))), flush=True)
    rewards = np.array([float(row["reward"]) for row in rows])
    return dict(game=game, n=len(rows), seed_base=args.random_seed_base,
                policy="uniform distribution over the same two permitted actions",
                reward_mean=float(rewards.mean()), reward_std_sample=float(rewards.std(ddof=1)),
                reward_min=float(rewards.min()), reward_max=float(rewards.max()),
                mean_episode_length=float(np.mean([int(row["length"]) for row in rows])),
                environment_steps=sum(int(row["length"]) for row in rows),
                natural_terminations=sum(row["terminated"] == "True" for row in rows),
                built_in_truncations=sum(row["truncated"] == "True" for row in rows),
                raw_episode_records=str(path.relative_to(rl.ROOT)) if path.is_relative_to(rl.ROOT) else str(path),
                elapsed_seconds=float(rows[-1]["elapsed_seconds"]))


def write_discussion(path, results, references):
    lines = [
        "# Paired gradient diagnostic under frozen learned policies", "",
        "Purpose: measure whether the actual saved scalar baseline changes Monte Carlo policy-gradient variance under each learned policy.", "",
        "Theory: a scalar baseline fixed before the diagnostic actions preserves the expected score-function gradient. It need not reduce its variance. The diagnostic measures a sample variance, not exact population variance or training-seed variability.", "",
        "Method: collect complete held-out diagnostic episodes, then compute both estimators on the same actual observations/actions/rewards. Use gamma^t return-to-go /100 and production float32 chunked recomputation. Vectors are reward gradients before clipping or Adam. No parameter updates occur. The scalar is the final checkpoint's saved EMA, held fixed for all episodes.", "",
        "For gradient vectors g_i, trace(sample covariance) = sum_i ||g_i - mean(g)||^2 / (n-1). Online float64 vector Welford updates avoid storing an episode-by-parameter matrix. Per-episode norms, paired inner products/distances, cosines, and one fixed unit Rademacher projection are preserved in CSV. Resume state preserves only two mean vectors and two scalar M2 accumulators.", "",
        "| Learned policy | n | Saved b | Trace covariance b=0 | Trace covariance saved b | Ratio b / 0 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        ratio = result["variance_ratio_saved_baseline_over_no_baseline"]
        lines.append(f"| {result['run']} | {result['n']} | {result['saved_baseline']:.6g} | {result['trace_sample_gradient_covariance_no_baseline']:.6g} | {result['trace_sample_gradient_covariance_saved_baseline']:.6g} | {ratio:.6g} |" if ratio is not None else
                     f"| {result['run']} | {result['n']} | {result['saved_baseline']:.6g} | 0 | {result['trace_sample_gradient_covariance_saved_baseline']:.6g} | undefined |")
    lines.extend(["", "Interpretation: ratios below one indicate reduced sampled gradient variance for that policy and those episodes; ratios above one indicate increased sampled variance. Sample means may differ because the empirical average baseline score is not exactly zero. This finite-sample difference does not establish estimator bias. Saved policy parameters are checked unchanged.", "",
                  "Evidence limits: this is a paired fixed-policy diagnostic, not a repeated-training comparison. Thirty-two episodes provide limited precision, particularly with sparse Pong rewards. The diagnostic cannot establish that baseline training improves final reward or lowers the standard deviation of evaluation rewards. Diagnostic seed base 92000000 is distinct from final evaluation seed base 80000000; these rewards are excluded from final 500-episode evaluations.", ""])
    for result in results:
        timing = result.get("pong_sparse_reward_timing")
        if timing:
            lines.extend(["## Actual Pong sparse-reward timing", "",
                          f"Across {result['n']} complete diagnostic games, {timing['episodes_with_nonzero_reward']} had a nonzero reward. The first nonzero reward occurred at mean zero-based decision {timing['first_nonzero_reward_step_mean']}, range [{timing['first_nonzero_reward_step_min']}, {timing['first_nonzero_reward_step_max']}]. Each decision uses configured frameskip=4. The mean first reward's episode-start discounted contribution was {timing['first_nonzero_reward_discounted_contribution_mean']}.", "",
                          f"The first complete episode's first ten actual image inputs/actions/rewards/returns are preserved in `{timing['first_episode_trace']}` and its compressed array companion. Foreground counts come from a seeded trace replay whose signed inputs are verified identical. A point is a sparse reward within the full game; its gamma^t attenuation helps explain the credit available under the stated discounted objective. This observation does not itself identify the cause of any training failure.", ""])
    if references:
        lines.extend(["## Uniform random-policy reference", "",
                      "Purpose: compare actual task-level behavior with an uninformed uniform policy under exactly the same environment, preprocessing, action set, native termination, and built-in limits. Method: zero logits, 32 complete episodes by default, no gradient recording, separate seed base 93000000. This reference is collected after checkpoints are frozen and cannot be used for checkpoint selection.", "",
                      "| Game | n | Mean raw episode reward | Sample SD (n-1) | Mean episode length |", "|---|---:|---:|---:|---:|"])
        for reference in references:
            lines.append(f"| {reference['game']} | {reference['n']} | {reference['reward_mean']:.6g} | {reference['reward_std_sample']:.6g} | {reference['mean_episode_length']:.6g} |")
        lines.extend(["", "Interpretation: compare this small reference with the separately reported 500-episode trained-policy scores descriptively. Differences include seed sampling error; these are not matched repeated training runs or a formal superiority test. Full raw random-policy rewards, lengths, and terminal/truncation flags are preserved in CSV.", ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", help="Frozen baseline run; repeat for each model")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed-base", type=int, default=92000000)
    parser.add_argument("--projection-seed", type=int, default=92155)
    parser.add_argument("--random-episodes", type=int, default=32)
    parser.add_argument("--random-seed-base", type=int, default=93000000)
    parser.add_argument("--skip-random-reference", action="store_true")
    parser.add_argument("--output", type=Path, default=rl.ROOT / "data/validation/gradient_diagnostic.json")
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("At least two episodes are needed for sample covariance")
    if not args.skip_random_reference and args.random_episodes < 2:
        parser.error("At least two episodes are needed for random-reference sample standard deviation")
    runs = args.run or ["runs/cartpole_baseline", "runs/pong_baseline"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    references = []
    for run in runs:
        results.append(diagnose_run(run, args))
        game = results[-1]["game"]
        if not args.skip_random_reference and game not in {reference["game"] for reference in references}:
            cfg = results[-1]
            if args.random_seed_base < cfg["seed_base"] + args.episodes and args.random_seed_base + args.random_episodes > cfg["seed_base"]:
                parser.error("Random-reference seeds overlap gradient-diagnostic seeds")
            if args.random_seed_base < cfg["eval_seed_base"] + 500 and args.random_seed_base + args.random_episodes > cfg["eval_seed_base"]:
                parser.error("Random-reference seeds overlap final evaluation seeds")
            if args.random_seed_base < cfg["train_seed_base"] + cfg["trained_episode_count"] and args.random_seed_base + args.random_episodes > cfg["train_seed_base"]:
                parser.error("Random-reference seeds overlap training seeds")
            references.append(random_policy_reference(game, args))
        rl.atomic_json(args.output, dict(status="complete" if len(results) == len(runs) else "partial",
                                        estimator="pre-clipping episodic reward gradient; gamma^t RTG / loss_scale",
                                        covariance_denominator="n-1", no_parameter_updates=True, models=results,
                                        random_policy_references=references))
        write_discussion(args.output.with_suffix(".md"), results, references)
    print(json.dumps(dict(output=str(args.output), models=len(results)), indent=2))


if __name__ == "__main__":
    main()
