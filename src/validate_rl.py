"""Independent, exact checks of the REINFORCE estimator and baseline.

Run with ``.venv/bin/python -m src.validate_rl``. This module deliberately uses
independent exhaustive enumeration, rather than the training rollout code, to
check the estimator. Gymnasium is used only for an actual environment trace.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import platform
import random
import sys
import tempfile
import time


def return_to_go(rewards: list[float], gamma: float) -> list[float]:
    """Independent quadratic reference, used only for tiny validation traces."""
    return [
        sum(gamma ** (j - t) * rewards[j] for j in range(t, len(rewards)))
        for t in range(len(rewards))
    ]


def backward_returns(rewards: list[float], gamma: float) -> list[float]:
    result = [0.0] * len(rewards)
    total = 0.0
    for t in reversed(range(len(rewards))):
        total = rewards[t] + gamma * total
        result[t] = total
    return result


def assert_close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} (tol={tolerance})")


def estimator_enumeration() -> dict:
    p, gamma, horizon = 0.35, 0.95, 3
    rows = []
    for actions in itertools.product((0, 1), repeat=horizon):
        rewards = [1.0 + 2 * actions[0], -0.5 + 3 * actions[1], 2.0 - actions[2]]
        returns = return_to_go(rewards, gamma)
        probability = math.prod(p if action else 1 - p for action in actions)
        scores = [action - p for action in actions]
        gradient = sum(gamma ** t * scores[t] * returns[t] for t in range(horizon))
        rows.append({
            "actions": list(actions), "rewards": rewards, "returns": returns,
            "probability": probability, "gradient": gradient,
            "baseline_score": sum(gamma ** t * scores[t] for t in range(horizon)),
            "gradient_without_time_weights": sum(scores[t] * returns[t] for t in range(horizon)),
            "gradient_with_action_dependent_baseline": sum(
                gamma ** t * scores[t] * (returns[t] - rewards[t]) for t in range(horizon)
            ),
        })
    exact_gradient = p * (1 - p) * (2.0 + gamma * 3.0 - gamma ** 2)
    expectation = sum(row["probability"] * row["gradient"] for row in rows)
    assert_close(expectation, exact_gradient)
    assert_close(sum(row["probability"] for row in rows), 1.0)
    baseline_results = []
    for baseline in (0.0, 2.0, 8.0):
        estimates = [(row["probability"], row["gradient"] - baseline * row["baseline_score"])
                     for row in rows]
        mean = sum(probability * estimate for probability, estimate in estimates)
        variance = sum(probability * (estimate - mean) ** 2 for probability, estimate in estimates)
        assert_close(mean, exact_gradient)
        baseline_results.append({"constant_baseline": baseline, "expected_gradient": mean,
                                 "gradient_estimator_variance": variance})
    unweighted = sum(row["probability"] * row["gradient_without_time_weights"] for row in rows)
    action_dependent = sum(row["probability"] * row["gradient_with_action_dependent_baseline"] for row in rows)
    assert abs(unweighted - exact_gradient) > 1e-3
    assert abs(action_dependent - exact_gradient) > 1e-3
    reduction = 1 - baseline_results[1]["gradient_estimator_variance"] / baseline_results[0]["gradient_estimator_variance"]
    return {
        "purpose": "Verify the gamma^t return-to-go estimator, action-independent baseline unbiasedness, and variance claims.",
        "theoretical_expectation": "For p=sigmoid(theta), dJ/dtheta=p(1-p)(2+3*gamma-gamma^2). Every fixed scalar baseline preserves it.",
        "method": "Exhaustively enumerate all 2^3 trajectories, with their exact probabilities; compute population moments (no Monte Carlo error).",
        "p": p, "gamma": gamma, "exact_gradient": exact_gradient,
        "enumerated_expected_gradient": expectation, "trajectory_count": len(rows),
        "constant_baselines": baseline_results,
        "variance_reduction_fraction_at_b2": reduction,
        "missing_gamma_t_expected_gradient": unweighted,
        "missing_gamma_t_bias": unweighted - exact_gradient,
        "action_dependent_baseline_expected_gradient": action_dependent,
        "action_dependent_baseline_bias": action_dependent - exact_gradient,
        "interpretation": "b=2 reduces gradient variance; b=8 increases it. Neither changes the expected gradient. Omitting gamma^t changes the declared discounted objective; a baseline depending on the current action is biased.",
        "trajectories": rows,
    }


def same_batch_mean_bias() -> dict:
    p, batch_size = 0.35, 4
    exact = 2 * p * (1 - p)
    mean = 0.0
    for actions in itertools.product((0, 1), repeat=batch_size):
        probability = math.prod(p if a else 1 - p for a in actions)
        rewards = [1 + 2 * a for a in actions]
        baseline = sum(rewards) / batch_size
        estimate = sum((a - p) * (r - baseline) for a, r in zip(actions, rewards)) / batch_size
        mean += probability * estimate
    expected = (1 - 1 / batch_size) * exact
    assert_close(mean, expected)
    return {
        "purpose": "Check why a current-batch return mean is not the implemented independent prior-batch baseline.",
        "theoretical_expectation": "For N iid one-step trajectories, including each trajectory in its own sample-mean baseline multiplies the expected gradient by (1-1/N).",
        "method": "Exact enumeration of all 16 Bernoulli action batches with N=4.",
        "unbiased_expected_gradient": exact, "current_batch_mean_expected_gradient": mean,
        "expected_ratio": 0.75, "actual_ratio": mean / exact,
        "interpretation": "A mean including the current action introduces finite-batch bias. An EMA frozen from completed earlier batches avoids this self-dependence.",
    }


def returns_checks() -> dict:
    rewards, gamma = [1.0, 2.0, 3.0], 0.95
    expected = [5.6075, 4.85, 3.0]
    computed = backward_returns(rewards, gamma)
    error = max(abs(a - b) for a, b in zip(computed, expected))
    assert error < 1e-12
    assert backward_returns(rewards, 0.0) == rewards
    assert backward_returns(rewards, 1.0) == [6.0, 5.0, 3.0]
    assert backward_returns([], 0.95) == []
    assert backward_returns([7.0], 0.95) == [7.0]
    return {
        "purpose": "Check return indexing, terminal handling, and gamma boundary cases.",
        "theoretical_expectation": "For rewards [1,2,3] and gamma=.95, G=[5.6075,4.85,3]; gamma=0 returns immediate rewards, gamma=1 suffix sums, terminal continuation=0.",
        "method": "Compare a backward recurrence to direct weighted sums and hand values; test empty and single-step episodes.",
        "rewards": rewards, "gamma": gamma, "expected_returns": expected,
        "actual_returns": computed, "absolute_max_error": error,
        "time_weights": [gamma ** t for t in range(3)],
        "interpretation": "The final reward is included once; the objective time weights gamma^t are separate from RTG's internal discounts.",
    }


def autograd_check(enumeration: dict) -> dict:
    import torch
    theta = torch.tensor(math.log(0.35 / 0.65), dtype=torch.float64, requires_grad=True)
    baseline = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(theta)
    loss = torch.zeros((), dtype=torch.float64)
    for row in enumeration["trajectories"]:
        log_probs = [torch.log(p) if a else torch.log1p(-p) for a in row["actions"]]
        trajectory_loss = -sum(
            enumeration["gamma"] ** t * log_probs[t] * (row["returns"][t] - baseline.detach())
            for t in range(3)
        )
        # Rollout sampling probabilities are frozen data, not differentiable factors.
        loss = loss + row["probability"] * trajectory_loss
    loss.backward()
    gradient = float(theta.grad)
    assert_close(gradient, -enumeration["exact_gradient"])
    assert baseline.grad is None
    old_theta = float(theta.detach())
    new_theta = old_theta - 0.01 * gradient
    def objective(value: float) -> float:
        prob = 1 / (1 + math.exp(-value))
        return (1 + 2 * prob) + 0.95 * (-0.5 + 3 * prob) + 0.95 ** 2 * (2 - prob)
    before, after = objective(old_theta), objective(new_theta)
    finite_difference = (objective(old_theta + 1e-6) - objective(old_theta - 1e-6)) / 2e-6
    assert_close(finite_difference, enumeration["exact_gradient"], 1e-8)
    assert after > before
    return {
        "purpose": "Verify the sign of minimizing policy loss, baseline detach, and an actual autograd computation.",
        "theoretical_expectation": "loss gradient is -dJ/dtheta; minimizing it increases J; detached baseline receives no gradient.",
        "method": "Float64 PyTorch exact-probability weighted policy loss; central finite difference of analytic J; one SGD step of size .01.",
        "loss_gradient": gradient, "finite_difference_objective_gradient": finite_difference,
        "baseline_gradient_is_none": baseline.grad is None,
        "objective_before": before, "objective_after": after,
        "objective_increase": after - before,
        "interpretation": "The negative sign implements reward maximization. This local SGD check isolates estimator mathematics; production uses Adam with the same loss sign.",
    }


def cartpole_trace(seed: int) -> dict:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rng = random.Random(seed + 1)
    observation, _ = env.reset(seed=seed)
    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.random() >= 0.5)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        rows.append({"step": len(rows), "observation": [float(x) for x in observation],
                     "action": action, "reward": float(reward),
                     "next_observation": [float(x) for x in next_observation],
                     "terminated": bool(terminated), "truncated": bool(truncated)})
        observation = next_observation
    env.close()
    rewards = [row["reward"] for row in rows]
    returns = backward_returns(rewards, 0.95)
    reference = return_to_go(rewards, 0.95)
    max_error = max(abs(a - b) for a, b in zip(returns, reference))
    assert max_error < 1e-10
    for row, value in zip(rows, returns):
        row["return_to_go"] = value
        row["gamma_t"] = 0.95 ** row["step"]
        row["policy_action_probability"] = 0.5
    assert len(rows) >= 10, "Choose a deterministic trace seed with at least 10 steps."
    return {
        "purpose": "Trace actual observations, actions, rewards and RTG on an unmodified complete CartPole episode.",
        "theoretical_expectation": "CartPole rewards are +1 at every observed step, so G_t=(1-.95^(L-t))/.05 through the episode's final reward.",
        "method": "Gymnasium CartPole-v1, uniform iid actions, fixed reset/action seeds, natural termination or built-in TimeLimit; gradients disabled by using a pure Python policy.",
        "environment": "CartPole-v1", "seed": seed, "action_seed": seed + 1,
        "episode_length": len(rows), "episode_reward": sum(rewards),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "backward_vs_direct_max_absolute_error": max_error,
        "geometric_formula_max_absolute_error": max(abs(returns[t] - (1 - .95 ** (len(rows) - t)) / .05) for t in range(len(rows))),
        "first_ten_steps": rows[:10], "full_episode": rows,
        "interpretation": "This is a reproducible random-policy diagnostic trace, not a trained model result and not part of the 500-episode final evaluations.",
    }


def trainer_regressions() -> dict:
    """Cross-check the actual trainer against independent small-case results."""
    import numpy as np
    import torch
    from . import rl

    return_errors = []
    for gamma in (0.0, 0.95, 0.99, 1.0):
        for rewards in ([], [7.0], [1.0, -0.25, 0.0, 3.0, -1.0]):
            actual = rl.returns_to_go(rewards, gamma)
            expected = return_to_go(rewards, gamma)
            return_errors.extend(abs(float(a) - b) for a, b in zip(actual, expected))
    max_return_error = max(return_errors)
    assert max_return_error < 1e-12

    class ScalarPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, x):
            return torch.stack([torch.zeros_like(self.theta), self.theta]).repeat(len(x), 1)

    traces = []
    for actions in itertools.product((0, 1), repeat=3):
        traces.append({"observations": [np.zeros(4, np.float32)] * 3,
                       "actions": list(actions),
                       "rewards": [1 + 2 * actions[0], -.5 + 3 * actions[1], 2 - actions[2]]})
    expected_gradient = .25 * (2 + .95 * 3 - .95 ** 2)
    results = []
    for baseline in (0.0, 2.0):
        model = ScalarPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        cfg = rl.config("cartpole", bool(baseline), 987, 8, .01, "cpu", 8)
        cfg["chunk_size"] = 2
        metrics = rl.update(model, optimizer, traces, cfg, "cpu", baseline)
        actual_theta = float(model.theta.detach())
        expected_theta = .01 * expected_gradient / cfg["loss_scale"]
        assert_close(actual_theta, expected_theta, 1e-10)
        weighted_g = sum(sum(.95 ** t * g for t, g in enumerate(return_to_go(tr["rewards"], .95))) for tr in traces)
        weight_sum = len(traces) * sum(.95 ** t for t in range(3))
        expected_next = .9 * baseline + .1 * weighted_g / weight_sum if baseline else 0.0
        assert_close(metrics["baseline_next"], expected_next)
        assert_close(metrics["baseline_used"], baseline)
        results.append({"baseline": baseline, "actual_theta_after_update": actual_theta,
                        "expected_theta_after_update": expected_theta,
                        "baseline_used": metrics["baseline_used"],
                        "baseline_next": metrics["baseline_next"],
                        "expected_baseline_next": expected_next})

    # Compare the actual chunked trainer to an independent single-graph expression.
    torch.manual_seed(912)
    model = rl.Policy("cartpole", hidden=7)
    reference = copy.deepcopy(model)
    rng = np.random.default_rng(912)
    trajectories = []
    for length in (5, 7, 9):
        trajectories.append({"observations": rng.normal(size=(length, 4)).astype(np.float32),
                             "actions": rng.integers(0, 2, length).tolist(),
                             "rewards": rng.normal(size=length).tolist()})
    cfg = rl.config("cartpole", True, 912, 3, .01, "cpu", 3)
    cfg.update(chunk_size=1, grad_clip=10.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.01)
    actual_metrics = rl.update(model, optimizer, trajectories, cfg, "cpu", .7)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = torch.zeros(())
    for tr in trajectories:
        observations = torch.from_numpy(tr["observations"])
        actions = torch.tensor(tr["actions"])
        returns = return_to_go(tr["rewards"], .95)
        advantage = torch.tensor([.95 ** t * (value - .7) / 100 for t, value in enumerate(returns)], dtype=torch.float32)
        log_probs = reference(observations).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        reference_loss = reference_loss - (log_probs * advantage).sum() / len(trajectories)
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg["grad_clip"])
    reference_optimizer.step()
    chunk_parameter_error = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(model.parameters(), reference.parameters()))
    chunk_loss_error = abs(actual_metrics["loss"] - float(reference_loss.detach()))
    assert chunk_parameter_error < 5e-7
    assert chunk_loss_error < 1e-7

    # Preprocessing must match the PDF, including signed image differences.
    raw = np.zeros((210, 160, 3), np.uint8)
    raw[35, 0, 0], raw[37, 0, 0], raw[39, 0, 0], raw[41, 0, 0] = 144, 109, 255, 42
    image = rl.preprocess(raw)
    assert image.shape == (6400,) and image.dtype == np.int8
    assert image[:4 * 80:80].tolist() == [0, 0, 1, 1]
    difference = image - np.ones_like(image)
    assert int(difference.min()) == -1 and int(difference.max()) == 0
    no_alias_original = raw.copy()
    rl.preprocess(raw)
    assert np.array_equal(raw, no_alias_original)

    # Actual complete rollouts, on-policy recomputation, and serialized resume.
    torch.manual_seed(924)
    uninterrupted = rl.Policy("cartpole", hidden=7)
    uninterrupted_optimizer = torch.optim.Adam(uninterrupted.parameters(), lr=.01)
    cfg = rl.config("cartpole", True, 924, 8, .01, "cpu", 4)
    cfg["hidden"] = 7
    first_batch = rl.collect(uninterrupted, "cartpole", "cpu", list(range(4)), 73400000)
    first_metrics = rl.update(uninterrupted, uninterrupted_optimizer, first_batch, cfg, "cpu", 0.)
    with tempfile.TemporaryDirectory(prefix="rl-resume-check-") as folder:
        checkpoint = Path(folder) / "checkpoint.pt"
        rl.save_checkpoint(checkpoint, {"model": uninterrupted.state_dict(),
                                       "optimizer": uninterrupted_optimizer.state_dict(),
                                       "baseline": first_metrics["baseline_next"]})
        saved = torch.load(checkpoint, weights_only=False)
    resumed = rl.Policy("cartpole", hidden=7)
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    second_batch = rl.collect(uninterrupted, "cartpole", "cpu", list(range(4, 8)), 73400000)
    replayed_batch = rl.collect(resumed, "cartpole", "cpu", list(range(4, 8)), 73400000)
    trajectory_equal = all(a["actions"] == b["actions"] and a["rewards"] == b["rewards"] and
                           all(np.array_equal(x, y) for x, y in zip(a["observations"], b["observations"]))
                           for a, b in zip(second_batch, replayed_batch))
    assert trajectory_equal
    second_metrics = rl.update(uninterrupted, uninterrupted_optimizer, second_batch, cfg, "cpu", first_metrics["baseline_next"])
    resumed_metrics = rl.update(resumed, resumed_optimizer, replayed_batch, cfg, "cpu", saved["baseline"])
    resume_parameter_error = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(uninterrupted.parameters(), resumed.parameters()))
    assert resume_parameter_error == 0.0
    assert second_metrics["baseline_next"] == resumed_metrics["baseline_next"]

    return {
        "purpose": "Check the production return, policy update, baseline timing, chunking, image preprocessing, and checkpoint resume against independent references.",
        "theoretical_expectation": "Exact complete trajectory enumeration at p=.5 gives dJ/dtheta=.986875; /100 loss and SGD .01 gives theta=.0000986875 for b=0 and b=2. Chunking and serialization must preserve the update.",
        "method": "Import src.rl; run real update() on all eight trajectories; independently calculate a single-graph Adam update; verify synthetic image pixels; save/load model+Adam+EMA between actual CartPole batches with deterministic episode seeds.",
        "production_returns_max_absolute_error": max_return_error,
        "enumerated_production_update": results,
        "chunk1_vs_independent_single_graph_parameter_max_absolute_error": chunk_parameter_error,
        "chunk1_vs_independent_single_graph_loss_absolute_error": chunk_loss_error,
        "preprocessing_test_pixels": image[:4 * 80:80].tolist(),
        "signed_image_difference_range": [int(difference.min()), int(difference.max())],
        "preprocessing_does_not_mutate_raw_observation": True,
        "resume_rollouts_identical": trajectory_equal,
        "resume_parameter_max_absolute_error": resume_parameter_error,
        "resume_baseline_exactly_equal": second_metrics["baseline_next"] == resumed_metrics["baseline_next"],
        "actual_rollout_episode_lengths": [len(tr["rewards"]) for tr in first_batch + second_batch],
        "resume_environment_seed_base": 73400000,
        "interpretation": "Production update agrees with analytic math including loss scaling, chunked recomputation, and a baseline used only from earlier batches. Actual CPU rollouts and the next update are bitwise identical after the checkpoint roundtrip. This validates update-boundary resume, not bitwise CPU/MPS equivalence or interruption inside an episode.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/validation/rl.json"))
    parser.add_argument("--trace-seed", type=int, default=74219)
    args = parser.parse_args()
    started = time.perf_counter()
    enumeration = estimator_enumeration()
    result = {
        "status": "passed", "python": sys.version, "platform": platform.platform(),
        "returns": returns_checks(), "exact_gradient_and_baseline": enumeration,
        "same_batch_mean_bias": same_batch_mean_bias(),
        "autograd": autograd_check(enumeration),
        "cartpole_trace": cartpole_trace(args.trace_seed),
        "production_trainer": trainer_regressions(),
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "exact_gradient": enumeration["exact_gradient"],
                      "baseline_variance_reduction_percent": 100 * enumeration["variance_reduction_fraction_at_b2"],
                      "trace_length": result["cartpole_trace"]["episode_length"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
