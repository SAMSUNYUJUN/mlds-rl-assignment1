"""Independent mathematical and actual CPU resume checks for the PPO extension.

No formal experiment is trained or changed. Tiny transient rollouts, optimizer
updates and checkpoint files are used to validate the implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import ppo


def close(actual, expected, tolerance=1e-12):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if error > tolerance:
        raise AssertionError(f"Maximum error {error} exceeds {tolerance}: {actual} vs {expected}")
    return error


def gae_reference(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Direct finite sums of TD residuals, independent of backward recurrence."""
    result = []
    for t in range(len(rewards)):
        total = 0.0
        for u in range(t, len(rewards)):
            delta = rewards[u] + gamma * next_values[u] * (not terminated[u]) - values[u]
            total += (gamma * lam) ** (u - t) * delta
            if terminated[u] or truncated[u]:
                break
        result.append(total)
    return np.asarray(result)


def gae_checks():
    cases = [
        dict(name="natural_terminal", rewards=[1., 2.], values=[.5, .8], next_values=[.8, 1.2],
             terminated=[False, True], truncated=[False, False], expected=[2.286, 1.2]),
        dict(name="time_limit_truncation", rewards=[1., 2.], values=[.5, .8], next_values=[.8, 1.2],
             terminated=[False, False], truncated=[False, True], expected=[3.2607, 2.34]),
        dict(name="ongoing_rollout_slice", rewards=[1., 2.], values=[.5, .8], next_values=[.8, 1.2],
             terminated=[False, False], truncated=[False, False], expected=[3.2607, 2.34]),
        dict(name="truncation_blocks_next_episode_carry", rewards=[1., 2., 100.], values=[.5, .8, 10.],
             next_values=[.8, 1.2, 20.], terminated=[False, False, False], truncated=[False, True, False],
             expected=[3.2607, 2.34, 109.]),
        dict(name="terminal_blocks_bootstrap_and_next_episode", rewards=[1., 2., 100.], values=[.5, .8, 10.],
             next_values=[.8, 1.2, 20.], terminated=[False, True, False], truncated=[False, False, False],
             expected=[2.286, 1.2, 109.]),
        dict(name="both_terminal_and_truncated", rewards=[2.], values=[.8], next_values=[100.],
             terminated=[True], truncated=[True], expected=[1.2]),
    ]
    results = []
    for case in cases:
        inputs = [case[key] for key in ("rewards", "values", "next_values", "terminated", "truncated")]
        actual, target = ppo.gae(*inputs, gamma=.95, lam=.9)
        reference = gae_reference(*inputs, .95, .9)
        error = max(close(actual, case["expected"]), close(actual, reference), close(target, actual + case["values"]))
        results.append({**case, "actual_advantage": actual.tolist(), "actual_value_target": target.tolist(),
                        "max_absolute_error": error})
    # Lambda=1 at a genuine terminal reduces to Monte Carlo RTG minus old value.
    actual, target = ppo.gae([1., 2., 3.], [.5, .8, .2], [.8, .2, 99.],
                             [False, False, True], [False] * 3, gamma=.95, lam=1.)
    mc_error = close(target, [5.6075, 4.85, 3.])
    # Lambda=0 is one-step TD; gamma=0 ignores every future value and reward.
    a0, _ = ppo.gae([1., 2.], [.5, .8], [.8, 1.2], [False, False], [False, False], .95, 0.)
    close(a0, [1.26, 2.34])
    ag0, _ = ppo.gae([1., 2.], [.5, .8], [.8, 1.2], [False, False], [False, False], 0., .9)
    close(ag0, [.5, 1.2])
    return dict(purpose="Distinguish terminal, TimeLimit and rollout boundaries; prevent reset leakage and wrong bootstrap.",
                theoretical_expectation="Natural terminal suppresses next-state value; TimeLimit uses final-observation value but stops carry across reset; rollout slice bootstraps without fabricating termination.",
                method="Compare production backward GAE to hand numbers and independently summed residuals; vary gamma/lambda and terminal masks.",
                gamma=.95, gae_lambda=.9, cases=results, lambda1_mc_target_max_absolute_error=mc_error,
                explanation="Identical final targets for a two-step truncation and slice are intentional: both bootstrap, but a truncation additionally blocks any later reset episode from entering the same recursion.")


def clipping_checks():
    cases = [(2., .5, -1., -1.), (2., 1., -2., -2.), (2., 1.5, -2.4, 0.),
             (-2., .5, 1.6, 0.), (-2., 1., 2., 2.), (-2., 1.5, 3., 3.)]
    rows = []
    for advantage, ratio, expected_loss, expected_log_gradient in cases:
        new_logp = torch.tensor([np.log(ratio)], dtype=torch.float64, requires_grad=True)
        loss = ppo.clipped_policy_loss(new_logp, torch.zeros(1, dtype=torch.float64),
                                       torch.tensor([advantage], dtype=torch.float64), .2)
        loss.backward()
        actual_loss, actual_gradient = float(loss.detach()), float(new_logp.grad)
        close(actual_loss, expected_loss)
        close(actual_gradient, expected_log_gradient)
        rows.append(dict(advantage=advantage, ratio=ratio, actual_loss=actual_loss,
                         expected_loss=expected_loss, gradient_wrt_new_log_probability=actual_gradient,
                         expected_gradient=expected_log_gradient))
    return dict(purpose="Verify asymmetric clipping for positive and negative advantages.",
                theoretical_expectation="Positive-A improvements saturate above ratio1.2; negative-A improvements saturate below ratio.8. Harmful changes on the opposite side remain penalized.",
                method="Float64 autograd on six hand-computable one-transition surrogates; compare loss and derivative with respect to new log probability.",
                clip_ratio=.2, cases=rows,
                explanation="Clipping is not a universal ratio clamp: the minimum with signed advantage is essential. At ratio1 both signs retain the ordinary score gradient.")


def max_parameter_error(model, reference):
    return max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(model.parameters(), reference.parameters()))


def initialization_check():
    cfg = ppo.default_config(seed=101)
    actor, _, _, _ = ppo.make_models(cfg)
    torch.manual_seed(101)
    original_constructor_actor = ppo.Policy("cartpole", 64)
    error = max_parameter_error(actor, original_constructor_actor)
    if error != 0.0:
        raise AssertionError("PPO actor initialization differs from original REINFORCE seed101 Policy")
    return dict(purpose="Validate matched starting actor weights for the algorithm comparison.",
                theoretical_expectation="Constructing the actor before the critic with torch seed101 and the same 4->64 Tanh->2 architecture reproduces the original REINFORCE actor initialization exactly.",
                method="Compare every production PPO actor parameter against Policy('cartpole',64) constructed immediately after torch.manual_seed(101).",
                actor_parameter_max_absolute_error=error, parameter_count=sum(p.numel() for p in actor.parameters()),
                explanation="This confirms matched constructor initialization, not equal subsequent policy trajectories under different algorithms.")


def optimizer_checks():
    cfg = ppo.default_config(seed=761, total_steps=4)
    cfg.update(hidden=7, epochs=1, minibatch_size=4, grad_clip=1e6,
               normalize_advantage=False, entropy_coef=0.)
    actor, critic, actor_opt, critic_opt = ppo.make_models(cfg)
    reference = copy.deepcopy(actor)
    reference_opt = torch.optim.Adam(reference.parameters(), lr=cfg["actor_lr"], eps=cfg["adam_eps"])
    observations = torch.tensor([[.01, .1, -.02, .2], [-.1, .05, .03, .02],
                                 [.02, -.1, -.01, .03], [.1, -.02, .02, -.1]])
    actions = torch.tensor([0, 1, 1, 0])
    advantage = torch.tensor([1., -2., .5, .1], requires_grad=True)
    targets = torch.tensor([2., 2., 2., 2.], requires_grad=True)
    with torch.no_grad():
        old_logp = actor(observations).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        old_values = critic(observations).clone()
    data = dict(obs=observations, action=actions, old_logp=old_logp,
                value=old_values, advantage=advantage, value_target=targets)
    # Independent vanilla score loss: at the first PPO step r=1 and clipping inactive.
    reference_logp = reference(observations).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    reference_loss = -(reference_logp * advantage.detach()).mean()
    reference_opt.zero_grad(set_to_none=True)
    reference_loss.backward()
    reference_gradient = torch.cat([parameter.grad.detach().flatten() for parameter in reference.parameters()])
    reference_opt.step()
    metrics = ppo.optimize(actor, critic, actor_opt, critic_opt, data, cfg, np.random.default_rng(51))
    actor_error = max_parameter_error(actor, reference)
    if actor_error > 1e-7:
        raise AssertionError(f"First PPO update differs from unclipped score update: {actor_error}")
    if advantage.grad is not None or targets.grad is not None:
        raise AssertionError("Actor advantage or critic target was not detached")
    close(metrics["clip_fraction"], 0.)
    close(metrics["approximate_kl"], 0.)
    initial_mse = float((old_values - targets.detach()).square().mean())
    with torch.no_grad():
        after_values = critic(observations)
        after_mse = float((after_values - targets.detach()).square().mean())
    if not after_mse < initial_mse:
        raise AssertionError("Critic's first regression step moved away from fixed targets")

    # Isolate regression sign with a scalar critic and known derivative.
    class ScalarCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, x):
            return self.value.expand(len(x))

    scalar = ScalarCritic()
    scalar_actor = copy.deepcopy(reference)
    scalar_opt = torch.optim.SGD(scalar.parameters(), lr=.1)
    frozen_actor_opt = torch.optim.SGD(scalar_actor.parameters(), lr=0.)
    scalar_data = {**data, "advantage": torch.zeros(4), "value_target": torch.full((4,), 2.)}
    with torch.no_grad():
        scalar_data["old_logp"] = scalar_actor(observations).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    scalar_metrics = ppo.optimize(scalar_actor, scalar, frozen_actor_opt, scalar_opt,
                                  scalar_data, cfg, np.random.default_rng(52))
    close(float(scalar.value.detach()), .65, 5e-8)
    close(scalar_metrics["value_loss"], .5 * 1.5 ** 2)
    return dict(purpose="Connect PPO's first update and critic regression to independent gradient calculations.",
                theoretical_expectation="At old=new policy, clipped surrogate has the ordinary advantage-weighted score gradient. A critic with V=.5,target2 has d(.5*MSE)/dV=-1.5; SGD .1 moves it to .65.",
                method="One actual production optimize() step versus independent unclipped Adam update on four fixed states; detached-target checks; scalar critic SGD sign check.",
                first_update_actor_parameter_max_absolute_error=actor_error,
                reference_actor_gradient_squared_norm=float(reference_gradient @ reference_gradient),
                initial_clip_fraction=metrics["clip_fraction"], initial_approximate_kl=metrics["approximate_kl"],
                advantage_gradient_is_none=advantage.grad is None, value_target_gradient_is_none=targets.grad is None,
                critic_fixed_target_mse_before=initial_mse, critic_fixed_target_mse_after=after_mse,
                scalar_critic_before=.5, scalar_critic_after=float(scalar.value.detach()),
                scalar_critic_expected_after=.65, scalar_critic_initial_value_loss=scalar_metrics["value_loss"],
                explanation="This tests the first on-policy step before clipping activates; repeated PPO epochs can later activate the clipped branches tested separately.")


def nested_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def resume_checks():
    cfg = ppo.default_config(seed=762, total_steps=143)
    cfg.update(hidden=7, epochs=2, minibatch_size=16, num_envs=3, rollout_steps=64)
    actor, critic, actor_opt, critic_opt = ppo.make_models(cfg)
    collector = ppo.Collector(cfg)
    restored_collector = None
    rng = np.random.default_rng(70762)
    try:
        first, first_episodes, _ = collector.rollout(actor, critic, 64, update=0)
        if len(first["action"]) != 64 or collector.steps != 64:
            raise AssertionError("Nonmultiple-of-lane-count rollout does not have exact requested size")
        if any(value.requires_grad for value in first.values()):
            raise AssertionError("Rollout unexpectedly retained an autograd graph")
        if abs(float(first["advantage"].mean())) > 2e-6 or abs(float(first["advantage"].std(unbiased=False)) - 1) > 2e-6:
            raise AssertionError("PPO advantage normalization does not match configured population SD")
        metrics = ppo.optimize(actor, critic, actor_opt, critic_opt, first, cfg, rng)
        with tempfile.TemporaryDirectory(prefix="ppo-validation-") as folder:
            path = Path(folder) / "checkpoint.pt"
            saved = ppo.snapshot(cfg, actor, critic, actor_opt, critic_opt, collector, rng,
                                 update=1, elapsed=0., gradient_steps=metrics["actor_gradient_steps"])
            ppo.save_checkpoint(path, saved)
            state = torch.load(path, map_location="cpu", weights_only=False)
        restored_actor, restored_critic, restored_actor_opt, restored_critic_opt = ppo.make_models(cfg)
        restored_actor.load_state_dict(state["model"])
        restored_critic.load_state_dict(state["critic"])
        restored_actor_opt.load_state_dict(state["optimizer"])
        restored_critic_opt.load_state_dict(state["critic_optimizer"])
        restored_collector = ppo.Collector(cfg)
        restored_collector.load_state_dict(state["collector"])
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = state["shuffle_rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        # Assert restoration includes the current physical state and remaining time limit.
        if not nested_equal(collector.state_dict(), restored_collector.state_dict()):
            raise AssertionError("Collector physical/RNG/TimeLimit/ongoing-episode state was not restored exactly")
        second, episodes, timing = collector.rollout(actor, critic, 64, update=1)
        restored_second, restored_episodes, restored_timing = restored_collector.rollout(restored_actor, restored_critic, 64, update=1)
        if not nested_equal(second, restored_second) or episodes != restored_episodes:
            raise AssertionError("Actual next rollout differs after checkpoint restore")
        actual_metrics = ppo.optimize(actor, critic, actor_opt, critic_opt, second, cfg, rng)
        restored_metrics = ppo.optimize(restored_actor, restored_critic, restored_actor_opt, restored_critic_opt,
                                        restored_second, cfg, restored_rng)
        actor_error = max_parameter_error(actor, restored_actor)
        critic_error = max_parameter_error(critic, restored_critic)
        if actor_error != 0 or critic_error != 0:
            raise AssertionError("CPU next update after resume is not bitwise identical")
        if not nested_equal(actor_opt.state_dict(), restored_actor_opt.state_dict()) or not nested_equal(critic_opt.state_dict(), restored_critic_opt.state_dict()):
            raise AssertionError("Adam states differ after resumed update")
        if not nested_equal(rng.bit_generator.state, restored_rng.bit_generator.state):
            raise AssertionError("Minibatch shuffle RNG states differ after resumed update")
        # Last partial rollout: 143 total steps = 64 + 64 + 15, not 3*64.
        remaining = cfg["total_steps"] - collector.steps
        last, last_episodes, _ = collector.rollout(actor, critic, remaining, update=2)
        if remaining != 15 or len(last["action"]) != 15 or collector.steps != cfg["total_steps"]:
            raise AssertionError("Final partial rollout overshot the fixed environment-step budget")
        final_metrics = ppo.optimize(actor, critic, actor_opt, critic_opt, last, cfg, rng)
        all_episodes = first_episodes + episodes + last_episodes
        completed_steps = sum(row["length"] for row in all_episodes)
        ongoing_lengths = [lane["length"] for lane in collector.lanes]
        if completed_steps + sum(ongoing_lengths) != collector.steps or len(all_episodes) != collector.completed:
            raise AssertionError("Completed episode records plus ongoing lanes do not account for all actual environment steps")
        for expected_update, episode_rows in enumerate((first_episodes, episodes, last_episodes)):
            if any(row["end_sampling_policy_update"] != expected_update or row["sampling_policy_update"] != expected_update or row["start_update"] > expected_update for row in episode_rows):
                raise AssertionError("Completed episode policy-version metadata is inconsistent")
        # Force a natural terminal exactly at a rollout boundary, then verify the
        # next episode's first action is attributed to the new policy version.
        boundary_cfg = {**cfg, "num_envs": 1}
        boundary_collector = ppo.Collector(boundary_cfg)
        try:
            physical = np.asarray([2.39, 10., 0., 0.], dtype=np.float64)
            boundary_collector.envs[0].unwrapped.state = physical.copy()
            boundary_collector.lanes[0]["obs"] = physical.astype(np.float32)
            _, boundary_episodes, _ = boundary_collector.rollout(actor, critic, 1, update=4)
            if len(boundary_episodes) != 1 or not boundary_episodes[0]["terminated"] or boundary_collector.lanes[0]["length"] != 0:
                raise AssertionError("Forced boundary episode did not naturally terminate and reset")
            boundary_collector.rollout(actor, critic, 1, update=5)
            first_action_update = boundary_collector.lanes[0]["start_update"]
            first_action_step = boundary_collector.lanes[0]["start_env_step"]
            if first_action_update != 5 or first_action_step != 1:
                raise AssertionError("Episode reset on previous rollout end was attributed to the wrong initial sampling policy")
        finally:
            boundary_collector.close()
        return dict(purpose="Verify exact continuation of actual CPU rollouts, Adam updates and incomplete training episodes, and exact final step-budget accounting.",
                    theoretical_expectation="Restoring actor, critic, both optimizers, lane observations/physical states/TimeLimit counts/action RNGs, shuffle RNG and torch RNG reproduces the next rollout and update. Total143 requires rollouts64,64,15.",
                    method="Tiny actual CartPole rollouts; serialize production snapshot() to a temporary file; create a new collector/models/optimizers and restore; compare all next-rollout tensors, completed episode records, parameters and optimizer states.",
                    train_seed_base=cfg["train_seed_base"], num_envs=3,
                    rollout_lengths=[len(first["action"]), len(second["action"]), len(last["action"])],
                    final_environment_steps=collector.steps, requested_environment_steps=cfg["total_steps"],
                    ongoing_episode_lengths_at_first_checkpoint=[lane["length"] for lane in state["collector"]["lanes"]],
                    first_rollout_complete_episodes=len(first_episodes), next_rollout_complete_episodes=len(episodes),
                    final_partial_rollout_complete_episodes=len(last_episodes),
                    all_completed_episode_lengths=[row["length"] for row in all_episodes],
                    total_completed_episode_steps=completed_steps, final_ongoing_episode_lengths=ongoing_lengths,
                    completed_plus_ongoing_steps=completed_steps + sum(ongoing_lengths),
                    reset_at_rollout_boundary_next_episode_first_sampling_update=first_action_update,
                    reset_at_rollout_boundary_next_episode_first_global_step=first_action_step,
                    rollout_tensors_exactly_equal=True, complete_episode_records_exactly_equal=True,
                    restored_collector_state_exactly_equal=True,
                    actor_parameter_max_absolute_error=actor_error, critic_parameter_max_absolute_error=critic_error,
                    both_adam_states_exactly_equal=True, shuffle_rng_exactly_equal=True,
                    final_partial_rollout_actor_gradient_steps=final_metrics["actor_gradient_steps"],
                    advantage_normalized_mean=float(first["advantage"].mean()),
                    advantage_normalized_population_sd=float(first["advantage"].std(unbiased=False)),
                    explanation="Unfinished episodes are preserved across rollouts and checkpoints, not counted as completed episodes. These short validation rollouts are not formal training or held-out performance evaluation.")
    finally:
        collector.close()
        if restored_collector is not None:
            restored_collector.close()


def train_loop_resume_check():
    """Exercise the actual file/log/checkpoint lifecycle in temporary folders."""
    cfg = ppo.default_config(seed=763, total_steps=143)
    cfg.update(hidden=7, epochs=2, minibatch_size=16, num_envs=3, rollout_steps=64)
    with tempfile.TemporaryDirectory(prefix="ppo-train-loop-validation-") as folder:
        full_folder = Path(folder) / "uninterrupted"
        resumed_folder = Path(folder) / "interrupted"
        # Keep the final stdout concise; these are transient validation runs.
        with contextlib.redirect_stdout(io.StringIO()):
            ppo.train(full_folder, cfg)
            ppo.train(resumed_folder, cfg, stop_after_updates=1)
        partial = torch.load(resumed_folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        if partial["updates"] != 1 or partial["steps"] != 64 or (resumed_folder / "FROZEN.json").exists():
            raise AssertionError("Interrupted train() did not stop at the first committed update")

        def read_csv(path):
            with path.open() as stream:
                return list(csv.DictReader(stream))

        # Reproduce a crash window: logs can be ahead of the atomic checkpoint.
        # These are copies of ACTUAL next-update rows from the transient full run.
        full_episode_rows = read_csv(full_folder / "train.csv")
        full_update_rows = read_csv(full_folder / "updates.csv")
        uncommitted_episodes = [row for row in full_episode_rows if int(row["update"]) == 2]
        uncommitted_updates = [row for row in full_update_rows if int(row["update"]) == 2]
        ppo.write_csv(resumed_folder / "train.csv", uncommitted_episodes)
        ppo.write_csv(resumed_folder / "updates.csv", uncommitted_updates)
        with contextlib.redirect_stdout(io.StringIO()):
            ppo.train(resumed_folder, cfg)
        full = torch.load(full_folder / "final.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(resumed_folder / "final.pt", map_location="cpu", weights_only=False)
        compared_keys = ["config", "model", "critic", "optimizer", "critic_optimizer", "collector",
                         "shuffle_rng_state", "torch_rng_state", "updates", "steps", "episode",
                         "baseline", "gradient_steps"]
        differences = [key for key in compared_keys if not nested_equal(full[key], resumed[key])]
        if differences:
            raise AssertionError(f"Actual resumed train() checkpoint differs: {differences}")
        resumed_episode_rows = read_csv(resumed_folder / "train.csv")
        resumed_update_rows = read_csv(resumed_folder / "updates.csv")
        excluded = {"elapsed_seconds"}
        full_core = [{key: value for key, value in row.items() if key not in excluded} for row in full_episode_rows]
        resumed_core = [{key: value for key, value in row.items() if key not in excluded} for row in resumed_episode_rows]
        if full_core != resumed_core:
            raise AssertionError("Actual resumed train() episode reward/seed/length/policy metadata differs")
        episode_ids = [int(row["episode"]) for row in resumed_episode_rows]
        update_ids = [int(row["update"]) for row in resumed_update_rows]
        if episode_ids != list(range(full["episode"])) or update_ids != [1, 2, 3]:
            raise AssertionError("Duplicate or missing episodes/updates after train() log rollback and resume")
        full_rollout_steps = [int(row["rollout_steps"]) for row in full_update_rows]
        resumed_rollout_steps = [int(row["rollout_steps"]) for row in resumed_update_rows]
        if full_rollout_steps != [64, 64, 15] or resumed_rollout_steps != full_rollout_steps:
            raise AssertionError("Actual train loop does not honor the final partial rollout budget")
        parameter_error = max(float((full["model"][key] - resumed["model"][key]).abs().max()) for key in full["model"])
        critic_error = max(float((full["critic"][key] - resumed["critic"][key]).abs().max()) for key in full["critic"])
        return dict(purpose="Verify actual train() interruption/resume and log-commit rollback end to end, beyond manually restoring snapshots.",
                    theoretical_expectation="Uninterrupted143 steps and stopping after update1 then resuming produce identical final numerical state and complete-episode logs; a log tail newer than the committed checkpoint is discarded.",
                    method="Two tiny CPU train() runs in a temporary directory, same seed763 and64-step rollouts across3 lanes; copy actual update2 log rows ahead of the stopped checkpoint to emulate a crash window; resume and compare final state and non-timing episode logs.",
                    interruption_update=partial["updates"], interruption_environment_steps=partial["steps"],
                    injected_uncommitted_episode_rows=len(uncommitted_episodes),
                    injected_uncommitted_update_rows=len(uncommitted_updates),
                    compared_final_state_keys=compared_keys, all_compared_final_state_exactly_equal=True,
                    actor_parameter_max_absolute_error=parameter_error, critic_parameter_max_absolute_error=critic_error,
                    complete_episode_logs_equal_excluding_elapsed=True,
                    complete_episode_ids=episode_ids, complete_episode_rewards=[float(row["reward"]) for row in resumed_episode_rows],
                    complete_episode_lengths=[int(row["length"]) for row in resumed_episode_rows],
                    complete_episode_seeds=[int(row["seed"]) for row in resumed_episode_rows],
                    logged_update_ids=update_ids, rollout_steps=full_rollout_steps,
                    final_environment_steps=full["steps"], final_complete_episode_count=full["episode"],
                    final_ongoing_episode_lengths=[lane["length"] for lane in full["collector"]["lanes"]],
                    no_duplicate_or_missing_episode_records=True, no_duplicate_or_missing_update_records=True,
                    explanation="The log tail is synthetic crash simulation copied from real transient rollout rows. It is excluded from final output and formal experimental records. Temporary runs and checkpoints are deleted automatically; existing formal checkpoints are never opened for writing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ppo.ROOT / "data/validation/ppo.json")
    args = parser.parse_args()
    start = time.perf_counter()
    results = dict(status="passed", command=".venv/bin/python -m src.validate_ppo",
                   initialization=initialization_check(), gae=gae_checks(), clipping=clipping_checks(),
                   first_update_and_critic=optimizer_checks(), checkpoint_resume=resume_checks(),
                   train_loop_resume=train_loop_resume_check())
    results["elapsed_seconds"] = time.perf_counter() - start
    ppo.atomic_json(args.output, results)
    print(json.dumps(dict(status=results["status"], output=str(args.output),
                          gae_cases=len(results["gae"]["cases"]), clip_cases=len(results["clipping"]["cases"]),
                          resumed_actor_error=results["checkpoint_resume"]["actor_parameter_max_absolute_error"],
                          resumed_critic_error=results["checkpoint_resume"]["critic_parameter_max_absolute_error"],
                          rollout_lengths=results["checkpoint_resume"]["rollout_lengths"],
                          elapsed_seconds=results["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()
