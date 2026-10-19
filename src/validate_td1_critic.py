"""Validate critic-only one-step TD change; no formal experiment is modified.

The actor still uses normalized GAE(lambda=.95). ``td1`` names a one-step TD
critic label, usually called TD(0), rather than TD(lambda=1).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import ppo
from .validate_ppo import nested_equal, max_parameter_error


def target_checks():
    rewards = np.array([1., 2., 3., 4.])
    next_values = np.array([.8, 1.2, 2., 5.])
    terminated = np.array([False, True, False, True])
    expected = np.array([1.76, 2., 4.9, 4.])
    actual = ppo.td1_targets(rewards, next_values, terminated, .95)
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-12:
        raise AssertionError(f"One-step targets differ from hand values: {actual}")
    retained = actual.copy()
    next_values[:] = 100.
    if not np.array_equal(actual, retained):
        raise AssertionError("Frozen targets share a mutable view of next values")
    cfg = ppo.default_config(seed=764, total_steps=4, critic_target="td1")
    cfg.update(hidden=7, num_envs=1)
    actor, _, _, _ = ppo.make_models(cfg)

    class ConstantCritic(torch.nn.Module):
        def forward(self, observations):
            return torch.full((len(observations),), 2.)

    critic = ConstantCritic()
    rows = []
    for name, physical, elapsed, expected_target, expected_term, expected_trunc in [
        ("natural_terminal", [2.39, 10., 0., 0.], 0, 1., True, False),
        ("time_limit_only", [0., 0., 0., 0.], 499, 2.9, False, True),
        ("terminal_and_time_limit", [2.39, 10., 0., 0.], 499, 1., True, True),
        ("ongoing_rollout_slice", [0., 0., 0., 0.], 0, 2.9, False, False),
    ]:
        collector = ppo.Collector(cfg)
        try:
            collector.envs[0].unwrapped.state = np.asarray(physical, np.float64)
            collector.envs[0]._elapsed_steps = elapsed
            collector.lanes[0]["obs"] = np.asarray(physical, np.float32)
            data, episodes, _ = collector.rollout(actor, critic, 1, update=0)
            actual_target = float(data["value_target"][0])
            if abs(actual_target - expected_target) > 2e-7:
                raise AssertionError(f"Actual collector target wrong at {name}: {actual_target}")
            if expected_term or expected_trunc:
                if len(episodes) != 1 or episodes[0]["terminated"] != expected_term or episodes[0]["truncated"] != expected_trunc:
                    raise AssertionError(f"Boundary masks did not match {name}")
            elif episodes:
                raise AssertionError("An ongoing rollout slice was counted as an episode")
            if data["value_target"].requires_grad:
                raise AssertionError("Collector retained a graph in critic labels")
            rows.append(dict(case=name, expected_target=expected_target, actual_target=actual_target,
                             terminated=expected_term, truncated=expected_trunc,
                             complete_episode_records=len(episodes)))
        finally:
            collector.close()
    return dict(purpose="Verify one-step critic targets and distinct physical-terminal, TimeLimit and slice handling.",
                theoretical_expectation="Target is r+.95*(1-terminated)*V_old(next). Truncation alone and an ongoing slice retain bootstrap; true terminal wins when both flags are set.",
                method="Independent hand targets plus actual CartPole collector transitions at forced physical/500-step boundaries, with constant old critic value2.",
                hand_expected_targets=expected.tolist(), hand_actual_targets=retained.tolist(),
                hand_max_absolute_error=error, targets_do_not_alias_next_values=True, actual_collector_cases=rows,
                explanation="A truncated boundary is not treated as a natural physical failure under this configured continuing-value bootstrap convention. One-step labels are numerical frozen data, while actor GAE is unchanged.")


def matched_first_update():
    gae_cfg = ppo.default_config(seed=764, total_steps=64)
    td_cfg = ppo.default_config(seed=764, total_steps=64, critic_target="td1")
    for cfg in (gae_cfg, td_cfg):
        cfg.update(hidden=7, num_envs=3, rollout_steps=64, epochs=2, minibatch_size=16)
    config_differences = sorted(key for key in gae_cfg.keys() | td_cfg.keys() if gae_cfg.get(key) != td_cfg.get(key))
    if config_differences != ["critic_target", "value_target"]:
        raise AssertionError(f"Comparison changes more than the critic target: {config_differences}")
    ga, gv, gao, gvo = ppo.make_models(gae_cfg)
    ta, tv, tao, tvo = ppo.make_models(td_cfg)
    initial_actor_error = max_parameter_error(ga, ta)
    initial_critic_error = max_parameter_error(gv, tv)
    if initial_actor_error or initial_critic_error:
        raise AssertionError("Matched-arm starting networks differ")
    gc, tc = ppo.Collector(gae_cfg), ppo.Collector(td_cfg)
    try:
        # Capture next observations/rewards only for the transient TD validation
        # collector, independently calculating its exact frozen labels below.
        history = [[] for _ in tc.envs]
        for lane_index, env in enumerate(tc.envs):
            original_step = env.step

            def record_step(action, original=original_step, index=lane_index):
                observation, reward, term, trunc, info = original(action)
                history[index].append((observation.copy(), float(reward), bool(term), bool(trunc)))
                return observation, reward, term, trunc, info

            env.step = record_step
        gd, ge, _ = gc.rollout(ga, gv, 64, update=0)
        td, te, _ = tc.rollout(ta, tv, 64, update=0)
        fixed_keys = ["obs", "action", "old_logp", "value", "raw_advantage", "advantage"]
        if not all(torch.equal(gd[key], td[key]) for key in fixed_keys) or ge != te:
            raise AssertionError("Changing critic labels changed the first rollout or actor advantages")
        flattened = [transition for lane in history for transition in lane]
        with torch.inference_mode():
            nv = tv(torch.tensor(np.stack([row[0] for row in flattened]), dtype=torch.float32)).numpy()
        reference_target = np.asarray([reward + .95 * (not term) * float(value)
                                       for (_, reward, term, _), value in zip(flattened, nv)], np.float32)
        target_error = float(np.max(np.abs(td["value_target"].numpy() - reference_target)))
        if target_error > 2e-7:
            raise AssertionError(f"Actual collected target differs from independently calculated TD label: {target_error}")
        target_difference = float((gd["value_target"] - td["value_target"]).abs().max())
        if target_difference == 0:
            raise AssertionError("TD labels did not differ from multistep GAE labels in the chosen diagnostic")
        gae_labels, td_labels = gd["value_target"].clone(), td["value_target"].clone()
        gm = ppo.optimize(ga, gv, gao, gvo, gd, gae_cfg, np.random.default_rng(77))
        tm = ppo.optimize(ta, tv, tao, tvo, td, td_cfg, np.random.default_rng(77))
        actor_error = max_parameter_error(ga, ta)
        critic_difference = max_parameter_error(gv, tv)
        if actor_error != 0 or critic_difference == 0:
            raise AssertionError(f"First update failed actor isolation: actor error={actor_error}, critic difference={critic_difference}")
        if not torch.equal(gd["value_target"], gae_labels) or not torch.equal(td["value_target"], td_labels):
            raise AssertionError("Critic labels changed during repeated epochs")
        return dict(purpose="Verify that the first paired comparison differs solely in critic regression labels.",
                    theoretical_expectation="Matched networks/policy/seed yield identical first observations/actions/GAE. Separate actor and critic networks imply identical actor updates on that first batch, while differing labels alter critic updates.",
                    method="Two actual64-step CPU rollouts with same seed764; independently recompute one-step labels from recorded next observations and frozen critic; optimize twice with identical minibatch RNG.",
                    config_differences=config_differences, actor_gae_lambda=gae_cfg["gae_lambda"],
                    initial_actor_parameter_max_absolute_error=initial_actor_error,
                    initial_critic_parameter_max_absolute_error=initial_critic_error,
                    first_rollout_tensors_exactly_equal=fixed_keys,
                    complete_episode_records_exactly_equal=True,
                    td_target_independent_reference_max_absolute_error=target_error,
                    gae_vs_td_target_max_absolute_difference=target_difference,
                    first_batch_actor_parameter_max_absolute_error=actor_error,
                    first_batch_critic_parameter_max_absolute_difference=critic_difference,
                    labels_frozen_through_all_epochs=True, actor_gradient_steps=gm["actor_gradient_steps"],
                    gae_label_first_batch_mean_optimization_value_loss=gm["value_loss"],
                    td_label_first_batch_mean_optimization_value_loss=tm["value_loss"],
                    explanation="The two recorded value losses use different labels and cannot establish which critic is more accurate. Later actor updates can diverge because each newly learned critic changes subsequent GAE estimates; only the controlled first batch must have identical actor updates.")
    finally:
        gc.close()
        tc.close()


def train_resume_check():
    cfg = ppo.default_config(seed=765, total_steps=143, critic_target="td1")
    cfg.update(hidden=7, num_envs=3, rollout_steps=64, epochs=2, minibatch_size=16)
    with tempfile.TemporaryDirectory(prefix="td-one-step-validation-") as folder:
        full_folder, resumed_folder = Path(folder) / "full", Path(folder) / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            ppo.train(full_folder, cfg)
            ppo.train(resumed_folder, cfg, stop_after_updates=1)
            ppo.train(resumed_folder, cfg)
        full = torch.load(full_folder / "final.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(resumed_folder / "final.pt", map_location="cpu", weights_only=False)
        keys = ["config", "model", "critic", "optimizer", "critic_optimizer", "collector",
                "shuffle_rng_state", "torch_rng_state", "updates", "steps", "episode", "gradient_steps"]
        if not all(nested_equal(full[key], resumed[key]) for key in keys):
            raise AssertionError("One-step TD train() differs after checkpoint resume")

        def rows(path):
            with path.open() as stream:
                return list(csv.DictReader(stream))

        full_rows, resumed_rows = rows(full_folder / "train.csv"), rows(resumed_folder / "train.csv")
        strip_elapsed = lambda records: [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in records]
        if strip_elapsed(full_rows) != strip_elapsed(resumed_rows):
            raise AssertionError("One-step TD resumed episode logs differ from uninterrupted logs")
        episode_ids = [int(row["episode"]) for row in resumed_rows]
        rollout_steps = [int(row["rollout_steps"]) for row in rows(resumed_folder / "updates.csv")]
        if episode_ids != list(range(full["episode"])) or rollout_steps != [64, 64, 15] or full["steps"] != 143:
            raise AssertionError("One-step TD train() lost/duplicated episodes or exceeded its step budget")
        actor_error = max(float((full["model"][key] - resumed["model"][key]).abs().max()) for key in full["model"])
        critic_error = max(float((full["critic"][key] - resumed["critic"][key]).abs().max()) for key in full["critic"])
        return dict(purpose="Verify actual one-step-critic train() configuration and numerical state survive interruption/resume.",
                    theoretical_expectation="Same143-step budget and seed with a pause after64 steps produce exact actor/critic/Adam/collector/RNG state and episode logs.",
                    method="Tiny full versus stopped/resumed production train() executions in automatically deleted temporary directories; exclude elapsed times from comparisons.",
                    critic_target_mode=full["config"]["critic_target"], actor_gae_lambda=full["config"]["gae_lambda"],
                    compared_state_keys=keys, all_state_fields_exactly_equal=True,
                    actor_parameter_max_absolute_error=actor_error, critic_parameter_max_absolute_error=critic_error,
                    episode_logs_equal_excluding_elapsed=True, episode_ids=episode_ids, rollout_steps=rollout_steps,
                    final_environment_steps=full["steps"],
                    explanation="This verifies the one-step flag persists through the real checkpoint lifecycle. It is a small correctness experiment, not a formal training-performance result.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ppo.ROOT / "data/validation/td1_critic.json")
    args = parser.parse_args()
    start = time.perf_counter()
    result = dict(status="passed", command=".venv/bin/python -m src.validate_td1_critic",
                  targets=target_checks(), matched_first_update=matched_first_update(),
                  checkpoint_resume=train_resume_check())
    result["elapsed_seconds"] = time.perf_counter() - start
    ppo.atomic_json(args.output, result)
    print(json.dumps(dict(status=result["status"], output=str(args.output),
          first_batch_actor_error=result["matched_first_update"]["first_batch_actor_parameter_max_absolute_error"],
          first_batch_critic_difference=result["matched_first_update"]["first_batch_critic_parameter_max_absolute_difference"],
          resumed_actor_error=result["checkpoint_resume"]["actor_parameter_max_absolute_error"],
          resumed_critic_error=result["checkpoint_resume"]["critic_parameter_max_absolute_error"],
          elapsed_seconds=result["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()
