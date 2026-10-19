"""Check frozen critics against already recorded initial-state Monte Carlo returns.

Run only after both 500-episode, seed-88000000 evaluation files are complete:
    .venv/bin/python -m src.diagnose_td1_critic

This script resets CartPole to reconstruct each recorded initial observation. It
never calls env.step(), collects another episode, or updates either network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

try:
    from .ppo import Critic
    from .rl import atomic_json
except ImportError:
    from ppo import Critic
    from rl import atomic_json


ROOT = Path(__file__).resolve().parents[1]
N_EPISODES = 500
SEED_BASE = 88000000
RUNS = {"gae": "cartpole_ppo", "td1": "cartpole_ppo_td1_critic"}
FIELDS = ["episode", "seed", "state0", "state1", "state2", "state3",
          "value_prediction", "mc_return", "error", "checkpoint"]


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def truth(value):
    if value is True or str(value).lower() in {"true", "1"}:
        return True
    if value is False or str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid Boolean {value!r}")


def prepare(root, label, run_name, evaluation_root):
    """Validate all saved evidence, then calculate predictions without stepping."""
    source = root / "runs" / run_name
    evaluation = root / evaluation_root / label
    frozen = read_json(source / "FROZEN.json")
    checkpoint = source / "final.pt"
    checksum = digest(checkpoint)
    if frozen.get("file", "final.pt") != "final.pt" or checksum != frozen["sha256"]:
        raise ValueError(f"{label}: final checkpoint does not match its frozen hash")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if cfg != read_json(source / "config.json"):
        raise ValueError(f"{label}: saved checkpoint and run configurations disagree")
    if cfg.get("game") != "cartpole" or cfg.get("environment") != "CartPole-v1":
        raise ValueError(f"{label}: the diagnostic requires CartPole-v1")
    gamma = float(cfg["gamma"])
    if not np.isclose(gamma, .95, rtol=0., atol=1e-15):
        raise ValueError(f"{label}: expected the recorded gamma=.95 protocol")
    if cfg.get("critic_target", "gae") != label:
        raise ValueError(f"{label}: critic target does not match the comparison label")
    critic = Critic(cfg["hidden"])
    critic.load_state_dict(saved["critic"], strict=True)
    critic.eval()
    if any(not torch.isfinite(p).all() for p in critic.parameters()):
        raise ValueError(f"{label}: critic has nonfinite parameters")

    protocol = read_json(evaluation / "evaluation_protocol.json")
    expected_protocol = dict(source_run=f"runs/{run_name}",
                             checkpoint_sha256=checksum, seed_base=SEED_BASE,
                             episodes=N_EPISODES, action_rule="sample categorical policy",
                             complete_episodes_only=True)
    if protocol != expected_protocol:
        raise ValueError(f"{label}: evaluation protocol differs from the frozen comparison")
    csv_path = evaluation / "evaluation.csv"
    with csv_path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != N_EPISODES:
        raise ValueError(f"{label}: need exactly {N_EPISODES} completed evaluations")
    total_steps = 0
    returns = []
    rewards = []
    for episode, row in enumerate(records):
        if int(row["episode"]) != episode or int(row["seed"]) != SEED_BASE + episode:
            raise ValueError(f"{label}: evaluation IDs or seed order do not match")
        if row["checkpoint"] != checksum:
            raise ValueError(f"{label}: an evaluation row uses a different checkpoint")
        terminated, truncated = truth(row["terminated"]), truth(row["truncated"])
        length, reward = int(row["length"]), float(row["reward"])
        mc_return = float(row["discounted_return"])
        if not (terminated or truncated) or not 1 <= length <= 500:
            raise ValueError(f"{label}: evaluation row is not a full CartPole episode")
        if truncated != (length == 500):
            raise ValueError(f"{label}: official 500-step TimeLimit flags disagree with length")
        if not np.isfinite(reward) or not np.isclose(reward, length, rtol=0., atol=1e-12):
            raise ValueError(f"{label}: complete CartPole reward must equal episode length")
        expected_return = (1. - gamma**length) / (1. - gamma)
        if not np.isfinite(mc_return) or not np.isclose(mc_return, expected_return,
                                                      rtol=1e-10, atol=1e-9):
            raise ValueError(f"{label}: Monte Carlo return differs from the geometric sum")
        total_steps += length
        if int(row["env_steps_total"]) != total_steps:
            raise ValueError(f"{label}: evaluation cumulative step counter is inconsistent")
        rewards.append(reward)
        returns.append(mc_return)
    summary = read_json(evaluation / "evaluation_summary.json")
    if (int(summary["n"]) != N_EPISODES or summary["checkpoint_sha256"] != checksum
            or int(summary["eval_seed_base"]) != SEED_BASE
            or int(summary["env_steps"]) != total_steps
            or not np.isclose(summary["mean"], np.mean(rewards))
            or not np.isclose(summary["std_sample"], np.std(rewards, ddof=1))):
        raise ValueError(f"{label}: raw evaluation data and saved summary disagree")

    observations = []
    env = gym.make("CartPole-v1")
    try:
        if env.spec.max_episode_steps != 500:
            raise ValueError("The installed CartPole-v1 TimeLimit is not 500 steps")
        for row in records:
            observation, _ = env.reset(seed=int(row["seed"]))
            if observation.shape != (4,) or not np.isfinite(observation).all():
                raise ValueError("Unexpected CartPole reset observation")
            observations.append(observation.copy())
    finally:
        env.close()
    states = np.asarray(observations, dtype=np.float32)
    with torch.inference_mode():
        predictions = critic(torch.as_tensor(states)).cpu().numpy().astype(np.float64)
    mc_returns = np.asarray(returns, np.float64)
    errors = predictions - mc_returns
    if not np.isfinite(predictions).all():
        raise ValueError(f"{label}: critic produced nonfinite initial predictions")
    rows = [dict(episode=i, seed=int(record["seed"]),
                 **{f"state{j}": float(states[i, j]) for j in range(4)},
                 value_prediction=float(predictions[i]), mc_return=float(mc_returns[i]),
                 error=float(errors[i]), checkpoint=checksum)
            for i, record in enumerate(records)]
    tail_bound = gamma**500 / (1. - gamma)
    result = dict(
        n=N_EPISODES, mean_pred=float(predictions.mean()),
        mean_return=float(mc_returns.mean()), bias=float(errors.mean()),
        rmse=float(np.sqrt(np.mean(errors**2))), mae=float(np.abs(errors).mean()),
        sample_sd_error=float(errors.std(ddof=1)), sample_sd_error_ddof=1,
        error_definition="value_prediction - mc_return",
        critic_target=label, gamma=gamma, environment="CartPole-v1", device="cpu",
        source_run=f"runs/{run_name}", checkpoint_sha256=checksum,
        evaluation_csv=str(csv_path.relative_to(root)), evaluation_csv_sha256=digest(csv_path),
        evaluation_seed_first=SEED_BASE, evaluation_seed_last=SEED_BASE+N_EPISODES-1,
        recorded_evaluation_episodes=N_EPISODES, additional_environment_steps=0,
        additional_completed_episodes=0, reset_calls=N_EPISODES,
        official_time_limit_steps=500,
        initial_state_remaining_discounted_tail_bound=tail_bound,
        notes=[
            "Each prediction is compared with one Monte Carlo return from that critic's own frozen policy; the return includes action-sampling noise.",
            "RMSE and MAE measure prediction-to-realized-return discrepancies, not directly the error against the unknown conditional true value V(s).",
            "The critics share the same ordered initial states, but target returns come from their respective policies; this is not a shared-policy critic ablation.",
            "Predictions use only initial states, avoiding the large target-convention mismatch near the official TimeLimit.",
            f"At an initial state, the possible post-500 continuation is bounded by gamma^500/(1-gamma)={tail_bound:.12g}, since CartPole rewards are at most one per step.",
            "Initial observations are reconstructed with seeded reset only. No env.step call, extra rollout, training update, or new evaluation episode is performed.",
            "This diagnostic complements the target-specific training regression losses; lower training loss across different labels is not itself evidence of better value predictions.",
        ])
    return evaluation, rows, result


def write_result(folder, rows, summary):
    path = folder / "initial_value_diagnostic.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    atomic_json(folder / "initial_value_diagnostic.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=Path("runs/td1_critic_eval"))
    args = parser.parse_args()
    root = args.root.resolve()
    # Complete both input checks/calculations before writing either diagnostic.
    results = [prepare(root, label, run_name, args.evaluation_root)
               for label, run_name in RUNS.items()]
    for folder, rows, summary in results:
        write_result(folder, rows, summary)
        print(json.dumps(dict(model=summary["critic_target"], n=summary["n"],
                              mean_pred=summary["mean_pred"], mean_return=summary["mean_return"],
                              bias=summary["bias"], rmse=summary["rmse"],
                              mae=summary["mae"], sample_sd_error=summary["sample_sd_error"])),
              flush=True)


if __name__ == "__main__":
    main()
