"""Rebuild all four REINFORCE figures and statistics from real episode records.

Default (final) mode requires four frozen runs and exactly 500 complete held-out
evaluation episodes per run.  --available permits unfinished runs for monitoring;
every partial figure and summary remains explicitly labelled as progress.

The training moving average includes the current episode and the previous 99.
For the first 99 episodes, use all episodes available so far (min_periods=1).
Evaluation standard deviation is the sample standard deviation (ddof=1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_NAMES = ("cartpole_basic", "cartpole_baseline", "pong_basic", "pong_baseline")
EXPECTED_EVALUATIONS = 500
REQUIRED_COLUMNS = {
    "episode", "seed", "reward", "discounted_return", "length", "terminated",
    "truncated", "env_steps_total", "elapsed_seconds", "checkpoint",
}


def trailing_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Trailing mean with a growing window until `window` values exist."""
    values = np.asarray(values, dtype=np.float64)
    if window < 1:
        raise ValueError("moving-average window must be positive")
    totals = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (totals[end] - totals[start]) / (end - start)


def read_bool(value: str, context: str) -> bool:
    value = value.strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{context}: invalid boolean {value!r}")
    return value in {"true", "1"}


def read_episodes(path: Path) -> list[dict]:
    """Validate complete, contiguous episode records; never infer missing data."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or []))
            raise ValueError(f"{path}: missing columns {missing}")
        raw = list(reader)
    if not raw:
        raise ValueError(f"{path}: no episode records yet")
    rows = []
    total_steps = 0
    previous_elapsed = -math.inf
    for index, record in enumerate(raw):
        context = f"{path}, row {index + 2}"
        try:
            row = {
                "episode": int(record["episode"]), "seed": int(record["seed"]),
                "reward": float(record["reward"]),
                "discounted_return": float(record["discounted_return"]),
                "length": int(record["length"]),
                "terminated": read_bool(record["terminated"], context),
                "truncated": read_bool(record["truncated"], context),
                "env_steps_total": int(record["env_steps_total"]),
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "checkpoint": record["checkpoint"],
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{context}: malformed episode record: {exc}") from exc
        if row["episode"] != index:
            raise ValueError(f"{context}: expected contiguous episode ID {index}, got {row['episode']}")
        if row["length"] <= 0:
            raise ValueError(f"{context}: episode length must be positive")
        if not (row["terminated"] or row["truncated"]):
            raise ValueError(f"{context}: episode is not complete (neither terminated nor truncated)")
        for key in ("reward", "discounted_return", "elapsed_seconds"):
            if not math.isfinite(row[key]):
                raise ValueError(f"{context}: nonfinite {key}")
        total_steps += row["length"]
        if row["env_steps_total"] != total_steps:
            raise ValueError(f"{context}: cumulative steps do not equal sum of episode lengths")
        if row["elapsed_seconds"] < 0 or row["elapsed_seconds"] < previous_elapsed:
            raise ValueError(f"{context}: elapsed time must be nonnegative and nondecreasing")
        previous_elapsed = row["elapsed_seconds"]
        rows.append(row)
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate per-episode seeds")
    return rows


def check_seed_sequence(rows: list[dict], seed_base: int, label: str) -> None:
    if any(row["seed"] != seed_base + row["episode"] for row in rows):
        raise ValueError(f"{label}: per-episode seeds do not match configured base + episode ID")


def episode_stats(rows: list[dict]) -> dict:
    rewards = np.array([row["reward"] for row in rows], dtype=np.float64)
    lengths = np.array([row["length"] for row in rows], dtype=np.int64)
    return {
        "n": len(rows), "mean": float(rewards.mean()),
        "std_sample": float(rewards.std(ddof=1)) if len(rows) > 1 else None,
        "std_population": float(rewards.std(ddof=0)),
        "min": float(rewards.min()), "max": float(rewards.max()),
        "mean_length": float(lengths.mean()), "min_length": int(lengths.min()),
        "max_length": int(lengths.max()), "env_steps": int(lengths.sum()),
        "elapsed_seconds": rows[-1]["elapsed_seconds"],
        "terminated": sum(row["terminated"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "first_seed": rows[0]["seed"], "last_seed": rows[-1]["seed"],
    }


def frozen_checkpoint(folder: Path) -> dict | None:
    path = folder / "FROZEN.json"
    if not path.exists():
        return None
    frozen = json.loads(path.read_text())
    checkpoint = folder / frozen["file"]
    if not checkpoint.is_file():
        raise ValueError(f"{folder}: frozen checkpoint file is missing")
    with checkpoint.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != frozen["sha256"]:
        raise ValueError(f"{checkpoint}: frozen checkpoint SHA-256 mismatch")
    return frozen


def optimizer_update_summary(folder: Path, cfg: dict, training_episodes: int,
                             available: bool) -> dict | None:
    """Audit optional pre-clipping update diagnostics from a stable CSV snapshot.

    Progress reads ignore an unfinished last line or updates newer than the
    training snapshot. Final reads require the entire formal update budget.
    """
    path = folder / "updates.csv"
    if not path.exists():
        return None
    snapshot = path.read_text()
    ignored_partial_line = bool(snapshot and not snapshot.endswith("\n"))
    if ignored_partial_line:
        if not available:
            raise ValueError(f"{path}: unfinished final CSV row")
        snapshot = snapshot.rsplit("\n", 1)[0] + "\n" if "\n" in snapshot else ""
    reader = csv.DictReader(snapshot.splitlines())
    required = {"update", "episode", "loss", "gradient_norm", "baseline_used", "baseline_next", "update_seconds"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"{path}: missing optimizer update columns")
    rows = []
    newer_records_ignored = 0
    budget, batch = int(cfg["episodes"]), int(cfg["batch_size"])
    for index, record in enumerate(reader, start=1):
        row = {"update": int(record["update"]), "episode": int(record["episode"])}
        row.update({key: float(record[key]) for key in required - {"update", "episode"}})
        if row["update"] != index or row["episode"] != min(index * batch, budget):
            raise ValueError(f"{path}: noncontiguous update IDs or incorrect cumulative episode at update {index}")
        if not all(math.isfinite(row[key]) for key in required - {"update", "episode"}):
            raise ValueError(f"{path}: nonfinite optimizer metric at update {index}")
        if row["gradient_norm"] < 0 or row["update_seconds"] < 0:
            raise ValueError(f"{path}: negative gradient norm or update time at update {index}")
        if available and row["episode"] > training_episodes:
            newer_records_ignored += 1
            continue
        rows.append(row)
    expected_updates = (budget + batch - 1) // batch
    if not available and (len(rows) != expected_updates or not rows or rows[-1]["episode"] != training_episodes):
        raise ValueError(f"{path}: optimizer records do not reach the final formal training budget")
    if not rows:
        return {"n": 0, "expected_updates": expected_updates, "complete": False,
                "source": str(path), "ignored_unfinished_last_line": ignored_partial_line,
                "newer_records_ignored_for_consistent_snapshot": newer_records_ignored}
    norms = np.array([row["gradient_norm"] for row in rows], dtype=np.float64)
    losses = np.array([row["loss"] for row in rows], dtype=np.float64)
    seconds = np.array([row["update_seconds"] for row in rows], dtype=np.float64)
    threshold = float(cfg["grad_clip"])
    clipped = int(np.count_nonzero(norms > threshold))
    return {
        "n": len(rows), "first_update": rows[0]["update"], "last_update": rows[-1]["update"],
        "expected_updates": expected_updates, "final_episode": rows[-1]["episode"],
        "complete": len(rows) == expected_updates and rows[-1]["episode"] == budget,
        "snapshot_matches_training_episode_count": rows[-1]["episode"] == training_episodes,
        "gradient_norm_definition": "Global parameter L2 norm before clip_grad_norm_; one record per optimizer update",
        "max_gradient_norm": float(norms.max()), "min_gradient_norm": float(norms.min()),
        "mean_gradient_norm": float(norms.mean()), "grad_clip_threshold": threshold,
        "updates_above_grad_clip": clipped, "fraction_above_grad_clip": clipped / len(rows),
        "loss_min": float(losses.min()), "loss_max": float(losses.max()),
        "last_baseline_used": rows[-1]["baseline_used"], "last_baseline_next": rows[-1]["baseline_next"],
        "update_seconds_total": float(seconds.sum()), "update_seconds_mean": float(seconds.mean()),
        "source": str(path), "ignored_unfinished_last_line": ignored_partial_line,
        "newer_records_ignored_for_consistent_snapshot": newer_records_ignored,
    }


def load_run(folder: Path, name: str, available: bool) -> tuple[dict, list[dict], list[dict]]:
    cfg = json.loads((folder / "config.json").read_text())
    expected_game = name.split("_")[0]
    expected_baseline = name.endswith("baseline")
    if cfg["game"] != expected_game or bool(cfg["baseline"]) != expected_baseline:
        raise ValueError(f"{folder}: configuration does not match canonical run name {name}")
    train_rows = read_episodes(folder / "train.csv")
    check_seed_sequence(train_rows, int(cfg["train_seed_base"]), f"{name} training")
    if len(train_rows) > int(cfg["episodes"]):
        raise ValueError(f"{name}: training records exceed the configured episode budget")
    frozen = frozen_checkpoint(folder)
    if frozen and (int(frozen["episode"]) != len(train_rows) or len(train_rows) != int(cfg["episodes"])):
        raise ValueError(f"{name}: frozen episode, training records and configured budget disagree")
    if not available and frozen is None:
        raise ValueError(f"{name}: no frozen final checkpoint")

    train_stats = episode_stats(train_rows)
    last = np.array([row["reward"] for row in train_rows[-100:]], dtype=np.float64)
    train_stats.update({
        "episodes": len(train_rows), "configured_episode_budget": int(cfg["episodes"]),
        "complete": len(train_rows) == int(cfg["episodes"]) and frozen is not None,
        "last100_episodes": len(last), "last100_mean_reward": float(last.mean()),
        "last100_std_sample": float(last.std(ddof=1)) if len(last) > 1 else None,
        "moving_average": {"window": 100, "min_periods": 1, "includes_current_episode": True},
        "checkpoint_sha256": frozen["sha256"] if frozen else None,
    })
    eval_path = folder / "evaluation.csv"
    eval_rows = read_episodes(eval_path) if eval_path.exists() else []
    eval_stats = None
    if eval_rows:
        if frozen is None:
            raise ValueError(f"{name}: evaluation records exist without a frozen checkpoint")
        if len(eval_rows) > EXPECTED_EVALUATIONS:
            raise ValueError(f"{name}: more than {EXPECTED_EVALUATIONS} evaluation records")
        if not available and len(eval_rows) != EXPECTED_EVALUATIONS:
            raise ValueError(f"{name}: expected {EXPECTED_EVALUATIONS} evaluation episodes, found {len(eval_rows)}")
        check_seed_sequence(eval_rows, int(cfg["eval_seed_base"]), f"{name} evaluation")
        train_seeds = {row["seed"] for row in train_rows}
        eval_seeds = {row["seed"] for row in eval_rows}
        if train_seeds.intersection(eval_seeds):
            raise ValueError(f"{name}: evaluation seeds overlap with training seeds")
        # Check against the complete configured training range, even in progress mode.
        train_start = int(cfg["train_seed_base"])
        train_stop = train_start + int(cfg["episodes"])
        if any(train_start <= seed < train_stop for seed in eval_seeds):
            raise ValueError(f"{name}: evaluation seed lies in configured training-seed range")
        if any(row["checkpoint"] != frozen["sha256"] for row in eval_rows):
            raise ValueError(f"{name}: evaluation does not consistently use the frozen checkpoint")
        eval_stats = episode_stats(eval_rows)
        eval_stats.update({"complete": len(eval_rows) == EXPECTED_EVALUATIONS,
                           "required_episodes": EXPECTED_EVALUATIONS,
                           "checkpoint_sha256": frozen["sha256"],
                           "sampling": cfg["action_sampling"], "std_convention": "sample standard deviation, ddof=1",
                           "seed_overlap_with_training": 0,
                           "all_records_complete": True,
                           "checkpoint_matches_frozen_file": True})
    elif not available:
        raise ValueError(f"{name}: missing evaluation.csv with 500 complete episodes")
    summary = {
        "run": name, "game": cfg["game"], "baseline": bool(cfg["baseline"]),
        "config": cfg, "training": train_stats, "evaluation": eval_stats,
        "optimizer_updates": optimizer_update_summary(folder, cfg, len(train_rows), available),
        "checkpoint_sha256": frozen["sha256"] if frozen else None,
        "checkpoint_selection_rule": frozen.get("rule") if frozen else None,
        "complete": train_stats["complete"] and eval_stats is not None and eval_stats["complete"],
        "source_files": {"training": str(folder / "train.csv"),
                         "evaluation": str(eval_path) if eval_rows else None,
                         "frozen": str(folder / "FROZEN.json") if frozen else None},
    }
    return summary, train_rows, eval_rows


def matplotlib_setup():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlds-assignment1-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.dpi": 120})
    return plt


def title_for(name: str) -> str:
    game = "CartPole-v1" if name.startswith("cartpole") else "ALE/Pong-v5"
    method = "REINFORCE + baseline" if name.endswith("baseline") else "REINFORCE (no baseline)"
    return f"{game}: {method}"


def draw_training(ax, name: str, rows: list[dict], complete: bool) -> None:
    rewards = np.array([row["reward"] for row in rows], dtype=np.float64)
    x = np.arange(1, len(rewards) + 1)
    ax.plot(x, rewards, color="#648BA2", alpha=0.35, linewidth=0.7, label="Raw episode total reward")
    ax.plot(x, trailing_mean(rewards), color="#176B87", linewidth=2,
            label="Trailing 100 episodes (growing window initially)")
    prefix = "" if complete else "PROGRESS — "
    ax.set(xlabel="Training episode (1-based)", ylabel="Undiscounted episode total reward",
           title=prefix + title_for(name))
    ax.set_xlim(left=1, right=max(2, len(rows)))
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)


def draw_evaluation(ax, name: str, rows: list[dict]) -> None:
    rewards = np.array([row["reward"] for row in rows], dtype=np.float64)
    bins = np.arange(-21.5, 22.5, 1) if name.startswith("pong") else (
        np.array([rewards[0] - 0.5, rewards[0] + 0.5]) if np.ptp(rewards) == 0 else 20)
    counts, edges, _ = ax.hist(rewards, bins=bins, color="#57A0A2", edgecolor="white", linewidth=0.6)
    mean = float(rewards.mean())
    std = float(rewards.std(ddof=1)) if len(rewards) > 1 else None
    ax.axvline(mean, color="#C8553D", linestyle="--", linewidth=1.5, label="Mean")
    std_text = f"{std:.3f}" if std is not None else "undefined (n < 2)"
    if name.startswith("pong"):
        ax.set_xlim(-21.5, 21.5)
    peak = int(np.argmax(counts))
    peak_center = 0.5 * (edges[peak] + edges[peak + 1])
    midpoint = 0.5 * sum(ax.get_xlim())
    # Put statistics opposite the tallest bar, keeping its visible height clear.
    annotation_left = peak_center >= midpoint
    ax.text(0.03 if annotation_left else 0.97, 0.96,
            f"n = {len(rows)}\nMean = {mean:.3f}\nSample SD = {std_text}\nSD uses ddof = 1",
            transform=ax.transAxes, ha="left" if annotation_left else "right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.4"})
    prefix = "" if len(rows) == EXPECTED_EVALUATIONS else f"PROGRESS ({len(rows)}/500) — "
    ax.set(xlabel="Undiscounted complete-episode total reward", ylabel="Number of episodes",
           title=prefix + title_for(name))
    ax.grid(axis="y", alpha=0.2)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--available", action="store_true",
                        help="Allow unfinished/missing canonical runs and explicitly label progress plots")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing runs/")
    parser.add_argument("--figures", type=Path, help="Figure destination (default: ROOT/figures)")
    parser.add_argument("--summary", type=Path, help="Metrics destination (default: ROOT/data/metrics_summary.json)")
    parser.add_argument("--panels", action="store_true", help="Also produce 2x2 training and evaluation comparison panels")
    args = parser.parse_args()
    root = args.root.resolve()
    figures = args.figures or root / "figures"
    summary_path = args.summary or root / "data/metrics_summary.json"
    loaded = {}
    errors = {}
    for name in RUN_NAMES:
        try:
            loaded[name] = load_run(root / "runs" / name, name, args.available)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors[name] = str(exc)
    if errors and not args.available:
        print("Final figures refused: all four genuine, complete evaluations are required.", file=sys.stderr)
        for name, error in errors.items():
            print(f"  {name}: {error}", file=sys.stderr)
        return 2
    # Validate all runs before creating any final-mode outputs.
    figures.mkdir(parents=True, exist_ok=True)
    plt = matplotlib_setup()
    for name, (summary, train_rows, eval_rows) in loaded.items():
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        draw_training(ax, name, train_rows, summary["training"]["complete"])
        fig.savefig(figures / f"{name}_training.png")
        plt.close(fig)
        if eval_rows:
            fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
            draw_evaluation(ax, name, eval_rows)
            fig.savefig(figures / f"{name}_evaluation.png")
            plt.close(fig)
    if args.panels and loaded:
        for kind in ("training", "evaluation"):
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            for ax, name in zip(axes.ravel(), RUN_NAMES):
                if name not in loaded:
                    ax.text(0.5, 0.5, f"{name}\nNot yet available", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                summary, train_rows, eval_rows = loaded[name]
                if kind == "training":
                    draw_training(ax, name, train_rows, summary["training"]["complete"])
                elif eval_rows:
                    draw_evaluation(ax, name, eval_rows)
                else:
                    ax.text(0.5, 0.5, f"{name}\nEvaluation not yet available", ha="center", va="center")
                    ax.set_axis_off()
            fig.savefig(figures / f"rl_{kind}_comparison.png")
            plt.close(fig)
    result = {
        "schema_version": 1, "generated_unix_seconds": time.time(),
        "mode": "available_progress" if args.available else "final",
        "complete": len(loaded) == 4 and all(item[0]["complete"] for item in loaded.values()),
        "required_runs": list(RUN_NAMES), "required_evaluation_episodes_per_run": EXPECTED_EVALUATIONS,
        "training_reward_definition": "Raw undiscounted sum of all rewards in the complete episode",
        "moving_average_definition": "Trailing 100 complete episodes including current; min_periods=1 for episodes 1–99",
        "evaluation_std_definition": "Sample standard deviation, denominator n-1 (ddof=1)",
        "seed_protocol": "Evaluation seeds are unique within each run and disjoint from its configured training range; shared evaluation seeds across methods permit paired comparison",
        "runs": {name: item[0] for name, item in loaded.items()}, "unavailable_or_invalid": errors,
    }
    write_json_atomic(summary_path, result)
    print(json.dumps({"complete": result["complete"], "runs_plotted": list(loaded),
                      "evaluation_episodes": {name: len(item[2]) for name, item in loaded.items()},
                      "summary": str(summary_path), "unavailable_or_invalid": errors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
