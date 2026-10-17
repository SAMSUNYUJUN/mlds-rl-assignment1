"""Exact finite-state shuttle dispatching solvers and independent validation.

All values are discounted *costs* to minimize.  At epoch t, observe s_t,
dispatch at most one shuttle, charge dispatch and post-dispatch holding cost,
then add A_t arrivals and clip the next queue to max_queue.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np


@dataclass(frozen=True)
class ShuttleConfig:
    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    max_queue: int = 200
    gamma: float = 0.95
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)
    horizon: int = 500
    tolerance: float = 1e-10


def build_model(cfg: ShuttleConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return c[a,s] and P[a,s,s']; duplicate capped branches accumulate."""
    if not 0 <= cfg.gamma < 1:
        raise ValueError("gamma must lie in [0,1)")
    if cfg.max_queue < 0 or cfg.capacity < 1 or not cfg.arrivals:
        raise ValueError("invalid finite-state shuttle model")
    if min(cfg.arrivals) < 0:
        raise ValueError("arrivals must be nonnegative")
    states = np.arange(cfg.max_queue + 1)
    costs = np.zeros((2, len(states)), dtype=np.float64)
    transitions = np.zeros((2, len(states), len(states)), dtype=np.float64)
    for action in (0, 1):
        remaining = np.maximum(states - cfg.capacity * action, 0)
        costs[action] = cfg.dispatch_cost * action + cfg.holding_cost * remaining
        for arrival in cfg.arrivals:
            next_states = np.minimum(remaining + arrival, cfg.max_queue)
            transitions[action, states, next_states] += 1.0 / len(cfg.arrivals)
    return costs, transitions


def action_values(value: np.ndarray, costs: np.ndarray, transitions: np.ndarray,
                  gamma: float) -> np.ndarray:
    return costs + gamma * np.einsum("asj,j->as", transitions, value)


def finite_horizon(cfg: ShuttleConfig, costs: np.ndarray,
                   transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate both actions and all arrival branches via backward recursion.

    V_T=0; no exponential enumeration of complete trajectories is needed.
    """
    values = np.zeros((cfg.horizon + 1, cfg.max_queue + 1), dtype=np.float64)
    policies = np.zeros((cfg.horizon, cfg.max_queue + 1), dtype=np.uint8)
    for t in range(cfg.horizon - 1, -1, -1):
        q = action_values(values[t + 1], costs, transitions, cfg.gamma)
        values[t] = q.min(axis=0)
        policies[t] = q.argmin(axis=0)
    return values, policies


def value_iteration(cfg: ShuttleConfig, costs: np.ndarray,
                    transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    value = np.zeros(cfg.max_queue + 1, dtype=np.float64)
    history = []
    for iteration in range(1, 100_001):
        q = action_values(value, costs, transitions, cfg.gamma)
        updated = q.min(axis=0)
        delta = float(np.max(np.abs(updated - value)))
        history.append(delta)
        value = updated
        if delta <= cfg.tolerance:
            break
    else:
        raise RuntimeError("value iteration failed to converge")
    q = action_values(value, costs, transitions, cfg.gamma)
    residual = float(np.max(np.abs(q.min(axis=0) - value)))
    return value, q.argmin(axis=0).astype(np.uint8), {
        "iterations": iteration, "successive_differences": history,
        "bellman_residual": residual,
        "residual_error_bound": residual / (1 - cfg.gamma),
    }


def policy_iteration(cfg: ShuttleConfig, costs: np.ndarray,
                     transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Howard policy iteration with exact float64 linear-system evaluation."""
    n = cfg.max_queue + 1
    states = np.arange(n)
    policy = np.zeros(n, dtype=np.uint8)
    history = []
    for iteration in range(1, 1001):
        p_policy = transitions[policy, states]
        c_policy = costs[policy, states]
        value = np.linalg.solve(np.eye(n) - cfg.gamma * p_policy, c_policy)
        q = action_values(value, costs, transitions, cfg.gamma)
        new_policy = q.argmin(axis=0).astype(np.uint8)
        changed = int(np.sum(new_policy != policy))
        history.append({"iteration": iteration, "changed_states": changed,
                        "evaluation_residual": float(np.max(np.abs(
                            c_policy + cfg.gamma * p_policy @ value - value)))})
        if changed == 0:
            break
        policy = new_policy
    else:
        raise RuntimeError("policy iteration failed to stabilize")
    residual = float(np.max(np.abs(q.min(axis=0) - value)))
    return value, policy, {"iterations": iteration, "history": history,
                           "bellman_residual": residual}


def explicit_tree_value(s: int, remaining_steps: int, cfg: ShuttleConfig) -> float:
    """Independent, deliberately exponential small-horizon oracle.

    No model tensors, Bellman routine, or cache are reused.  Every action and
    outcome subtree is traversed explicitly, including duplicate clipped states.
    """
    if remaining_steps == 0:
        return 0.0
    candidates = []
    for send in (False, True):
        people_after_service = max(0, s - cfg.capacity) if send else s
        cost = (cfg.dispatch_cost if send else 0.0) + cfg.holding_cost * people_after_service
        expected_future = sum(explicit_tree_value(
            min(people_after_service + a, cfg.max_queue), remaining_steps - 1, cfg
        ) for a in cfg.arrivals) / len(cfg.arrivals)
        candidates.append(cost + cfg.gamma * expected_future)
    return min(candidates)


def no_arrival_closed_form(s: int, cfg: ShuttleConfig) -> float:
    """Independent analytic cost: m immediate dispatches, then wait forever.

    With zero arrivals, a stationary deterministic optimal policy dispatches
    immediately at a state or waits there forever.  Enumerating m covers both.
    """
    candidates = []
    for m in range((s + cfg.capacity - 1) // cfg.capacity + 1):
        total = 0.0
        queue = s
        for t in range(m):
            queue = max(0, queue - cfg.capacity)
            total += cfg.gamma**t * (cfg.dispatch_cost + cfg.holding_cost * queue)
        total += cfg.gamma**m * cfg.holding_cost * queue / (1 - cfg.gamma)
        candidates.append(total)
    return min(candidates)


def validate(cfg: ShuttleConfig, costs: np.ndarray, transitions: np.ndarray,
             finite: np.ndarray, vi_value: np.ndarray, vi_policy: np.ndarray,
             pi_value: np.ndarray, pi_policy: np.ndarray, vi_info: dict,
             pi_info: dict) -> dict:
    tests = []

    # A tiny, fully traversed decision tree is independent of the matrix solver.
    small = ShuttleConfig(capacity=3, dispatch_cost=7, holding_cost=2,
                          max_queue=7, gamma=0.9, arrivals=(1, 2), horizon=4)
    small_c, small_p = build_model(small)
    small_v, _ = finite_horizon(small, small_c, small_p)
    comparisons = [
        {"state": s, "horizon": h, "dp": float(small_v[small.horizon - h, s]),
         "explicit_tree": explicit_tree_value(s, h, small)}
        for h in range(small.horizon + 1) for s in range(small.max_queue + 1)
    ]
    tree_error = max(abs(row["dp"] - row["explicit_tree"]) for row in comparisons)
    assert tree_error < 1e-11
    tests.append({
        "name": "independent_small_tree", "purpose": "Check finite-horizon expectation and minimization against a distinct implementation.",
        "theoretical_expectation": "An explicit tree over both actions and both arrival branches agrees with dynamic programming for all eight states and horizons 0 through 4.",
        "method": "Recursively traverse the complete tiny decision tree without shared transition arrays or memoization; compare all 40 state/horizon pairs.",
        "result": {"comparisons": len(comparisons), "max_absolute_error": tree_error, "rows": comparisons},
        "interpretation": "Agreement supports the finite recursion including terminal value, branching probabilities, and queue clipping; the separate boundary test checks the shared timing convention.",
    })

    # Hand-derived one-stage cases test exactly which customers incur holding.
    boundary_specs = [
        (0, 0, 0.0, [1, 2, 3, 4, 5]),
        (0, 1, 100.0, [1, 2, 3, 4, 5]),
        (14, 1, 100.0, [1, 2, 3, 4, 5]),
        (15, 1, 100.0, [1, 2, 3, 4, 5]),
        (16, 1, 102.0, [2, 3, 4, 5, 6]),
        (198, 0, 396.0, [199, 200, 200, 200, 200]),
        (200, 0, 400.0, [200, 200, 200, 200, 200]),
        (200, 1, 470.0, [186, 187, 188, 189, 190]),
    ]
    boundary_rows = []
    max_boundary_error = 0.0
    for s, action, expected_cost, outcomes in boundary_specs:
        expected_p = np.zeros(cfg.max_queue + 1)
        for sp in outcomes:
            expected_p[sp] += 0.2
        error = max(abs(float(costs[action, s]) - expected_cost),
                    float(np.max(np.abs(transitions[action, s] - expected_p))))
        max_boundary_error = max(max_boundary_error, error)
        boundary_rows.append({"state": s, "action": action, "cost": float(costs[action, s]),
                              "next_states": np.flatnonzero(transitions[action, s]).tolist(),
                              "probabilities": transitions[action, s][transitions[action, s] > 0].tolist(),
                              "max_error": error})
    assert max_boundary_error < 1e-12
    tests.append({
        "name": "hand_calculated_boundaries", "purpose": "Check empty and underfull service, holding timing, and capped-arrival probability accumulation.",
        "theoretical_expectation": "Empty dispatch costs 100, dispatching 14 or 15 leaves zero holding cost, dispatching 16 costs 102; waiting at 198 puts probability 0.8 on state 200; waiting at 200 costs 400 and remains at 200.",
        "method": "Compare eight manually specified costs and outcome multisets with the constructed model.",
        "result": {"max_absolute_error": max_boundary_error, "cases": boundary_rows},
        "interpretation": "The implementation charges only the passengers left after service and clips arrivals after that charge. Duplicated clipped branches retain their total probability.",
    })

    # A deterministic special case has an independent closed-form solution.
    deterministic = replace(cfg, arrivals=(0,))
    det_c, det_p = build_model(deterministic)
    det_v, det_pi, det_info = value_iteration(deterministic, det_c, det_p)
    analytic = np.array([no_arrival_closed_form(s, deterministic) for s in range(cfg.max_queue + 1)])
    analytic_error = float(np.max(np.abs(det_v - analytic)))
    assert analytic_error < 5e-8
    tests.append({
        "name": "deterministic_zero_arrivals", "purpose": "Test infinite-horizon decisions against an analytic deterministic limit.",
        "theoretical_expectation": "With arrivals fixed to zero, enumerate m immediate dispatches then perpetual waiting; each candidate has a finite dispatch sum plus a geometric holding-cost tail. States 0,1,2 wait forever and all states 3 through 200 dispatch.",
        "method": "Solve the zero-arrival model with VI and compare every state with the independently derived geometric-series formula.",
        "result": {"max_absolute_error": analytic_error, "dispatch_states": np.flatnonzero(det_pi).tolist(),
                   "values_at_0_1_2_3_15_16_200": det_v[[0, 1, 2, 3, 15, 16, 200]].tolist(),
                   "bellman_residual": det_info["bellman_residual"]},
        "interpretation": "The exact low-state costs (0,40,80,100) and dispatch threshold 3 are recovered without stochastic arrivals. This tests discounting, stage costs, and repeated dispatches.",
    })

    pi_diff = float(np.max(np.abs(vi_value - pi_value)))
    policy_disagreements = int(np.sum(vi_policy != pi_policy))
    assert vi_info["bellman_residual"] < 2e-10
    assert pi_info["bellman_residual"] < 1e-8
    assert pi_diff < 1e-7 and policy_disagreements == 0
    tests.append({
        "name": "optimality_residual_and_solver_agreement", "purpose": "Quantify infinite-horizon optimality and test an iterative fixed point against exact policy evaluation.",
        "theoretical_expectation": "A discounted Bellman contraction bounds VI value error by residual/(1-gamma); stable PI has zero Bellman residual in exact arithmetic and should select the same strict minimizers.",
        "method": "Compute max_s |min_a Q(s,a)-V(s)|, compare independent solution paths VI and PI, and check every selected action.",
        "result": {"vi_bellman_residual": vi_info["bellman_residual"],
                   "vi_residual_error_bound": vi_info["residual_error_bound"],
                   "pi_bellman_residual": pi_info["bellman_residual"],
                   "max_vi_pi_value_difference": pi_diff, "policy_disagreements": policy_disagreements},
        "interpretation": "The residual provides an absolute numerical error certificate. Agreement alone is weaker because the solvers share model tensors, which is why independent analytic and hand-calculated tests are also included.",
    })

    horizon_gap = float(np.max(np.abs(finite[0] - pi_value)))
    tail_bound = float(cfg.gamma**cfg.horizon * np.max(costs) / (1 - cfg.gamma))
    assert np.all(finite[0] <= pi_value + 1e-8)
    assert horizon_gap <= tail_bound + 1e-8
    tests.append({
        "name": "finite_infinite_tail_bound", "purpose": "Check the zero-terminal-value interpretation and long finite horizon.",
        "theoretical_expectation": "Nonnegative costs imply V_0^(500) <= V*. The omitted tail is at most gamma^500 max(c)/(1-gamma).",
        "method": "Compare all 201 finite-horizon initial values with the converged PI values and compute the geometric bound.",
        "result": {"max_finite_infinite_difference": horizon_gap, "geometric_tail_bound": tail_bound,
                   "min_infinite_minus_finite": float(np.min(pi_value - finite[0]))},
        "interpretation": "Finite and infinite curves almost overlap because gamma^500 is tiny, rather than because the finite computation reused the infinite solution.",
    })

    # Exact evaluation of a poor policy supplies a task-level baseline.
    all_wait = np.linalg.solve(np.eye(cfg.max_queue + 1) - cfg.gamma * transitions[0], costs[0])
    assert abs(all_wait[-1] - 8000.0) < 1e-8
    assert np.all(pi_value <= all_wait + 1e-8)
    tests.append({
        "name": "never_dispatch_reference", "purpose": "Demonstrate that optimization improves a real task cost and verify capped absorbing behavior.",
        "theoretical_expectation": "Never dispatching at state 200 incurs 400 forever, hence value 400/(1-.95)=8000. An optimal policy cannot cost more than this feasible policy at any state.",
        "method": "Solve the never-dispatch policy exactly, check the hand-computed cap value, and compare against V* across all states.",
        "result": {"never_dispatch_value_at_200": float(all_wait[-1]), "optimal_value_at_200": float(pi_value[-1]),
                   "improvement_at_200": float(all_wait[-1] - pi_value[-1]),
                   "minimum_improvement_over_states": float(np.min(all_wait - pi_value))},
        "interpretation": "The optimized policy reduces discounted cost relative to a plausible poor policy, including at a congested initial state.",
    })

    q = action_values(pi_value, costs, transitions, cfg.gamma)
    action_gap = np.abs(q[1] - q[0])
    return {"passed": True, "test_count": len(tests), "tests": tests,
            "minimum_optimal_action_gap": float(action_gap.min()),
            "minimum_gap_state": int(action_gap.argmin()),
            "commands": [".venv/bin/python src/shuttle.py all", ".venv/bin/python src/shuttle.py validate"],
            "limits": "These are exact model-based checks, not field validation of passenger behavior. The artificial hard cap can discard arrivals; alternative cap and overflow-cost assumptions should be studied separately."}


def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def solve(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cfg = ShuttleConfig()
    start = time.perf_counter()
    costs, transitions = build_model(cfg)
    timings = {}
    t = time.perf_counter()
    finite, finite_policy = finite_horizon(cfg, costs, transitions)
    timings["finite_horizon_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    vi_value, vi_policy, vi_info = value_iteration(cfg, costs, transitions)
    timings["value_iteration_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    pi_value, pi_policy, pi_info = policy_iteration(cfg, costs, transitions)
    timings["policy_iteration_seconds"] = time.perf_counter() - t
    validation = validate(cfg, costs, transitions, finite, vi_value, vi_policy,
                          pi_value, pi_policy, vi_info, pi_info)
    timings["total_seconds_including_validation"] = time.perf_counter() - start
    q = action_values(pi_value, costs, transitions, cfg.gamma)
    np.savez_compressed(output / "results.npz", finite_values=finite,
                        finite_policies=finite_policy, vi_value=vi_value,
                        vi_policy=vi_policy, pi_value=pi_value, pi_policy=pi_policy,
                        q_wait=q[0], q_dispatch=q[1])
    with (output / "state_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["waiting_passengers", "finite_T500_V0_cost", "infinite_VI_cost",
                         "infinite_PI_cost", "optimal_action", "Q_wait_cost", "Q_dispatch_cost"])
        writer.writerows((s, finite[0, s], vi_value[s], pi_value[s], int(pi_policy[s]), q[0, s], q[1, s])
                         for s in range(cfg.max_queue + 1))
    dispatch_states = np.flatnonzero(pi_policy)
    summary = {"config": asdict(cfg), "value_convention": "nonnegative discounted costs, minimized",
               "enumeration_definition": "finite-horizon backward recursion enumerating actions and arrival branches, with V_T=0",
               "optimal_policy": {"dispatch_states": dispatch_states.tolist(),
                                   "wait_states": np.flatnonzero(pi_policy == 0).tolist(),
                                   "is_threshold": bool(np.all(pi_policy[dispatch_states[0]:] == 1)) if len(dispatch_states) else False,
                                   "threshold": int(dispatch_states[0]) if len(dispatch_states) else None},
               "values_at_states": {str(s): {"finite": float(finite[0, s]), "vi": float(vi_value[s]),
                                               "pi": float(pi_value[s]), "action": int(pi_policy[s])}
                                    for s in [0, 1, 14, 15, 16, 20, 50, 100, 198, 200]},
               "value_iteration": vi_info, "policy_iteration": pi_info,
               "timing": timings,
               "runtime": {"platform": platform.platform(), "machine": platform.machine(),
                           "python": sys.version, "numpy": np.__version__, "executable": sys.executable}}
    dump_json(output / "summary.json", summary)
    dump_json(output / "config.json", asdict(cfg))
    dump_json(output / "validation.json", validation)
    return summary


def plot(output: Path, figures: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlds-assignment1-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    results = np.load(output / "results.npz")
    summary = json.loads((output / "summary.json").read_text())
    states = np.arange(len(results["pi_value"]))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.dpi": 120})
    for key, filename, title, ylabel in [
        ("finite_values", "shuttle_enumeration.png", "Finite-horizon enumeration: T = 500, terminal cost 0", "$V_0(s)$: expected discounted cost"),
        ("vi_value", "shuttle_value_iteration.png", "Infinite-horizon value iteration", "$V^*(s)$: expected discounted cost"),
    ]:
        value = results[key][0] if key == "finite_values" else results[key]
        fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
        ax.plot(states, value, color="#176B87", linewidth=2)
        ax.set(xlabel="Passengers waiting at the decision epoch, s", ylabel=ylabel, title=title,
               xlim=(0, states[-1]))
        ax.grid(alpha=0.2)
        fig.savefig(figures / filename)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    policy = results["pi_policy"]
    threshold = summary["optimal_policy"]["threshold"]
    ax.step(states, policy, where="mid", color="#C8553D", linewidth=2)
    ax.axvline(threshold - 0.5, color="#707070", linestyle="--", linewidth=1,
               label=f"Dispatch from s = {threshold}")
    ax.set(xlabel="Passengers waiting at the decision epoch, s", ylabel="Optimal action",
           title="Infinite-horizon policy iteration", yticks=[0, 1], yticklabels=["Wait", "Dispatch"],
           xlim=(0, states[-1]), ylim=(-0.15, 1.15))
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(figures / "shuttle_policy_iteration.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "solve", "validate", "plot"], nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=Path("data/shuttle"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.command in ("all", "solve", "validate"):
        summary = solve(args.output)
        print(json.dumps({"optimal_threshold": summary["optimal_policy"]["threshold"],
                          "VI_iterations": summary["value_iteration"]["iterations"],
                          "PI_iterations": summary["policy_iteration"]["iterations"],
                          "VI_Bellman_residual": summary["value_iteration"]["bellman_residual"],
                          "PI_Bellman_residual": summary["policy_iteration"]["bellman_residual"],
                          "timing": summary["timing"]}, indent=2))
    if args.command in ("all", "plot"):
        plot(args.output, args.figures)


if __name__ == "__main__":
    main()
