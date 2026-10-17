# Shuttle dispatch: model, exact solution, and quantitative verification

The supplied assignment specifies capacity 15, dispatch cost 100, cost 2 per customer left after dispatch, uniform arrivals in {1,2,3,4,5}, a queue cap of 200, discount factor 0.95, and horizon 500 for enumeration. There were no lecture notes or starter code in the initial project folder. The event ordering, overflow behavior, and finite terminal value below are explicit implementation assumptions.

## MDP and event order

At decision epoch t:

1. Observe the integer queue s_t in S={0,...,200}.
2. Choose a_t in {0,1}, where 0 means wait and 1 means send one shuttle.
3. Board min(s_t,15) passengers if a_t=1. The remaining queue is r(s,a)=max(s-15a,0).
4. Incur c(s,a)=100a+2r(s,a).
5. Draw A_t independently and uniformly from {1,2,3,4,5}. Set s_(t+1)=min(r(s_t,a_t)+A_t,200).

New arrivals do not incur holding cost in their arrival interval. The passengers left after boarding do. Empty and partially filled shuttles are allowed, each at full fixed cost. Sending an empty shuttle is dominated by waiting because it has identical transitions and costs 100 more. At most one shuttle is available per epoch; there is no travel-time state or future fleet constraint. These are assumptions rather than quotations from the PDF.

The cap is enforced by clipping the queue after new arrivals; overflow customers are discarded with no extra cost. For example, at s=198 and a=0 the next state is 199 with probability 0.2 and 200 with probability 0.8; at s=200 and a=0 it stays 200. This artificial cap is part of the computational model and is not a real-station conservation claim. Under the optimal policy starting near empty, dispatch keeps queues far below this cap.

The transition kernel is

P(s'|s,a) = (1/5) sum over u=1,...,5 of 1{s'=min(max(s-15a,0)+u,200)}.

All reported values are **costs**, minimized, rather than negative rewards. For a policy pi,

J_T^pi(s)=E_pi[sum from t=0 to T-1 of gamma^t c(s_t,a_t) | s_0=s],

and the infinite-horizon objective replaces T-1 with infinity. There is no additional terminal cost.

## Algorithms

**Enumeration.** No classroom-specific enumeration definition was available. We explicitly enumerate both actions and five possible arrival outcomes in a finite-horizon backward recursion, with V_T(s)=0, T=500:

V_t(s)=min_a {c(s,a)+gamma sum_s' P(s'|s,a)V_(t+1)(s')}.

The transition tensor aggregates the five arrival branches once; backward recursion evaluates this exact expectation. It does not enumerate an exponentially large collection of all 500-step trajectories. The full V_t and policy arrays are saved, so V_0 is an actual finite-horizon result.

**Value iteration.** Start at V^(0)=0 and repeatedly apply the infinite-horizon Bellman operator. Stop when the maximum successive difference is at most 1e-10. The post-update Bellman residual is separately measured. Float64 arithmetic is used on the CPU.

**Policy iteration.** Start with the never-dispatch policy. Evaluate each policy by solving (I-gamma P_pi)V_pi=c_pi using a dense float64 linear solve, then greedily minimize both action values at every state. Stop when no state changes its action. Argmin breaks an exact tie in favor of waiting, but the final smallest action gap is 3.7780043855, so tie handling does not determine this solution.

## Results and figure interpretation

The optimal infinite-horizon policy waits at states 0 through 12 and dispatches at every state 13 through 200. This threshold is found by unconstrained per-state optimization; no threshold structure was imposed. A full shuttle is not always worth waiting for: at state 13, waiting has Q=680.7698310199, while dispatching costs Q=676.6701677662. At state 12, waiting costs Q=672.8921633807 versus Q=676.6701677662 for dispatch.

| Quantity | Measured result |
|---|---:|
| V*(0) | 576.6701677662 |
| V*(15) | 676.6701677662 |
| V*(200) | 3821.5864771839 |
| VI iterations | 519 |
| PI policy evaluations | 6 |
| VI Bellman residual | 9.4132701633e-11 |
| PI Bellman residual | 9.0949470177e-13 |
| Maximum VI/PI value difference | 1.8649188860e-9 |
| Policy disagreements | 0 of 201 states |
| Maximum finite-T500 / infinite value difference | 4.9403752200e-9 |

The exact runtimes and runtime identity are saved in `data/shuttle/summary.json`; the official run uses the project's native arm64 Python environment on the Mac and CPU. These tiny tabular problems complete in well under one second, excluding importing and drawing figures.

- `figures/shuttle_enumeration.png`: V_0(s) with T=500 and zero terminal cost. Congestion increases discounted cost; short flat sections arise when dispatch clears any queue up to capacity. The near coincidence with the infinite-horizon curve is expected because gamma^500 is tiny.
- `figures/shuttle_value_iteration.png`: the converged V*(s), using positive cost units. At highly congested states, a single shuttle cannot clear the queue, so costs rise substantially.
- `figures/shuttle_policy_iteration.png`: the complete policy on states 0 through 200. The transition from wait to dispatch occurs at 13; underfull departures are optimal in this model.

## Validation: purpose → theory → method → result → interpretation

The full machine-readable evidence, including each small-tree comparison and each boundary case, is in `data/shuttle/validation.json`. Run `.venv/bin/python src/shuttle.py validate` to reproduce it. These checks test consequences of the model and algorithms beyond shape or execution tests.

| Purpose | Theoretical prediction | Actual method and quantitative result | Interpretation |
|---|---|---|---|
| Independently verify finite-horizon optimization | Full small decision-tree traversal and backward recursion agree. | With capacity 3, dispatch cost 7, holding cost 2, cap 7, arrivals {1,2}, gamma .9 and horizons 0–4, recursively enumerate every action/outcome subtree without shared transition arrays or memoization. All 40 state/horizon comparisons have error **0**. | Tests expectation, action minimization, clipping and terminal value through an independent implementation. |
| Verify physical timing and boundaries | Empty and underfull departure costs 100; s=16 departure costs 102; waiting at 198 yields probabilities .2 on 199 and .8 on 200. | Compare eight hand-calculated one-stage distributions and costs, including s=0,14,15,16,198,200. Maximum error **0**. | Detects charging arrivals too early, charging boarded customers, losing probability at the cap, or treating empty service as free. |
| Verify discounting in a deterministic limit | With arrivals fixed to zero, at each state a stationary policy either dispatches immediately or waits there forever. Enumerate the number of immediate dispatches, followed by a geometric holding-cost tail. States 0,1,2 wait; states >=3 dispatch. | Independently compute this closed form for all 201 states and compare with zero-arrival VI. Maximum difference **1.7989805201e-9**; recovered low-state values are approximately (0,40,80,100) at s=(0,1,2,3). | Recovers an analytical policy and absolute cost scale, including repeated departures, without the stochastic model. |
| Certify numerical optimality | For discounted Bellman operator T, ||V−V*||_infinity <= ||TV−V||_infinity/(1−gamma). Stable PI should agree. | VI residual **9.4133e-11** implies error bound **1.8827e-9**; PI residual **9.0949e-13**; max VI/PI gap **1.8649e-9**; action disagreements **0**. | Absolute residual gives a certificate; cross-solver agreement alone would be insufficient because both use the same model. |
| Verify finite versus infinite interpretation | Nonnegative costs give V_0^(500)<=V*, with tail at most gamma^500 max(c)/(1−gamma). | Maximum gap **4.9404e-9**, below the bound **6.8380e-8**; minimum V*−V_0 is **4.9390e-9**. | Confirms a genuine finite computation with zero terminal cost and explains the nearly overlapping curves. |
| Compare against a feasible poor policy | Never dispatching at the cap costs 400/(1−.95)=8000; optimal cost is no larger at every state. | Exact never-dispatch policy evaluation gives V_wait(200)=**7999.999999999993**. The optimized policy saves **4178.4135228161** there and improves cost at every state. | Confirms a meaningful reduction in task cost and independently checks the capped absorbing state. |

## Additional three months

Test sensitivity to cap sizes and explicit overflow penalties; verify that conclusions at normal queues do not depend on the artificial cap. Vary arrival distributions, holding costs, capacity, and discount factor to test whether the threshold and monotonicity persist. Use a separate event-driven simulator with many replications and confidence intervals to validate predicted discounted costs and service metrics. Compare expected queue size, waiting time, load factor and loss rate; consider nonstationary or correlated arrivals and an explicit fleet/travel-time state. Such extensions should be reported as changed models rather than silently substituted for this assignment.

## Reproduction and saved evidence

From the project root:

```bash
.venv/bin/python src/shuttle.py all
.venv/bin/python src/shuttle.py solve
.venv/bin/python src/shuttle.py validate
.venv/bin/python src/shuttle.py plot
```

`plot` reads saved numerical results and never reruns a solver. `solve` and `validate` both reproduce all solver outputs and the six validation experiments. No random sampling is used in this exact tabular portion, so no seed affects its result.

Saved outputs:

- `data/shuttle/config.json`: exact problem parameters and numerical tolerance.
- `data/shuttle/results.npz`: all 501 finite-horizon value vectors, 500 finite policies, converged VI/PI values and policies, and both action-value vectors.
- `data/shuttle/state_values.csv`: all 201 states, finite/infinite values, actions and action values.
- `data/shuttle/summary.json`: solution, convergence traces, CPU runtimes and native runtime identity.
- `data/shuttle/validation.json`: six quantitative validation experiments and their commands.
- `figures/shuttle_enumeration.png`, `figures/shuttle_value_iteration.png`, `figures/shuttle_policy_iteration.png`: the three required plot contents.
