# AB-MDP simulation

Reproduction code for the illustrative study in

> Li, W. *Ontological Bounds of Autoregressive Architectures: An Autonomic-Boundary MDP
> Framework for Physical AI* (2026).

## What this code does

It compares two ways of enforcing a viability constraint in a bounded 2D navigation task:

1. **Penalty relaxation** — the standard Lagrangian objective
   `J = E[sum gamma^t r_t] - lambda * E[sum gamma^t c_t]`, with a per-step violation cost
   `c_t = 1{||s_{t+1}|| > 1}`. Trained by REINFORCE with separate reward- and cost-to-go
   policy-gradient estimates.
2. **AB-MDP filter** — the *same* trained policy, with the action projected onto the robust
   one-step invariance set
   `A(s) = { a : ||a|| <= a_max,  ||s + a||_2 + eps_max <= 1 }`,
   which is exactly the condition `f(s,a) + eps in K` for all `||eps|| <= eps_max`. The
   projection of the nominal action onto `A(s)` (an intersection of two Euclidean balls) is
   computed with Dykstra's alternating projection algorithm.

Using the same nominal policy for both controllers is deliberate: it isolates the effect of
restricting the admissible set from the effect of learning a different policy.

The `sigma_pain` scalar `exp(-dist(s, dK)/eta)` modulates the exploration noise scale,
`(1 - sigma_pain(s))`, i.e. exploration is damped as the boundary approaches. It is a
modulation signal only — the confinement guarantee comes from the filter, not from
`sigma_pain`.

## Why there is no adaptive-multiplier baseline

A primal–dual (dual-ascent) baseline was implemented and tested during preparation. It did not
converge stably in our hands: the multiplier saturated and the policy froze, so the comparison
would have confounded solver tuning with the structural question. It was removed rather than
reported, and the paper states its claim narrowly — *no instance-independent finite penalty
weight* — which is what the reward-scale and penalty-weight sweeps test.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python ab_mdp_simulation.py            # full study
python ab_mdp_simulation.py --figure-only   # rebuild figure1.* and figure2.* from results/ (seconds)
python ab_mdp_simulation.py --skip-study --only-eps   # re-run only the disturbance sweep
python ab_mdp_simulation.py --runs 400 --sweep-runs 150 --sweep-seeds 2   # quick check
```

`--figure-only` regenerates `figure1.pdf/.png` and `figure2.pdf/.png` from the released
`results/*.json` and `results/trajectories.npz` without retraining. It is the recommended way to
check that the figures and the JSON agree: the panel titles read the seed count **out of the
data**, so a caption can no longer drift from the released numbers.

Outputs land in `results/`:

| File | Contents |
|---|---|
| `summary.json` | headline multi-seed statistics, including **per-run** breach rates and the margin profile |
| `sweep_reward_scale.json` | violation rate vs. task reward scale (per-seed arrays) |
| `sweep_lambda.json` | violation rate vs. penalty weight (per-seed arrays) |
| `sweep_eps_max.json` | **the disturbance-bound sweep**, including the measured vacuous-step count and how breaches are attributed to vacuous vs. non-vacuous steps |
| `figure1.pdf/.png` | four-panel figure used in the manuscript |
| `figure2.pdf/.png` | three-panel figure, including the disturbance sweep in panel (c) |
| `trajectories.npz` | raw trajectories for re-analysis |

Both sweeps use the same seed count (`--sweep-seeds`, 8 by default), and the figure titles and
the manuscript caption both read it **out of the JSON** rather than stating it in prose, so a
caption cannot drift from the released data.

## The disturbance sweep is a test of Lemma 1, not an illustration of it

The manuscript's Lemma 1 says the robust one-step invariance set is non-empty on the whole
kernel exactly when `eps_max <= a_max`, and that above the threshold a shell of states
`{ s : dist(s, dK) < eps_max - a_max }` admits no admissible action at all. `sweep_eps_max.json`
sweeps `eps_max` across `a_max = 0.10`, and `rollout` records vacuity **directly**, from the sign
of the margin `m(s)` evaluated before the action is chosen, rather than inferring it from a
failed projection:

* at every `eps_max <= 0.10` there are exactly zero vacuous steps and exactly zero breaches;
* at the first `eps_max > 0.10` both switch on together, and the vacuous shell then widens with
  `eps_max - a_max` as the lemma predicts;
* every breach step above the threshold is a step on which `A(s) = ∅`, so Proposition 3 (the
  confinement guarantee) is not contradicted — the mechanism has to fail before the guarantee can.

On a vacuous step the agent applies the maximally inward action `-a_max * s_hat`, i.e. the most
favourable response available to it, so any breach that follows is attributable to the eroded
kernel rather than to an arbitrary fallback rule.

`make_sweep_table.py` regenerates Supplementary Tables S1 and S2 from these records and **fails**
if the lemma's prediction ever disagrees with the measured vacuity.

## Validation scripts

| Script | What it asserts |
|---|---|
| `verify_numbers.py` | 39 checks that each number quoted in the manuscript (including the three in the abstract) matches the released JSON, plus record-level consistency (Lemma 1 vs. the sweep; the reward-scale and penalty-weight patterns the text claims), plus a rounding-tie screen over every quoted value |
| `make_sweep_table.py` | generates Tables S1/S2 from the JSON and fails if Lemma 1's prediction and the measured vacuity disagree |
| `test_intervals.py` | validates the in-repo exact Clopper–Pearson implementation against closed-form boundary identities, published table values, the defining tail property, and beta symmetry |

## Reported results (seed 20260917, defaults)

| Quantity | Value |
|---|---|
| Penalty relaxation, kernel breach rate | **87.25% ± 32.5%** of episodes across 15 independent training runs (median 100%; 13 of 15 runs breached in 100% of their episodes) |
| Run-level leaky fraction | 13/15 = 86.7%, exact Clopper–Pearson 95% interval [59.5%, 98.3%] |
| AB-MDP, kernel breaches | **0** across 3×10^6 evaluated steps (exact one-sided 95% bound 1.2×10^-6 per step) |
| AB-MDP, filter active | 80.3% of steps; pooled mean correction ‖Δa‖₂ = 0.094 (bound 0.1) |
| Mean task return | 18.5 (filtered) vs. 4.9 (unfiltered); final distance 1.69 vs. 5.36 |
| Mean `sigma_pain` along filtered trajectories | 0.660 |
| Disturbance sweep, ε_max ≤ a_max = 0.10 | zero vacuous steps, zero breaches (all 4 grid points) |
| Disturbance sweep, ε_max > a_max | vacuous steps 22,355 / 61,881 / 113,305 / 190,567 and breaches 290 / 2,630 / 10,521 / 36,457 at ε_max = 0.12 / 0.15 / 0.20 / 0.30 — **every breach step is a vacuous step** |

Two conventions are worth stating explicitly, because both are places a reader could otherwise
recompute a different number:

* Every `±` above is a **population** standard deviation over the stated number of runs
  (`ddof = 0`, numpy's default), not a standard error and not a sample standard deviation. At
  n = 15 the sample convention would be larger by √(15/14) ≈ 1.035, i.e. 33.65% rather than
  32.5%.
* The 15-run mean is **exactly 87.25%**, which is a rounding tie at one decimal place: Python's
  `%.1f` gives 87.2 while rounding by hand gives 87.3. The manuscript quotes two decimals, and
  `verify_numbers.py` screens every quoted value for a tie at the precision it is printed with.

Absolute rates depend on the trained policy: the run-to-run spread is part of the reported
result, not noise to be averaged away.

## Configuration

All parameters live in the `Config` dataclass: `a_max=0.10`, `eps_max=0.05`, `radius=1.0`,
`gamma=0.99`, `lam=10.0`, `reward_scale=2.0`, `horizon=100`, `s0=(-0.5,-0.5)`,
`goal=(1.4,0.0)`, `eta=0.20`, `train_episodes=1200`, `seed=20260917`.

## License

MIT — see `LICENSE` at the repository root.
