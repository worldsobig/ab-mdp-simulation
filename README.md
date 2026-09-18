# AB-MDP — Autonomic-Boundary Markov Decision Process

Code, released numerical records and verification scripts for the manuscript

> Li, W. *Ontological Bounds of Autoregressive Architectures: An Autonomic-Boundary MDP
> Framework for Physical AI* (2026).

**The idea in one sentence.** Conventional safe reinforcement learning puts viability in the
*objective* (a penalty term, a constraint on occupancy measures); the AB-MDP puts survival in
the *domain* — an action is available only if the next state remains inside the viability
kernel for every disturbance within the assumed bound. This repository contains the study that
separates the two, the raw records behind every number in the manuscript, and the scripts that
check the prose against those records.

## The comparison

A bounded 2D navigation task, trained both ways:

1. **Penalty relaxation** — the standard Lagrangian objective
   `J = E[Σ γᵗ rₜ] − λ · E[Σ γᵗ cₜ]` with a per-step violation cost `cₜ = 1{‖s_{t+1}‖ > 1}`.
   Trained by REINFORCE with separate reward- and cost-to-go policy-gradient estimates.
2. **AB-MDP filter** — the *same* trained policy, with the action projected onto the robust
   one-step invariance set

   ```
   A(s) = { a : ‖a‖ ≤ a_max  and  ‖s + a‖₂ + ε_max ≤ 1 }
   ```

   which is the condition `f(s,a) + ε ∈ K` for every `‖ε‖ ≤ ε_max`. The projection onto `A(s)`
   — an intersection of two Euclidean balls — uses Dykstra's alternating projection.

Using the same nominal policy for both controllers is deliberate: it isolates the effect of
restricting the *admissible set* from the effect of learning a *different policy*.

`σ_pain = exp(−dist(s, ∂K)/η)` modulates the exploration-noise scale, `1 − σ_pain(s)`. It is a
modulation signal only: the confinement guarantee comes from the filter. Lemma 1 in the
manuscript shows `σ_pain` is an order-isomorphic encoding of the admissibility margin,
`m(s) = −η·log σ_pain(s) + a_max − ε_max`, which is what gives it a derived rather than
decorative role.

## Headline results

| Quantity | Value |
|---|---|
| Penalty relaxation, kernel breach rate | **87.25% ± 32.5%** of episodes over 15 independent training runs (median 100%; 13 of 15 runs breached in 100% of their episodes) |
| Run-level leaky fraction | 13/15 = 86.7%, exact Clopper–Pearson 95% interval [59.5%, 98.3%] |
| AB-MDP, kernel breaches | **0** across 3×10⁶ evaluated steps (exact one-sided 95% bound 1.2×10⁻⁶ per step) |
| AB-MDP, filter active | 80.3% of steps; pooled mean correction ‖Δa‖₂ = 0.094 (actuation bound a_max = 0.1) |
| Mean task return | 18.5 (filtered) vs. 4.9 (unfiltered); final distance 1.69 vs. 5.36 |
| Mean `σ_pain` along filtered trajectories | 0.660 |

Two conventions, stated because both are places a reader could otherwise recompute a
different number:

* Every `±` is a **population** standard deviation over the stated number of runs (`ddof = 0`),
  not a standard error and not a sample standard deviation. At n = 15 the sample convention
  would be larger by √(15/14) ≈ 1.035, i.e. 33.65% rather than 32.5%.
* The 15-run mean is **exactly 87.25%**, a rounding tie at one decimal place: Python's `%.1f`
  gives 87.2, rounding by hand gives 87.3. The manuscript quotes two decimals, and
  `verify_numbers.py` screens every quoted value for a tie at its printed precision.

Absolute rates depend on the trained policy. The run-to-run spread is part of the reported
result, not noise to be averaged away.

## The disturbance sweep tests Lemma 1 rather than illustrating it

Lemma 1 states that `A(s)` is non-empty on the whole kernel exactly when `ε_max ≤ a_max`, and
that above the threshold a shell `{ s : dist(s, ∂K) < ε_max − a_max }` admits no admissible
action at all. `results/sweep_eps_max.json` sweeps `ε_max` across `a_max = 0.10`, and `rollout`
records vacuity **directly** — from the sign of `m(s)` evaluated before the action is chosen,
not inferred from a failed projection:

* at every `ε_max ≤ 0.10` there are exactly zero vacuous steps and exactly zero breaches;
* at the first `ε_max > 0.10` both switch on together, and the vacuous shell widens with
  `ε_max − a_max` as the lemma predicts;
* **every breach step above the threshold is a step on which `A(s) = ∅`**, so the confinement
  guarantee is not contradicted — the mechanism has to fail before the guarantee can.

On a vacuous step the agent applies the maximally inward action `−a_max·ŝ`, the most favourable
response available to it, so any breach that follows is attributable to the eroded kernel rather
than to an arbitrary fallback rule.

`make_sweep_table.py` regenerates Supplementary Tables S1 and S2 from these records and **fails**
if the lemma's prediction ever disagrees with the measured vacuity.

## Reproduce

```bash
git clone <this repo> && cd ab-mdp-simulation/simulation
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python ab_mdp_simulation.py                            # full study (rewrites results/)
python ab_mdp_simulation.py --figure-only              # rebuild both figures from results/ (seconds)
python ab_mdp_simulation.py --skip-study --only-eps    # re-run only the disturbance sweep
python ab_mdp_simulation.py --runs 400 --sweep-runs 150 --sweep-seeds 2   # quick smoke test
```

Only `numpy` and `matplotlib` are required. `scipy` is deliberately **not** a dependency: the
exact Clopper–Pearson intervals used throughout are implemented in-repo by continued-fraction
incomplete beta plus bisection, and validated against published table values in
`test_intervals.py`.

`--figure-only` regenerates both figures from the released `results/*.json` and
`results/trajectories.npz` without retraining. It is the recommended way to check that the
figures and the JSON agree: panel titles read the seed count **out of the data**, so a caption
cannot silently drift from the released numbers.

## Verification

| Script | What it asserts |
|---|---|
| `verify_numbers.py` | 39 checks that every number quoted in the manuscript — including the three in the abstract — matches the released JSON; plus record-level consistency (Lemma 1 against the sweep; the reward-scale and penalty-weight patterns the text claims); plus a rounding-tie screen over every quoted value |
| `make_sweep_table.py` | generates Tables S1/S2 from the JSON and fails if Lemma 1's prediction and the measured vacuity disagree |
| `test_intervals.py` | validates the in-repo exact Clopper–Pearson implementation against closed-form boundary identities, published table values, the defining tail property, and beta symmetry |

```bash
cd simulation
python test_intervals.py
python verify_numbers.py --outdir results --tex /path/to/manuscript.tex
```

These are wired into the manuscript's build, which **fails rather than warns** on any divergence
between prose and data.

## What is in this repository

```
simulation/
  ab_mdp_simulation.py     self-contained study: environment, two mechanisms, sweeps, figures
  make_sweep_table.py      Supplementary Tables S1/S2, generated from the JSON
  verify_numbers.py        manuscript-vs-records checks (39 numeric + record-level consistency)
  test_intervals.py        validation of the exact Clopper-Pearson implementation
  README.md                method notes: what the code does and why it is built this way
  requirements.txt         numpy, matplotlib
  results/
    summary.json           headline multi-run statistics, per-run breach rates, margin profile
    sweep_reward_scale.json  violation rate vs. task reward scale (per-seed arrays)
    sweep_lambda.json        violation rate vs. penalty weight (per-seed arrays)
    sweep_eps_max.json       the disturbance-bound sweep, with measured vacuous-step counts
    trajectories.npz         raw trajectories for re-analysis
    figure1.pdf/.png         four-panel figure (vector + 300 dpi)
    figure2.pdf/.png         three-panel figure, incl. the disturbance sweep
```

**Not included:** the manuscript source and the internal verification log. The code-availability
statement covers the simulation environment and the validation scripts, which is exactly what is
above; the manuscript source is withheld until the paper is accepted.

## Configuration

All parameters live in the `Config` dataclass: `a_max=0.10`, `eps_max=0.05`, `radius=1.0`,
`gamma=0.99`, `lam=10.0`, `reward_scale=2.0`, `horizon=100`, `s0=(-0.5,-0.5)`, `goal=(1.4,0.0)`,
`eta=0.20`, `train_episodes=1200`, `seed=20260917`.

## Why there is no adaptive-multiplier baseline

A primal–dual (dual-ascent) baseline was implemented and tested during preparation. It did not
converge stably: the multiplier saturated and the policy froze, so the comparison would have
confounded solver tuning with the structural question. It was removed rather than reported, and
the paper states its claim narrowly — *no instance-independent finite penalty weight* — which is
what the reward-scale and penalty-weight sweeps test.

## Citation

See `CITATION.cff`. If you use this code or these data, please cite the manuscript.

## License

MIT — see `LICENSE`.
