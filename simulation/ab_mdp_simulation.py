"""
AB-MDP vs. soft Lagrangian penalty relaxation: illustrative 2D navigation study.

Reproduces the numerical results and Figure 1 of:

    Li, W. "Ontological Bounds of Autoregressive Architectures:
    An Autonomic-Boundary MDP Framework for Physical AI" (2026).

Setup
-----
State space      : S = R^2
Viability kernel : K = { s : ||s||_2 <= 1 }
Dynamics         : s_{t+1} = s_t + a_t + eps_t,  ||a_t||_2 <= a_max, ||eps_t||_2 <= eps_max
Task goal        : s_goal = (1.4, 0.0)  (outside K; adversarial target)
Horizon          : T = 100 steps, gamma = 0.99, start s0 = (-0.5, -0.5)

Three controllers are compared on identical dynamics and disturbances:

  1. FIXED-PENALTY relaxation:  J_lambda = E[sum gamma^t r_t] - lambda E[sum gamma^t c_t]
     with a fixed lambda and per-step violation cost c_t = 1{||s_{t+1}|| > 1}.
     Trained by REINFORCE.

  2. AB-MDP: the same nominal policies filtered by the robust one-step invariance filter
         A(s) = { a : ||a|| <= a_max, ||s + a||_2 + eps_max <= 1 },
     which is exactly the condition  f(s,a) + eps in K  for all ||eps|| <= eps_max.
     Confinement is then *structural* (Proposition 2) and requires no tuning.
     The endogenous scalar sigma_pain(s) = exp(-dist(s, dK)/eta) modulates the
     exploration noise scale, (1 - sigma_pain(s)), i.e. exploratory drive is damped as
     the boundary approaches. sigma_pain is a modulation signal only: the confinement
     guarantee comes from the filter, not from sigma_pain, and this is stated as such.

Reported metrics per controller: kernel violation rate, mean task return, mean final
distance to the goal, projection activity (AB-MDP), and mean sigma_pain (AB-MDP).

Two additional measurements support the paper's Lemma 1 (admissibility margin):

  * MARGIN PROFILE. Every filtered step records the distance of the *pre-action* state
    to the boundary, whether the filter intervened, and by how much. Steps are binned
    by that distance, so the released JSON contains the filter intervention rate and
    mean displacement as a function of the viability margin. This is the empirical
    content of the claim that sigma_pain is an order-isomorphic readout of the margin:
    it is checked against the mechanism that actually carries the guarantee.

  * DISTURBANCE SWEEP. a_max > eps_max is the condition under which the one-step
    invariance filter is non-vacuous on the whole kernel (Lemma 1). Sweeping eps_max
    below a_max tests that the filter remains breach-free there, while the penalty
    baseline does not.

All proportions are reported with exact Clopper-Pearson 95% intervals (computed here,
without scipy). The between-seed spread of the penalty baseline is the object of
interest, so per-seed rates are released alongside the aggregate.

Outputs
-------
  results/summary.json            : headline multi-seed statistics + margin profile
  results/sweep_reward_scale.json : violation rate vs. reward scale (per-seed detail)
  results/sweep_lambda.json       : violation rate vs. penalty weight (per-seed detail)
  results/sweep_eps_max.json      : violation rate vs. disturbance bound (per-seed detail)
  results/figure1.pdf / .png      : leakage vs. confinement (four panels)
  results/figure2.pdf / .png      : margin readout + disturbance sweep (two panels)
  results/trajectories.npz        : representative trajectories

Usage
-----
  python ab_mdp_simulation.py                      # full study (~15 min)
  python ab_mdp_simulation.py --figure-only        # rebuild figures from released JSON
  python ab_mdp_simulation.py --runs 400 --sweep-runs 200 --sweep-seeds 2   # quick test
"""

import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Exact binomial intervals (no scipy dependency)
# --------------------------------------------------------------------------------------
#
# Reported proportions are accompanied by exact Clopper-Pearson 95% intervals. scipy is
# deliberately not a dependency of this study, so the regularized incomplete beta
# function I_x(a, b) is evaluated here via its continued fraction (Lentz's algorithm,
# as in Numerical Recipes 6.4) and inverted by bisection. This is validated in
# test_intervals.py against published Clopper-Pearson values, and by an independent
# route: for k = 0 the interval must reproduce the closed form 1 - (alpha/2)^(1/n).

def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), accurate to ~1e-14."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) - lbeta) \
        * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact Clopper-Pearson interval for a binomial proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return (float(lo), float(hi))


# --------------------------------------------------------------------------------------
# Margin binning
# --------------------------------------------------------------------------------------
#
# Filtered steps are binned by the pre-action distance to the boundary,
# dist(s, dK) = radius - ||s||. Bin b covers [EDGES[b-1], EDGES[b]); bin 0 is the
# innermost class [0, 0.05) and the final bin collects every step at margin >= 0.50.

MARGIN_EDGES = np.arange(0.05, 0.55, 0.05)
N_MARGIN_BINS = len(MARGIN_EDGES) + 1


def margin_bin_label(b: int) -> str:
    if b == 0:
        return "[0.00, %.2f)" % MARGIN_EDGES[0]
    if b >= len(MARGIN_EDGES):
        return "[%.2f, inf)" % MARGIN_EDGES[-1]
    return "[%.2f, %.2f)" % (MARGIN_EDGES[b - 1], MARGIN_EDGES[b])


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    a_max: float = 0.10           # action (velocity) bound
    eps_max: float = 0.05         # bounded environmental disturbance (Assumption 1)
    radius: float = 1.0           # viability kernel radius
    gamma: float = 0.99           # discount factor
    lam: float = 10.0             # fixed penalty weight
    reward_scale: float = 2.0     # task reward scale (instance parameter)
    horizon: int = 100            # steps per episode
    runs: int = 10_000            # evaluation episodes (headline)
    s0: tuple = (-0.50, -0.50)    # start in the viable interior (||s0|| = 0.707)
    goal: tuple = (1.40, 0.00)    # out-of-bounds adversarial goal
    eta: float = 0.20             # sigma_pain boundary sensitivity
    train_episodes: int = 1200    # REINFORCE episodes per policy
    lr: float = 0.05
    policy_sigma: float = 0.15
    seed: int = 20260917


def sigma_pain(s: np.ndarray, cfg: Config) -> float:
    """Endogenous neuromodulatory scalar, Eq. (4): exp(-dist(s, dK)/eta).

    For the unit disk, dist(s, dK) = radius - ||s||.
    """
    dist = cfg.radius - float(np.linalg.norm(s))
    return float(np.exp(-dist / cfg.eta))


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------

class NavigationEnv:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.s0 = np.asarray(cfg.s0, dtype=float)
        self.goal = np.asarray(cfg.goal, dtype=float)

    def reset(self) -> np.ndarray:
        return self.s0.copy()

    def reward(self, s: np.ndarray) -> float:
        return self.cfg.reward_scale * (1.0 - float(np.tanh(np.linalg.norm(s - self.goal))))

    def cost(self, s: np.ndarray) -> float:
        return 1.0 if np.linalg.norm(s) > self.cfg.radius else 0.0

    def disturbance(self) -> np.ndarray:
        d = self.rng.normal(size=2)
        n = float(np.linalg.norm(d))
        if n == 0.0:
            return np.zeros(2)
        return d / n * (self.cfg.eps_max * self.rng.random() ** 0.5)

    def step(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        return s + a + self.disturbance()


# --------------------------------------------------------------------------------------
# Linear Gaussian policy (REINFORCE)
# --------------------------------------------------------------------------------------

class LinearGaussianPolicy:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.W = rng.normal(scale=0.1, size=(2, 2))
        self.b = np.zeros(2)
        self._last_s = np.zeros(2)

    def mean(self, s: np.ndarray) -> np.ndarray:
        return self.cfg.a_max * np.tanh(self.W @ s + self.b)

    def sample(self, s: np.ndarray, rng: np.random.Generator, noise_scale: float = 1.0):
        mu = self.mean(s)
        a = mu + self.cfg.policy_sigma * noise_scale * rng.normal(size=2)
        n = float(np.linalg.norm(a))
        if n > self.cfg.a_max:
            a = a / n * self.cfg.a_max
        return a, mu

    def log_prob_grad(self, a: np.ndarray, mu: np.ndarray, noise_scale: float = 1.0) -> np.ndarray:
        raw = np.arctanh(np.clip(mu / self.cfg.a_max, -0.999, 0.999))
        d_mu_d_raw = self.cfg.a_max * (1.0 - np.tanh(raw) ** 2)
        sigma2 = (self.cfg.policy_sigma * noise_scale) ** 2
        resid = (a - mu) / sigma2
        dW = np.outer(resid * d_mu_d_raw, self._last_s)
        db = resid * d_mu_d_raw
        return np.concatenate([dW.ravel(), db])

    def apply_grad(self, grad: np.ndarray, lr: float):
        self.W += lr * grad[:4].reshape(2, 2)
        self.b += lr * grad[4:]

    def set_state(self, s: np.ndarray):
        self._last_s = s.copy()


# --------------------------------------------------------------------------------------
# The robust invariance filter
# --------------------------------------------------------------------------------------

def admissible_margin(s: np.ndarray, cfg: Config) -> float:
    """The admissibility margin m(s) of Lemma 1, in closed form for this instance.

    A(s) = {a : ||a|| <= a_max}  n  {a : ||s + a|| <= radius - eps_max}
    is an intersection of two closed balls, B(0, a_max) and B(-s, radius - eps_max).
    Two closed balls intersect iff the distance between their centres does not exceed
    the sum of their radii:

        ||s|| <= a_max + (radius - eps_max)
        <=>  (radius - ||s||) - (eps_max - a_max) >= 0

    so with dist(s, dK) = radius - ||s|| the margin is exactly

        m(s) = dist(s, dK) - (eps_max - a_max),

    which is the quantity Lemma 1 identifies as deciding non-vacuity. This is used
    to *measure* vacuity rather than infer it from a failure to project: Dykstra's
    alternating projection is only meaningful on a non-empty intersection, so when
    the two balls are disjoint it has no fixed point to converge to and its iterate
    is not a projection at all. Detecting that case explicitly keeps the reported
    "breaches" attributable to the right cause.
    """
    return (cfg.radius - float(np.linalg.norm(s))) - (cfg.eps_max - cfg.a_max)


def maximal_inward_action(s: np.ndarray, cfg: Config) -> np.ndarray:
    """Least-bad action when A(s) is empty: move inward at full authority.

    Used only on vacuous steps, where no admissible action exists by definition.
    Choosing the maximally inward action is the most favourable possible response
    available to the agent, so any breach that follows is attributable to the eroded
    kernel and not to a perverse fallback rule.
    """
    n = float(np.linalg.norm(s))
    if n <= 1e-12:
        return np.zeros_like(s)
    return s * (-cfg.a_max / n)


def project_onto_admissible(a_des: np.ndarray, s: np.ndarray, cfg: Config) -> np.ndarray:
    """Euclidean projection of a_des onto A(s) = {a : ||a||<=a_max, ||s+a||<=radius-eps_max}.

    The admissible set is an intersection of two Euclidean balls (convex); the projection
    is computed with Dykstra's alternating projection algorithm, which converges to the
    exact projection onto an intersection of convex sets. The caller must have checked
    that the intersection is non-empty (see admissible_margin); on a vacuous step there
    is nothing to project onto.
    """
    r_inner = cfg.radius - cfg.eps_max

    def proj_act(z):
        n = float(np.linalg.norm(z))
        return z if n <= cfg.a_max else z * (cfg.a_max / n)

    def proj_next(z):
        v = z + s
        n = float(np.linalg.norm(v))
        return z if n <= r_inner else (v * (r_inner / n)) - s

    x = a_des.copy()
    p = np.zeros(2)
    q = np.zeros(2)
    for _ in range(60):
        y = proj_act(x + p)
        p = x + p - y
        x_new = proj_next(y + q)
        q = y + q - x_new
        if float(np.linalg.norm(x_new - x)) < 1e-11:
            x = x_new
            break
        x = x_new
    return x


# --------------------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------------------

def rollout(s_cfg: Config, policy: LinearGaussianPolicy, env: NavigationEnv,
            rng: np.random.Generator, mode: str):
    """mode in {'raw', 'filtered'}. Returns a metrics dict."""
    cfg = s_cfg
    s = env.reset()
    traj = [s.copy()]
    viol = []
    returns = 0.0
    proj_active, proj_norm, pain_vals = 0, 0.0, []
    vacuous = 0
    rec_dist, rec_disp, rec_act, rec_sp = [], [], [], []
    rec_vac, rec_margin = [], []

    for _ in range(cfg.horizon):
        policy.set_state(s)
        if mode == "filtered":
            sp = sigma_pain(s, cfg)
            pain_vals.append(sp)
            a_nom, _ = policy.sample(s, rng, noise_scale=1.0 - sp)
            # Lemma 1: the filter is non-vacuous iff m(s) >= 0. When m(s) < 0 the
            # admissible set is empty and there is no projection to compute, so we
            # record the step as vacuous and take the maximally inward action.
            m_s = admissible_margin(s, cfg)
            is_vac = m_s < 0.0
            if is_vac:
                a = maximal_inward_action(s, cfg)
                vacuous += 1
            else:
                a = project_onto_admissible(a_nom, s, cfg)
            d = float(np.linalg.norm(a - a_nom))
            rec_dist.append(cfg.radius - float(np.linalg.norm(s)))
            rec_disp.append(d)
            rec_act.append(1 if (not is_vac and d > 1e-6) else 0)
            rec_sp.append(sp)
            rec_vac.append(1 if is_vac else 0)
            rec_margin.append(m_s)
            if not is_vac and d > 1e-6:
                proj_active += 1
                proj_norm += d
        else:
            a, _ = policy.sample(s, rng)
        s = env.step(s, a)
        traj.append(s.copy())
        returns += env.reward(s)          # undiscounted task return
        viol.append(float(np.linalg.norm(s) > cfg.radius))

    viol = np.asarray(viol)
    return {
        "trajectory": np.asarray(traj),
        "breached": bool(viol.any()),
        "breach_steps": int(viol.sum()),
        "task_return": returns,
        "final_dist": float(np.linalg.norm(traj[-1] - env.goal)),
        "proj_active_frac": proj_active / cfg.horizon,
        "proj_norm_sum": float(proj_norm),
        "proj_active_count": int(proj_active),
        "vacuous_steps": int(vacuous),
        "pain_mean": float(np.mean(pain_vals)) if pain_vals else float("nan"),
        "rec_dist": np.asarray(rec_dist),
        "rec_disp": np.asarray(rec_disp),
        "rec_act": np.asarray(rec_act, dtype=np.int64),
        "rec_sp": np.asarray(rec_sp),
        "rec_vac": np.asarray(rec_vac, dtype=np.int64),
        "rec_margin": np.asarray(rec_margin),
        # Breach indicator aligned with the *action* step that produced it, so a
        # breach can be attributed to whether the step it followed was vacuous.
        "rec_viol": viol.astype(np.int64),
    }


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def train_policy(cfg: Config, policy: LinearGaussianPolicy, env: NavigationEnv,
                 rng: np.random.Generator, lam: float):
    """REINFORCE with *separate* policy-gradient estimates for reward and cost.

    The update is the standard Lagrangian form  grad = PG_R - lambda * PG_C,  where PG_R
    and PG_C are reward- and cost-to-go weighted policy gradients. Keeping the two
    estimates separate makes lambda behave as a genuine multiplier (its influence scales
    linearly with lambda) instead of being attenuated by a lambda-dependent normalizer.
    The normalizer is independent of lambda and shrinks updates as the reward scale grows
    --- a conservative choice that weakens, rather than strengthens, the penalty baseline
    at large reward scales.
    """
    lr0 = cfg.lr
    for ep in range(cfg.train_episodes):
        cfg.lr = lr0 * (1.0 - 0.5 * ep / cfg.train_episodes)
        s = env.reset()
        steps = []
        for _ in range(cfg.horizon):
            policy.set_state(s)
            a, mu = policy.sample(s, rng)
            s_next = env.step(s, a)
            steps.append((a.copy(), mu.copy(), env.reward(s_next), env.cost(s_next)))
            s = s_next

        T = len(steps)
        G_R = np.zeros(T)
        G_C = np.zeros(T)
        acc_r = acc_c = 0.0
        for t in range(T - 1, -1, -1):
            acc_r = steps[t][2] + cfg.gamma * acc_r
            acc_c = steps[t][3] + cfg.gamma * acc_c
            G_R[t], G_C[t] = acc_r, acc_c

        n_R = 1.0 + cfg.reward_scale
        n_C = 2.0
        pg_R = np.zeros(6)
        pg_C = np.zeros(6)
        for t, (a, mu, _, _) in enumerate(steps):
            g = policy.log_prob_grad(a, mu)
            pg_R += G_R[t] * g
            pg_C += G_C[t] * g
        pg_R /= (T * n_R)
        pg_C /= (T * n_C)

        policy.apply_grad(pg_R - lam * pg_C, cfg.lr)

    cfg.lr = lr0
    return policy


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------

def evaluate(cfg: Config, policy: LinearGaussianPolicy, env: NavigationEnv,
             rng: np.random.Generator, runs: int, mode: str):
    n_breach, breach_steps, ret, fdist, pa, pn, pac, pain = 0, 0, 0.0, 0.0, 0.0, 0.0, 0, []
    bin_cnt = np.zeros(N_MARGIN_BINS)
    bin_act = np.zeros(N_MARGIN_BINS)
    bin_disp = np.zeros(N_MARGIN_BINS)
    bin_sp = np.zeros(N_MARGIN_BINS)
    vacuous_total = 0
    breach_on_vacuous = 0
    min_margin = float("inf")

    for _ in range(runs):
        m = rollout(cfg, policy, env, rng, mode)
        n_breach += int(m["breached"])
        breach_steps += m["breach_steps"]
        ret += m["task_return"]
        fdist += m["final_dist"]
        pa += m["proj_active_frac"]
        pn += m["proj_norm_sum"]
        pac += m["proj_active_count"]
        if not np.isnan(m["pain_mean"]):
            pain.append(m["pain_mean"])

        vac = m["rec_vac"]
        if vac.size:
            vacuous_total += int(vac.sum())
            min_margin = min(min_margin, float(m["rec_margin"].min()))
            breach_on_vacuous += int((m["rec_viol"] & vac).sum())

        dist = m["rec_dist"]
        if dist.size:
            idx = np.digitize(dist, MARGIN_EDGES)
            for b in range(N_MARGIN_BINS):
                sel = idx == b
                c = int(sel.sum())
                if c:
                    bin_cnt[b] += c
                    bin_act[b] += int(m["rec_act"][sel].sum())
                    bin_disp[b] += float(m["rec_disp"][sel].sum())
                    bin_sp[b] += float(m["rec_sp"][sel].sum())

    profile = []
    for b in range(N_MARGIN_BINS):
        profile.append({
            "bin": b,
            "range": margin_bin_label(b),
            "steps": int(bin_cnt[b]),
            "active_frac": float(bin_act[b] / bin_cnt[b]) if bin_cnt[b] else None,
            # mean_disp_active pools over steps on which the filter actually acted;
            # mean_disp_per_step pools over every step in the bin. Reporting the pooled
            # ratio matters: averaging per-episode ratios silently injects a zero for
            # every episode in which the filter never intervened, which understates the
            # correction magnitude.
            "mean_disp_active": float(bin_disp[b] / bin_act[b]) if bin_act[b] else None,
            "mean_disp_per_step": float(bin_disp[b] / bin_cnt[b]) if bin_cnt[b] else None,
            "mean_sigma_pain": float(bin_sp[b] / bin_cnt[b]) if bin_cnt[b] else None,
        })

    return {
        "runs": runs,
        "breach_rate": n_breach / runs,
        "breach_rate_ci95_exact": list(clopper_pearson(n_breach, runs)),
        "breaches": n_breach,
        "breach_steps": breach_steps,
        "evaluated_steps": runs * cfg.horizon,
        "mean_task_return": ret / runs,
        "mean_final_dist": fdist / runs,
        "mean_proj_active_frac": pa / runs,
        "proj_active_steps": pac,
        # Pooled over the steps on which the filter acted. This is deliberately not the
        # mean of per-episode (or per-seed) ratios: those are unweighted averages of
        # ratios and drift from the pooled quantity whenever episodes differ in how often
        # the filter acts. The pooled form is the one the manuscript quotes.
        "mean_proj_norm": (pn / pac) if pac else 0.0,
        "mean_sigma_pain": float(np.mean(pain)) if pain else None,
        "vacuous_steps": vacuous_total,
        "vacuous_step_frac": vacuous_total / (runs * cfg.horizon),
        "breach_steps_on_vacuous": breach_on_vacuous,
        "vacuous_breach_rate": (breach_on_vacuous / vacuous_total) if vacuous_total else None,
        "nonvacuous_breach_steps": breach_steps - breach_on_vacuous,
        "min_admissibility_margin": min_margin if vacuous_total or min_margin != float("inf")
        else None,
        "margin_profile": profile,
    }


def run_study(cfg: Config, outdir: str, headline_seeds: int = 15, verbose: bool = True):
    """Multi-seed study of the fixed-penalty relaxation vs. the AB-MDP filter.

    Both controllers are evaluated on the *same* trained policy, which isolates the
    effect that is being studied (whether the admissible set is restricted). Adaptive
    multiplier methods are deliberately *not* used as baselines: a fair primal-dual
    implementation itself requires per-instance tuning of the multiplier dynamics, and
    the resulting comparison would confound solver tuning with the structural question.
    We therefore state the theoretical claim narrowly (no instance-independent finite
    penalty weight) and report the penalty-weight sweep instead (see run_sweep).

    Statistics are reported per seed as well as aggregated, because training of the
    penalty baseline is itself stochastic and the seed-to-seed spread is informative.
    """
    os.makedirs(outdir, exist_ok=True)
    eval_runs = max(1, cfg.runs // headline_seeds)
    per_seed = []

    for k in range(headline_seeds):
        rng = np.random.default_rng(cfg.seed + k)
        env = NavigationEnv(cfg, rng)
        policy = train_policy(cfg, LinearGaussianPolicy(cfg, rng), env, rng, cfg.lam)
        raw = evaluate(cfg, policy, env, rng, eval_runs, "raw")
        filt = evaluate(cfg, policy, env, rng, eval_runs, "filtered")
        per_seed.append({"seed": cfg.seed + k, "penalty": raw, "abmdp": filt})
        if verbose:
            print(f"  seed {cfg.seed + k}: penalty breach {raw['breach_rate']:.3f} "
                  f"(return {raw['mean_task_return']:.1f}) | filtered breaches "
                  f"{filt['breach_steps']} (return {filt['mean_task_return']:.1f})")

    rates = np.array([r["penalty"]["breach_rate"] for r in per_seed])
    ab_steps = sum(r["abmdp"]["evaluated_steps"] for r in per_seed)
    ab_breaches = sum(r["abmdp"]["breach_steps"] for r in per_seed)

    proj_act_total = sum(r["abmdp"]["proj_active_steps"] for r in per_seed)
    proj_disp_total = sum(r["abmdp"]["mean_proj_norm"] * r["abmdp"]["proj_active_steps"]
                          for r in per_seed)
    proj_disp_pooled = (proj_disp_total / proj_act_total) if proj_act_total else 0.0

    # Run-level heterogeneity: the unit of analysis is the *training run*, because the
    # question posed by Proposition 1 is whether the penalty weight holds, and that is a
    # property of the optimisation run rather than of the episode. A run is called
    # "leaky" if its policy breaches in at least half of its evaluation episodes.
    n_leaky = int((rates >= 0.5).sum())
    n_clean = int((rates == 0.0).sum())

    # Margin profile aggregated over seeds (counts are summed, then ratios formed).
    agg = [{"bin": b, "range": margin_bin_label(b), "steps": 0, "act": 0,
            "disp": 0.0, "sp": 0.0} for b in range(N_MARGIN_BINS)]
    for rec in per_seed:
        for row in rec["abmdp"]["margin_profile"]:
            b = row["bin"]
            agg[b]["steps"] += row["steps"]
            agg[b]["act"] += int(round((row["active_frac"] or 0.0) * row["steps"]))
            if row["steps"]:
                agg[b]["disp"] += (row["mean_disp_per_step"] or 0.0) * row["steps"]
                agg[b]["sp"] += (row["mean_sigma_pain"] or 0.0) * row["steps"]
    margin_profile = [{
        "bin": a["bin"],
        "range": a["range"],
        "steps": a["steps"],
        "active_steps": a["act"],
        "active_frac": (a["act"] / a["steps"]) if a["steps"] else None,
        "mean_disp_active": (a["disp"] / a["act"]) if a["act"] else None,
        "mean_disp_per_step": (a["disp"] / a["steps"]) if a["steps"] else None,
        "mean_sigma_pain": (a["sp"] / a["steps"]) if a["steps"] else None,
    } for a in agg]

    summary = {
        "config": asdict(cfg),
        "headline_seeds": headline_seeds,
        "eval_runs_per_seed": eval_runs,
        "fixed_penalty": {
            "lambda": cfg.lam,
            "breach_rate_mean": float(rates.mean()),
            "breach_rate_std": float(rates.std()),
            "breach_rate_median": float(np.median(rates)),
            "breach_rate_per_seed": rates.tolist(),
            "min": float(rates.min()),
            "max": float(rates.max()),
            "leaky_runs": n_leaky,
            "clean_runs": n_clean,
            "leaky_runs_frac": n_leaky / headline_seeds,
            "leaky_runs_frac_ci95_exact": list(clopper_pearson(n_leaky, headline_seeds)),
            "mean_task_return": float(np.mean([r["penalty"]["mean_task_return"]
                                               for r in per_seed])),
            "mean_final_dist": float(np.mean([r["penalty"]["mean_final_dist"]
                                              for r in per_seed])),
        },
        "abmdp": {
            "evaluated_steps": ab_steps,
            "breach_steps_total": ab_breaches,
            "breach_rate": ab_breaches / ab_steps,
            "breach_rate_ci95_exact": list(clopper_pearson(ab_breaches, ab_steps)),
            "mean_task_return": float(np.mean([r["abmdp"]["mean_task_return"]
                                               for r in per_seed])),
            "mean_final_dist": float(np.mean([r["abmdp"]["mean_final_dist"]
                                              for r in per_seed])),
            "mean_proj_active_frac": float(np.mean([r["abmdp"]["mean_proj_active_frac"]
                                                    for r in per_seed])),
            # Pooled over every step on which the filter acted, across all seeds. Reported
            # alongside the mean over seeds, which is an unweighted average of per-run
            # ratios and therefore differs from the pooled value whenever runs differ in
            # how frequently the filter acts. Both are released; the pooled form is quoted.
            "mean_proj_norm_pooled": proj_disp_pooled,
            "mean_proj_norm_seed_mean": float(np.mean([r["abmdp"]["mean_proj_norm"]
                                                       for r in per_seed])),
            "proj_active_steps_total": sum(r["abmdp"]["proj_active_steps"]
                                           for r in per_seed),
            "seeds_without_intervention": int(sum(
                1 for r in per_seed if r["abmdp"]["proj_active_steps"] == 0)),
            "mean_sigma_pain": float(np.mean([r["abmdp"]["mean_sigma_pain"]
                                              for r in per_seed])),
        },
        "margin_profile": margin_profile,
        "per_seed": per_seed,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("per_seed", "margin_profile")}, indent=2))
        print("margin profile (distance to boundary -> filter behaviour):")
        for row in summary["margin_profile"]:
            if row["steps"]:
                da = row["mean_disp_active"]
                print(f"  {row['range']:>14s}  steps={row['steps']:8d}  "
                      f"active={row['active_frac']:.3f}  "
                      f"|delta_a|_active={'n/a  ' if da is None else f'{da:.4f}'}  "
                      f"|delta_a|_per_step={row['mean_disp_per_step']:.4f}  "
                      f"sigma_pain={row['mean_sigma_pain']:.3f}")

    # representative trajectories from a fresh policy at the default configuration
    rng = np.random.default_rng(cfg.seed + 999)
    env = NavigationEnv(cfg, rng)
    pol = train_policy(cfg, LinearGaussianPolicy(cfg, rng), env, rng, cfg.lam)
    pen_t, ab_t = [], []
    for _ in range(40):
        pen_t.append(rollout(cfg, pol, env, rng, "raw")["trajectory"])
        ab_t.append(rollout(cfg, pol, env, rng, "filtered")["trajectory"])
    np.savez_compressed(os.path.join(outdir, "trajectories.npz"),
                        penalty=np.asarray(pen_t), abmdp=np.asarray(ab_t))
    return summary, np.asarray(pen_t), np.asarray(ab_t)


def run_sweep(cfg: Config, outdir: str, kind: str, values, sweep_runs: int, train_seeds: int):
    """Sweep one instance parameter, evaluating the raw (unfiltered) and the filtered
    controller built on the *same* trained policy.

    kind='reward_scale' : reward scale varied, lambda fixed
    kind='lambda'       : penalty weight varied, reward scale fixed
    kind='eps_max'      : disturbance bound varied (tests Lemma 1's a_max > eps_max
                          non-vacuity condition), lambda and reward scale fixed
    """
    rows = []
    for v in values:
        rates_raw, rates_filt = [], []
        filt_breach_steps = 0
        filt_steps = 0
        proj_active = []
        vac_steps = 0
        vac_breach_steps = 0
        for k in range(train_seeds):
            rng = np.random.default_rng(cfg.seed + 7919 * (k + 1) + int(v * 1000))
            if kind == "reward_scale":
                cfg_i = dataclasses.replace(cfg, reward_scale=v)
            elif kind == "lambda":
                cfg_i = dataclasses.replace(cfg, lam=v)
            elif kind == "eps_max":
                cfg_i = dataclasses.replace(cfg, eps_max=v)
            else:
                raise ValueError(f"unknown sweep kind: {kind}")
            env = NavigationEnv(cfg_i, rng)
            pol = train_policy(cfg_i, LinearGaussianPolicy(cfg_i, rng), env, rng, cfg_i.lam)
            raw = evaluate(cfg_i, pol, env, rng, sweep_runs, "raw")
            filt = evaluate(cfg_i, pol, env, rng, sweep_runs, "filtered")
            rates_raw.append(raw["breach_rate"])
            rates_filt.append(filt["breach_rate"])
            filt_breach_steps += filt["breach_steps"]
            filt_steps += filt["evaluated_steps"]
            proj_active.append(filt["mean_proj_active_frac"])
            vac_steps += filt["vacuous_steps"]
            vac_breach_steps += filt["breach_steps_on_vacuous"]
        n_leaky = int(sum(1 for r in rates_raw if r >= 0.5))
        rows.append({
            "parameter": kind,
            "value": v,
            "fixed_lambda": cfg.lam,
            "fixed_reward_scale": cfg.reward_scale,
            "fixed_eps_max": cfg.eps_max,
            "sweep_runs": sweep_runs,
            "train_seeds": train_seeds,
            "penalty_breach_rate": float(np.mean(rates_raw)),
            "penalty_breach_rate_std": float(np.std(rates_raw)),
            "penalty_breach_rate_per_seed": rates_raw,
            "penalty_leaky_runs": n_leaky,
            "penalty_leaky_runs_frac": n_leaky / train_seeds,
            "penalty_leaky_runs_frac_ci95_exact": list(clopper_pearson(n_leaky, train_seeds)),
            "abmdp_breach_rate": float(np.mean(rates_filt)),
            "abmdp_breach_rate_per_seed": rates_filt,
            "abmdp_breach_steps_total": filt_breach_steps,
            "abmdp_evaluated_steps": filt_steps,
            "abmdp_breach_rate_pooled_ci95_exact":
                list(clopper_pearson(filt_breach_steps, filt_steps)),
            "abmdp_mean_proj_active_frac": float(np.mean(proj_active)),
            "abmdp_vacuous_steps_total": vac_steps,
            "abmdp_vacuous_step_frac": vac_steps / filt_steps,
            "abmdp_vacuous_breach_steps": vac_breach_steps,
            "abmdp_breach_steps_on_nonvacuous": filt_breach_steps - vac_breach_steps,
            # The theoretical threshold of Lemma 1, so the table cannot be read
            # without it: non-vacuity fails exactly when eps_max > a_max.
            "predicted_vacuous": bool(v > cfg.a_max),
        })
        print(f"  {kind}={v:8.3f}  penalty breach={np.mean(rates_raw):6.3f}"
              f" +-{np.std(rates_raw):.3f}   AB-MDP breach steps={filt_breach_steps}"
              f" / {filt_steps} (on vacuous steps: {vac_breach_steps})"
              f"   filter active={np.mean(proj_active):.3f}"
              f"   vacuous={vac_steps}")
    out = os.path.join(outdir, f"sweep_{kind}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
#
# Both figures are laid out at their final print width (\textwidth of the manuscript
# under 1-inch A4 margins is ~6.3 in), so the nominal font sizes below are the font sizes
# the reader actually sees. Laying out wider and letting LaTeX scale the figure down
# silently shrinks every label.

FIG_WIDTH = 7.2


def _params(base: float):
    plt.rcParams.update({"font.size": base, "axes.linewidth": 0.7,
                         "font.family": "DejaVu Sans", "axes.titlesize": base + 0.6,
                         "xtick.labelsize": base - 0.4, "ytick.labelsize": base - 0.4,
                         "legend.fontsize": base - 1.2})


def make_figure1(cfg: Config, penalty_trajs: np.ndarray, abmdp_trajs: np.ndarray,
                 sweep_rs, sweep_lam, outdir: str):
    _params(7.6)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 6.9))
    theta = np.linspace(0, 2 * np.pi, 400)
    bx, by = cfg.radius * np.cos(theta), cfg.radius * np.sin(theta)

    for ax, trajs, title, color, show_breach in (
        (axes[0, 0], penalty_trajs,
         r"(a) Fixed penalty relaxation ($\lambda = %.0f$)" % cfg.lam, "#c0392b", True),
        (axes[0, 1], abmdp_trajs, "(b) AB-MDP invariance filter", "#1f6fb4", False),
    ):
        ax.fill(bx, by, color="#eef6ee", zorder=0)
        ax.plot(bx, by, "--", color="#c0392b", lw=1.2, zorder=1,
                label=r"Boundary $\partial\mathcal{K}$")
        for tr in trajs:
            ax.plot(tr[:, 0], tr[:, 1], color=color, lw=0.6, alpha=0.32, zorder=2)
            if show_breach:
                out = np.linalg.norm(tr, axis=1) > cfg.radius
                if out.any():
                    ax.plot(tr[out, 0], tr[out, 1], ".", color="#c0392b", ms=2.0,
                            alpha=0.6, zorder=3)
        ax.plot(trajs[0][:, 0], trajs[0][:, 1], color=color, lw=1.6, zorder=4,
                label="Representative trajectory")
        ax.plot(*cfg.s0, "o", color="#2c3e50", ms=4.5, zorder=5, label=r"Start $s_0$")
        ax.plot(*cfg.goal, "*", color="#8e44ad", ms=11, zorder=6,
                label=r"Out-of-bounds goal")
        ax.set_xlim(-1.35, 1.75); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$s^{(1)}$"); ax.set_ylabel(r"$s^{(2)}$")
        ax.set_title(title)
        ax.grid(alpha=0.15, lw=0.5)
        ax.legend(loc="lower left", framealpha=0.92)

    axes[0, 0].annotate("kernel breached\n(Proposition 1)", xy=(1.05, 0.25),
                        xytext=(1.20, 0.98), fontsize=6.6, color="#c0392b", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))
    axes[0, 1].annotate("confinement within $\\mathcal{K}$\n(Proposition 3)",
                        xy=(0.70, 0.72), xytext=(0.30, 1.24), fontsize=6.6,
                        color="#1f6fb4", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#1f6fb4", lw=0.9))

    for ax, rows, xlabel, tag in (
        (axes[1, 0], sweep_rs, "Task reward scale " + r"$\bar{R}$", "(c)"),
        (axes[1, 1], sweep_lam, r"Penalty weight $\lambda$", "(d)"),
    ):
        xs = [r["value"] for r in rows]
        pen = [r["penalty_breach_rate"] for r in rows]
        err = [r["penalty_breach_rate_std"] for r in rows]
        ab = [r["abmdp_breach_rate"] for r in rows]
        ax.errorbar(xs, pen, yerr=err, fmt="o-", color="#c0392b", lw=1.3, ms=3.8,
                    capsize=2.5, label="Penalty relaxation")
        ax.plot(xs, ab, "s--", color="#1f6fb4", lw=1.3, ms=3.8,
                label="AB-MDP (invariance filter)")
        ax.set_xscale("log"); ax.set_ylim(-0.06, 1.06)
        ax.set_xlabel(xlabel); ax.set_ylabel("Kernel violation rate")
        ax.grid(alpha=0.2, lw=0.5); ax.legend(loc="center right")
        # Seed counts are read from the data, never hard-coded: a caption that disagrees
        # with the released JSON is the single most damaging kind of erratum.
        n_seeds = rows[0].get("train_seeds")
        n_runs = rows[0].get("sweep_runs")
        seed_tag = ("%d seeds x %s episodes" % (n_seeds, f"{n_runs:,}")
                    if n_seeds and n_runs else "seed count n/a")
        if tag == "(c)":
            ax.set_title("(c) Violation vs. reward scale\n(%s)" % seed_tag)
        else:
            ax.set_title("(d) Violation vs. penalty weight\n(%s; "
                         r"$\bar{R}=%.0f$)" % (seed_tag, cfg.reward_scale))

    fig.tight_layout()
    pdf = os.path.join(outdir, "figure1.pdf")
    png = os.path.join(outdir, "figure1.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {pdf}, {png}")
    return pdf


def make_figure2(margin_profile, sweep_eps, cfg: Config, outdir: str):
    """Margin readout and the disturbance test of Lemma 1.

    Panel (a) overlays the measured filter intervention rate with sigma_pain on a common
    [0,1] axis, binned by the pre-action distance to the boundary. Lemma 1 identifies
    sigma_pain with an order-isomorphic readout of the admissibility margin, so the two
    curves are predicted to be monotonically related; the figure is the check.

    Panel (b) shows the mean displacement the filter applies, against the actuation bound.

    Panel (c) is the decisive one. Lemma 1 says the admissible set is non-empty iff
    a_max > eps_max, so sweeping the disturbance bound across a_max should produce a
    sharp transition: no breaches and no vacuous steps below the threshold, and both
    appearing immediately above it. Plotting the sweep is what turns the lemma from an
    assumption into something the reader can see fail.
    """
    _params(7.3)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.85))

    rows = [r for r in margin_profile
            if r["steps"] and r["bin"] < len(MARGIN_EDGES) and r["active_steps"]]
    if rows:
        x = [((r["bin"] + 0.5) * 0.05) for r in rows]
        act = [r["active_frac"] for r in rows]
        sig = [r["mean_sigma_pain"] for r in rows]
        disp = [r["mean_disp_active"] for r in rows]

        ax = axes[0]
        ax.plot(x, act, "o-", color="#1f6fb4", lw=1.4, ms=4.5,
                label="Filter intervened")
        ax.plot(x, sig, "s--", color="#d68910", lw=1.4, ms=3.6,
                label=r"$\sigma_{\mathrm{pain}}$")
        ax.set_xlabel(r"Distance to boundary  $\mathrm{dist}(s,\partial\mathcal{K})$")
        ax.set_ylabel("Fraction of steps / scalar")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2, lw=0.5); ax.legend(loc="upper right")
        ax.set_title("(a) Certified margin readout\nvs. measured mechanism")

        ax = axes[1]
        ax.plot(x, disp, "o-", color="#1f6fb4", lw=1.4, ms=4.2)
        ax.axhline(cfg.a_max, color="#7f8c8d", ls=":", lw=1.1,
                   label=r"Scale reference $a_{\max}$")
        ax.set_xlabel(r"Distance to boundary  $\mathrm{dist}(s,\partial\mathcal{K})$")
        ax.set_ylabel(r"Mean correction  $\|a - a_{\mathrm{nom}}\|_2$" "\n"
                      r"(steps on which the filter acted)")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.2, lw=0.5); ax.legend(loc="upper right")
        ax.set_title("(b) Size of the corrective action")

    if sweep_eps:
        # Sort by the swept value; the JSON is written in grid order but do not rely on it.
        sw = sorted(sweep_eps, key=lambda r: r["value"])
        xe = [r["value"] for r in sw]
        # Breaches per 10^5 evaluated steps. Zeros cannot be shown on a log axis, so
        # they are re-drawn as open markers on the floor and labelled as exact zeros:
        # the fact that the count is exactly zero below threshold is the result.
        floor = 0.3
        rate = [max(r["abmdp_breach_steps_total"] / r["abmdp_evaluated_steps"] * 1e5, floor)
                for r in sw]
        zero_at = [i for i, r in enumerate(sw) if r["abmdp_breach_steps_total"] == 0]
        vac = [100.0 * r.get("abmdp_vacuous_step_frac", 0.0) for r in sw]

        ax = axes[2]
        ax.axvspan(cfg.a_max, max(xe) * 1.06, color="#e74c3c", alpha=0.07, lw=0)
        ax.axvline(cfg.a_max, color="#c0392b", ls="--", lw=1.2,
                   label=r"$a_{\max}=%.2f$ (Lemma 1 threshold)" % cfg.a_max)
        ax.plot(xe, rate, "o-", color="#1f6fb4", lw=1.4, ms=4.2,
                label="Breaches / $10^5$ steps")
        ax.plot([xe[i] for i in zero_at], [rate[i] for i in zero_at], "o",
                mfc="white", mec="#1f6fb4", ms=5.0, mew=1.2, zorder=5)
        ax.plot(xe, vac, "s--", color="#d68910", lw=1.3, ms=3.6,
                label="Vacuous steps (\\%)")
        ax.set_yscale("log")
        # The y-range is derived from the data rather than fixed. A hard-coded top of
        # 200.0 clipped the two largest breach rates off the top of the panel, hiding
        # exactly the points the sweep exists to show.
        hi = max(max(rate), max(vac), 1.0)
        ax.set_ylim(floor * 0.7, hi * 3.0)
        ax.set_xlabel(r"Disturbance bound  $\varepsilon_{\max}$")
        ax.set_ylabel(r"Per $10^5$ steps  /  percent" "\n" "(log scale)")
        ax.grid(alpha=0.2, lw=0.5, which="both")
        # Upper left is the empty quadrant: both series sit on the floor for every
        # eps_max below the threshold, so a lower-left legend covers the open circles
        # that mark those exact zeros -- which is the result the panel is showing.
        ax.legend(loc="upper left", fontsize=5.6, framealpha=0.95)
        n_seeds = sw[0].get("train_seeds")
        n_runs = sw[0].get("sweep_runs")
        tag = ("%d seeds $\\times$ %s episodes" % (n_seeds, f"{n_runs:,}")
               if n_seeds and n_runs else "seed count n/a")
        ax.set_title("(c) Does the kernel erode where\nLemma 1 says?  (%s)" % tag)
        if zero_at:
            ax.annotate("exactly zero", xy=(xe[zero_at[-1]], floor),
                        xytext=(xe[zero_at[0]] + 0.012, floor * 2.6), fontsize=5.6,
                        color="#1f6fb4",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#1f6fb4"))

    fig.tight_layout()
    pdf = os.path.join(outdir, "figure2.pdf")
    png = os.path.join(outdir, "figure2.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {pdf}, {png}")
    return pdf


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=30_000)
    ap.add_argument("--headline-seeds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument("--sweep-runs", type=int, default=800)
    ap.add_argument("--sweep-seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-study", action="store_true",
                    help="reuse results/summary.json + results/trajectories.npz and "
                         "re-run only the sweeps and the figures")
    ap.add_argument("--only-eps", action="store_true",
                    help="with --skip-study: re-run only the disturbance-bound sweep "
                         "and figure 2")
    ap.add_argument("--figure-only", action="store_true",
                    help="rebuild figure1/figure2 from results/*.json + trajectories.npz "
                         "without retraining (deterministic, seconds)")
    args = ap.parse_args()

    cfg = Config(runs=args.runs, horizon=args.steps, lam=args.lam,
                 reward_scale=args.reward_scale, seed=args.seed)

    def load(name):
        with open(os.path.join(args.outdir, name)) as f:
            return json.load(f)

    if args.figure_only:
        npz = np.load(os.path.join(args.outdir, "trajectories.npz"))
        summary = load("summary.json")
        rs_rows = load("sweep_reward_scale.json")
        lam_rows = load("sweep_lambda.json")
        eps_rows = load("sweep_eps_max.json")
        make_figure1(cfg, npz["penalty"], npz["abmdp"], rs_rows, lam_rows, args.outdir)
        make_figure2(summary["margin_profile"], eps_rows, cfg, args.outdir)
        print("done (figure-only)")
        return

    if args.skip_study:
        summary = load("summary.json")
        npz = np.load(os.path.join(args.outdir, "trajectories.npz"))
        pen_t, ab_t = npz["penalty"], npz["abmdp"]
        print("reusing summary.json and trajectories.npz (--skip-study)")
    else:
        summary, pen_t, ab_t = run_study(cfg, args.outdir,
                                         headline_seeds=args.headline_seeds)

    rs_rows, lam_rows, eps_rows = [], [], []
    if args.only_eps:
        print("sweeping disturbance bound (only) ...")
        eps_rows = run_sweep(cfg, args.outdir, "eps_max",
                             (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30),
                             args.sweep_runs, args.sweep_seeds)
        make_figure2(summary["margin_profile"], eps_rows, cfg, args.outdir)
        print("done (eps sweep only)")
        return
    if not args.skip_sweep:
        print("sweeping reward scale ...")
        rs_rows = run_sweep(cfg, args.outdir, "reward_scale",
                            (1.0, 2.0, 4.0, 8.0, 16.0), args.sweep_runs, args.sweep_seeds)
        print("sweeping penalty weight ...")
        lam_rows = run_sweep(cfg, args.outdir, "lambda",
                             (1.0, 5.0, 10.0, 50.0, 200.0), args.sweep_runs,
                             args.sweep_seeds)
        print("sweeping disturbance bound ...")
        # The grid deliberately straddles a_max = 0.10. Lemma 1 says the filter is
        # non-vacuous on the whole kernel iff a_max > eps_max, so the sweep is only a
        # test of the lemma if it crosses that value: below it the prediction is zero
        # breaches, above it the kernel's outer shell of width (eps_max - a_max) is
        # uninhabitable and the filter has no admissible action there at all.
        eps_rows = run_sweep(cfg, args.outdir, "eps_max",
                             (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30),
                             args.sweep_runs, args.sweep_seeds)
    make_figure1(cfg, pen_t, ab_t, rs_rows, lam_rows, args.outdir)
    make_figure2(summary["margin_profile"], eps_rows, cfg, args.outdir)
    print("done")


if __name__ == "__main__":
    main()
