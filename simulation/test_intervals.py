"""Checks on the exact Clopper-Pearson implementation used in ab_mdp_simulation.py.

The manuscript reports every proportion with an exact 95% interval. Those intervals are
computed in this repository without scipy, so they are themselves a claim that needs an
external check. Two independent checks are applied:

  1. Published Clopper-Pearson table values (textbook / standard software output).
  2. Closed-form identities, which hold exactly and are derived independently of the
     incomplete beta function:
         k = 0 : upper = 1 - (alpha/2)^(1/n)
         k = n : lower = (alpha/2)^(1/n)
         k = 1 : lower = 1 - (1 - alpha/2)^(1/n)
     and the defining property, verified numerically for every case:
         P(X >= k | p = lower) = alpha/2,  P(X <= k | p = upper) = alpha/2.

Run:  python test_intervals.py
Exit code is 0 only if every check passes.
"""

import math
import sys

from ab_mdp_simulation import betainc, clopper_pearson

TOL = 5e-4

# Published Clopper-Pearson 95% intervals (k successes out of n trials).
PUBLISHED = [
    (0, 5, 0.00000, 0.52182),
    (4, 5, 0.28360, 0.99492),
    (0, 10, 0.00000, 0.30850),
    (1, 10, 0.00253, 0.44500),
    (5, 10, 0.18709, 0.81291),
    (10, 10, 0.69150, 1.00000),
    (0, 20, 0.00000, 0.16843),
    (1, 20, 0.00126, 0.24874),
    (2, 20, 0.01235, 0.31698),
    (3, 20, 0.03207, 0.37893),
]

fails = []


def check(label, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got:.6f}  want {want:.6f}")
    if not ok:
        fails.append(label)


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), summed in log space for stability."""
    if k <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        lp = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
              + i * math.log(p) + (n - i) * math.log1p(-p)) if 0.0 < p < 1.0 else None
        if lp is None:
            continue
        total += math.exp(lp)
    return total


print("1. Published table values")
for k, n, lo, hi in PUBLISHED:
    got_lo, got_hi = clopper_pearson(k, n)
    check(f"CP({k}/{n}) lower", got_lo, lo)
    check(f"CP({k}/{n}) upper", got_hi, hi)

print("2. Closed-form identities at the boundaries")
alpha = 0.05
for n in (5, 10, 15, 20, 2000):
    lo, hi = clopper_pearson(0, n)
    check(f"k=0, n={n} upper = 1-(a/2)^(1/n)", hi, 1.0 - (alpha / 2) ** (1.0 / n))
    check(f"k=0, n={n} lower = 0", lo, 0.0)
    lo, hi = clopper_pearson(n, n)
    check(f"k=n, n={n} lower = (a/2)^(1/n)", lo, (alpha / 2) ** (1.0 / n))
    check(f"k=n, n={n} upper = 1", hi, 1.0)
    lo, hi = clopper_pearson(1, n)
    check(f"k=1, n={n} lower = 1-(1-a/2)^(1/n)", lo, 1.0 - (1.0 - alpha / 2) ** (1.0 / n))

print("3. Defining property: tail probabilities at the reported limits")
for k, n in ((0, 15), (3, 15), (7, 15), (11, 15), (15, 15), (2, 8), (59, 2000)):
    lo, hi = clopper_pearson(k, n)
    if k > 0:
        check(f"P(X>={k} | p=lo) = a/2  (n={n})", binom_tail(k, n, lo), alpha / 2, tol=2e-3)
    if k < n:
        check(f"P(X<={k} | p=hi) = a/2  (n={n})", 1.0 - binom_tail(k + 1, n, hi),
              alpha / 2, tol=2e-3)

print("4. betainc normalisation: I_x(a,b) + I_{1-x}(b,a) = 1")
for a, b, x in ((0.5, 3.0, 0.2), (2.0, 2.0, 0.37), (7.0, 12.0, 0.61), (1.0, 1.0, 0.9)):
    check(f"I({a},{b},{x}) symmetry", betainc(a, b, x) + betainc(b, a, 1.0 - x), 1.0, tol=1e-9)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("All interval checks passed.")
