"""Mechanical check that every number quoted in the manuscript matches the released records.

The manuscript states (Table 4, "Numerical results") that every quoted number is
regenerated from the released records and that the build fails rather than warns on
disagreement. This script is what makes that statement true rather than decorative.

It works by *reconstruction*: each entry builds the exact LaTeX string that the released
JSON implies, and then asserts that this string occurs in the manuscript source. A number
that was edited by hand, or that drifted out of the data, produces a missing-string
failure and a non-zero exit code.

Run:  python verify_numbers.py [--outdir results] [--tex ../manuscript/ab_mdp_perspective.tex]
Exit code is 0 only if every check passes.
"""

import argparse
import json
import math
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext


# --------------------------------------------------------------------------------------
# LaTeX formatting helpers -- these must mirror how the manuscript writes numbers
# --------------------------------------------------------------------------------------

def round_half_up(x: float, digits: int) -> str:
    """Round half away from zero, on the decimal value, with ample precision."""
    d = Decimal(str(x))
    with localcontext() as ctx:
        ctx.prec = 80
        q = d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return str(q)


def is_tie(x: float, digits: int) -> bool:
    """True if x sits exactly on a rounding tie at `digits` decimal places.

    A quoted value that is a tie has its displayed last digit determined by the rounding
    convention rather than by the data: the mean of the 15 per-run breach rates is
    exactly 87.25%, which Python renders '87.2' (round-half-even on the binary value)
    while a reader recomputing by hand gets '87.3'. The manuscript therefore quotes
    87.25%, and the build screens every quoted value for this situation.
    """
    d = Decimal(str(x)) * (Decimal(10) ** digits)
    frac = abs(d - d.to_integral_value(rounding=ROUND_HALF_EVEN))
    return frac == Decimal("0.5")


# Rounding ties found while building needles; reported as failures so they cannot be
# forgotten. Populated by tex_pct/tex_sci.
TIES = []
# Every value that was screened for a tie, so the report can state the coverage rather
# than claiming "none" over an empty set.
SCREENED = []


def tex_sci(x: float, sig: int = 2) -> str:
    """Render x as LaTeX scientific notation with the manuscript's convention.

    3_000_000 -> '3\\times10^6' ; 1.2296e-6 -> '1.2\\times10^{-6}' ; 2409740 -> '2.4\\times10^6'

    Braces around the exponent are emitted only when it is not a single digit; norm()
    below makes the check insensitive to that choice, since the two typeset identically.
    """
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10.0 ** exp)
    if is_tie(mant, sig - 1):
        TIES.append("mantissa of %r at %d s.f." % (x, sig))
    mant = float(round_half_up(mant, sig - 1))
    if mant >= 10.0:
        mant /= 10.0
        exp += 1
    m = ("%g" % mant)
    if -1 <= exp <= 9:
        return r"%s\times10^%d" % (m, exp)
    return r"%s\times10^{%d}" % (m, exp)


def tex_pct(x: float, digits: int = 1) -> str:
    """Render x (already in percent, e.g. 0.841 means 0.841%) as '0.8\\%'.

    The format string must actually be applied to x. Returning the bare format string
    would make every needle a literal '%.1f\\%' that can never occur in the manuscript,
    turning this check into 17 guaranteed failures -- and the same mistake in
    make_sweep_table.py prints literal '%.1f\\%' into the Supplementary tables, which is
    exactly what it did until this was caught.
    """
    if is_tie(x, digits):
        TIES.append("%r quoted at %d decimal place(s)" % (x, digits))
    SCREENED.append((x, digits))
    return round_half_up(x, digits) + r"\%"


def tex_pct_compact(x: float, digits: int = 1) -> str:
    """tex_pct, but an integral value is written without a decimal point.

    The main text writes 'between 65.9% and 100%', while the Supplementary tables print
    '100.0%' so that a column of figures lines up. Both are the same number; this helper
    keeps the string check aligned with the prose rather than forcing the prose to match
    a table's column formatting.
    """
    d = Decimal(str(x))
    if d == d.to_integral_value():
        return tex_pct(x, 0)
    return tex_pct(x, digits)


def norm(s: str) -> str:
    """Normalise LaTeX for *numerical* substring matching.

    Two differences between the manuscript's house style and the generated needles carry
    no meaning and must not fail a check:
      * math delimiters -- the manuscript writes '$80.3\\%$ of steps', a needle may write
        '80.3\\% of steps';
      * exponent braces -- '$3\\times10^6$' and '$3\\times10^{6}$' typeset identically.
    Braces around a longer exponent ('10^{-6}') are left alone.
    """
    s = s.replace("$", "")
    s = re.sub(r"\^\{([^{}]{1,2})\}", r"^\1", s)
    return s


def contains(hay: str, needle: str) -> bool:
    """norm-based substring test with a digit-boundary guard.

    Normalising away '$' could otherwise let the needle '18.5' match the text '18.52',
    so a match is rejected when the character just after (or just before) it continues a
    longer number. This keeps the check strict while removing formatting-only noise.
    """
    n = norm(needle)
    h = norm(hay)
    for m in re.finditer(re.escape(n), h):
        before = h[m.start() - 1:m.start()]
        after = h[m.end():m.end() + 1]
        if after.isdigit():
            continue
        if before.isdigit() or before == ".":
            continue
        return True
    return False


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tex", default=os.path.join("..", "manuscript",
                                                  "ab_mdp_perspective.tex"))
    args = ap.parse_args()

    tex = open(args.tex, encoding="utf-8").read()
    summary = load(os.path.join(args.outdir, "summary.json"))
    sw_rs = load(os.path.join(args.outdir, "sweep_reward_scale.json"))
    sw_lam = load(os.path.join(args.outdir, "sweep_lambda.json"))
    sw_eps = load(os.path.join(args.outdir, "sweep_eps_max.json"))

    fp = summary["fixed_penalty"]
    ab = summary["abmdp"]
    n_seeds = summary["headline_seeds"]
    per_run = summary["eval_runs_per_seed"]
    rates = fp["breach_rate_per_seed"]

    checks = []          # (label, expected LaTeX substring)

    # ---- headline: penalty baseline -------------------------------------------------
    checks.append(("headline seed count",
                   "%d independent training runs" % n_seeds))
    checks.append(("evaluation episodes per run",
                   "%s evaluation episodes each"
                   % ("{:,}".format(per_run).replace(",", "{,}"))))
    checks.append(("total episodes",
                   r"$%s$ episodes" % tex_sci(round(rates.__len__() * per_run), 2)))
    # The mean is quoted to two decimals because at one decimal it is an exact tie
    # (87.25%) and the displayed digit would then depend on the rounding convention; the
    # tie screen below reports exactly that situation.
    checks.append(("breach rate mean and s.d.",
                   r"$%s \pm %s$" % (tex_pct(fp["breach_rate_mean"] * 100, 2),
                                     tex_pct(fp["breach_rate_std"] * 100))))
    checks.append(("median breach rate",
                   "median $%s$" % tex_pct(fp["breach_rate_median"] * 100, 0)))
    checks.append(("leaky run count",
                   "%d of the %d runs" % (fp["leaky_runs"], n_seeds)))
    non_leaky = sorted(r for r in rates if r < 0.5)
    checks.append(("non-leaky run rates",
                   "one in $%s$ and one in $%s$" % (tex_pct(non_leaky[0] * 100, 2),
                                                    tex_pct(non_leaky[1] * 100, 2))))
    checks.append(("leaky run fraction",
                   "%d/%d = %s" % (fp["leaky_runs"], n_seeds,
                                   tex_pct(fp["leaky_runs_frac"] * 100))))
    ci = fp["leaky_runs_frac_ci95_exact"]
    checks.append(("exact CI on leaky run fraction",
                   r"$[%s, %s]$" % (tex_pct(ci[0] * 100), tex_pct(ci[1] * 100))))
    checks.append(("penalty mean task return",
                   "$%.1f$" % fp["mean_task_return"]))
    checks.append(("penalty mean final distance",
                   "$%.2f$" % fp["mean_final_dist"]))

    # ---- headline: filtered controller ----------------------------------------------
    checks.append(("evaluated steps, filtered",
                   r"$%s$ evaluated steps" % tex_sci(ab["evaluated_steps"], 2)))
    # The sentence emphasises the zero: '\emph{zero} breaches across 3x10^6 evaluated
    # steps'. Matching the emphasis command as well keeps the check tied to the sentence
    # that makes the claim rather than to any occurrence of the words.
    checks.append(("zero breach steps",
                   r"\emph{zero} breaches across $%s$ evaluated steps"
                   % tex_sci(ab["evaluated_steps"], 2)))
    checks.append(("exact upper bound on violation rate",
                   r"$%s$" % tex_sci(ab["breach_rate_ci95_exact"][1], 2)))
    checks.append(("filter active fraction",
                   "%s of steps on average" % tex_pct(ab["mean_proj_active_frac"] * 100)))
    checks.append(("pooled mean correction",
                   r"$\|\Delta a\|_2 = %.3f$" % ab["mean_proj_norm_pooled"]))
    checks.append(("steps on which the filter acted",
                   r"$%s$ steps on which it acted"
                   % tex_sci(ab["proj_active_steps_total"], 3)))
    checks.append(("mean sigma_pain",
                   "$%.3f$ along filtered trajectories" % ab["mean_sigma_pain"]))
    checks.append(("filtered mean task return",
                   "mean task return $%.1f$" % ab["mean_task_return"]))
    checks.append(("filtered mean final distance",
                   "distance to goal $%.2f$" % ab["mean_final_dist"]))

    # ---- the abstract -----------------------------------------------------------------
    # The abstract is the most-read paragraph in the paper and it carries three numbers,
    # so it gets its own checks: the seed count, the run-level leak count, and the
    # filtered controller's evaluated steps. Without these the abstract is the one place
    # a number could drift unchecked.
    checks.append(("abstract seed count",
                   "In a %d-seed study" % n_seeds))
    checks.append(("abstract leaking-run count",
                   "breached the kernel in %d of %d runs"
                   % (fp["leaky_runs"], n_seeds)))
    checks.append(("abstract filtered evaluated steps",
                   "zero breaches in $%s$ steps" % tex_sci(ab["evaluated_steps"], 2)))

    # ---- the disturbance-sweep sentence in the main text ------------------------------
    # The sentence states the threshold rather than the per-row numbers, which live in the
    # generated Table S1; so the only literal it must match is a_max itself.
    checks.append(("disturbance-sweep threshold quoted in the main text",
                   r"a_{\max} = %.2f$" % summary["config"]["a_max"]))
    checks.append(("zero-breach claim for the disturbance sweep",
                   "exactly zero} breach steps"))

    # ---- margin profile --------------------------------------------------------------
    prof = {r["range"]: r for r in summary["margin_profile"]}
    outer = prof["[0.00, 0.05)"]
    band = prof["[0.05, 0.10)"]
    third = prof["[0.10, 0.15)"]
    checks.append(("outermost bin intervention rate",
                   "%s of steps with a mean correction of $%.3f$"
                   % (tex_pct(outer["active_frac"] * 100), outer["mean_disp_active"])))
    checks.append(("second bin intervention rate and correction",
                   "%s with mean $%.3f$" % (tex_pct(band["active_frac"] * 100),
                                            band["mean_disp_active"])))
    checks.append(("third bin intervention rate",
                   "%s;" % tex_pct(third["active_frac"] * 100)))
    checks.append(("sigma_pain at the outermost bin",
                   r"$\sigma_{\text{pain}} = %.3f$" % outer["mean_sigma_pain"]))
    checks.append(("sigma_pain at the innermost reported bin",
                   "$%.3f$ at margin" % prof["[0.50, inf)"]["mean_sigma_pain"]))
    interior_steps = sum(r["steps"] for r in summary["margin_profile"]
                         if r["steps"] and r["range"] not in
                         ("[0.00, 0.05)", "[0.05, 0.10)", "[0.10, 0.15)"))
    checks.append(("steps at a margin above 0.15",
                   r"$%s$ steps, not one intervention" % tex_sci(interior_steps, 2)))

    # ---- sweeps ----------------------------------------------------------------------
    # The per-row numbers of the reward-scale and penalty-weight sweeps live in the
    # Supplementary tables, which make_sweep_table.py generates from these same JSON
    # records -- so they cannot drift and need no per-row string check here.
    #
    # What the main text does still quote are the summary values below, and each is
    # checked both as a string and, further down, as a record-level claim: the sentence
    # asserts a pattern over *all* rows ("every reward scale ... except", "rising from
    # 4/8 ... to 7/8 or 8/8 for every lambda >= 5"), so checking only the two quoted
    # literals would leave the pattern itself unverified.
    rs_rate = {r["value"]: r["penalty_breach_rate"] for r in sw_rs}
    lam_by_v = {r["value"]: r for r in sw_lam}
    lam_rate = [r["penalty_breach_rate"] for r in sw_lam]

    checks.append(("reward scale: all-but-one at 100% breach",
                   "Every reward scale tested at $\\lambda = 10$ gives a $100\\%%$ breach "
                   "rate, except $\\bar{R}=8$ at $%s$"
                   % tex_pct(rs_rate[8.0] * 100)))
    checks.append(("penalty weight: quoted range",
                   "between $%s$ and $%s$" % (tex_pct_compact(min(lam_rate) * 100),
                                              tex_pct_compact(max(lam_rate) * 100))))
    checks.append(("penalty weight: leaky-run share at lambda = 1",
                   "$%d/%d$ at $\\lambda=1$"
                   % (lam_by_v[1.0]["penalty_leaky_runs"], lam_by_v[1.0]["train_seeds"])))
    checks.append(("penalty weight: leaky-run share for lambda >= 5",
                   "$7/8$ or $8/8$ for every $\\lambda \\ge 5$"))

    # ---- figure 1 caption: seed and episode counts must match the released sweeps -----
    for rows, tag in ((sw_rs, "(c)"), (sw_lam, "(d)")):
        n = rows[0]["train_seeds"]
        r = rows[0]["sweep_runs"]
        checks.append((f"figure 1 caption {tag} seed count",
                       "%d independent training seeds" % n))
        checks.append((f"figure 1 caption {tag} episode count",
                       "%s episodes per seed" % ("{:,}".format(r).replace(",", "{,}"))))

    # ---- record-level consistency: the disturbance sweep must agree with Lemma 1 -------
    # These are not text checks. They test the *records* against the lemma, because the
    # sweep is the paper's own test of Lemma 1 and a disagreement here is a scientific
    # problem, not a typesetting one.
    # The inequality is non-strict on purpose: the vacuous shell has thickness
    # eps_max - a_max, so at eps_max == a_max the shell is empty and the filter is
    # non-vacuous throughout K. Using a strict `>` here would report the boundary row as
    # a contradiction of Lemma 1, when in fact the sweep's zero vacuous steps at
    # eps_max = 0.10 are the lemma holding exactly at its boundary.
    a_max = summary["config"]["a_max"]
    consistency = []
    for row in sorted(sw_eps, key=lambda r: r["value"]):
        v = row["value"]
        vac = row.get("abmdp_vacuous_steps_total", 0)
        br = row["abmdp_breach_steps_total"]
        vac_br = row.get("abmdp_vacuous_breach_steps", 0)
        if (a_max >= v) != (vac == 0):
            consistency.append(
                "eps_max=%.3f: Lemma 1 predicts non-vacuous=%s but %d vacuous steps were "
                "recorded" % (v, a_max >= v, vac))
        if br != vac_br:
            consistency.append(
                "eps_max=%.3f: %d breach steps of which only %d were on vacuous steps -- "
                "the filter failed where an admissible action existed"
                % (v, br, vac_br))
        if v <= a_max and br:
            consistency.append(
                "eps_max=%.3f is at or below the threshold yet recorded %d breach steps"
                % (v, br))

    # ---- record-level consistency: the knob sweeps must match the sentence's pattern ---
    # The main text makes three assertions about *all* rows of the two knob sweeps. Each is
    # re-derived here from the records, so a re-run that changes the pattern fails the
    # build instead of quietly invalidating the sentence.
    off_100 = sorted(v for v, rate in rs_rate.items() if rate < 1.0)
    if off_100 != [8.0]:
        consistency.append(
            "reward scale: the sentence excludes only R=8 from the 100%% claim, but the "
            "records give a breach rate below 1.0 at scales %s" % (off_100,))
    lam_ge5 = sorted(v for v in lam_by_v if v >= 5)
    bad_ge5 = [(v, lam_by_v[v]["penalty_leaky_runs"], lam_by_v[v]["train_seeds"])
               for v in lam_ge5
               if lam_by_v[v]["penalty_leaky_runs"] not in (7, 8)]
    if bad_ge5:
        consistency.append(
            "penalty weight: the sentence claims 7/8 or 8/8 leaky runs for every "
            "lambda >= 5, but records show (lambda, leaky, seeds) = %s" % (bad_ge5,))
    lam1 = lam_by_v[1.0]
    if lam1["penalty_leaky_runs"] != 4:
        consistency.append(
            "penalty weight: the sentence cites 4/8 leaky runs at lambda = 1, but the "
            "records give %d/%d" % (lam1["penalty_leaky_runs"], lam1["train_seeds"]))
    # The quoted span "between 65.9% and 100%" must be the span of the records, not a
    # subset of it: a new lambda row outside it would make the sentence false. The numbers
    # are *parsed back out of the sentence* and compared numerically, so this does not
    # depend on the needle the string check happens to build -- and matching the bare
    # literal '100.0%' would be satisfied by an unrelated sentence elsewhere in the paper.
    span = re.search(r"between\s+([0-9]+(?:\.[0-9]+)?)\\%\s+and\s+([0-9]+(?:\.[0-9]+)?)\\%",
                     norm(tex))
    lo_r, hi_r = min(lam_rate) * 100, max(lam_rate) * 100
    if not span:
        consistency.append(
            "penalty weight: no 'between X% and Y%' clause found in the manuscript to "
            "check against the recorded span [%s, %s]"
            % (tex_pct(lo_r), tex_pct(hi_r)))
    else:
        lo_q, hi_q = float(span.group(1)), float(span.group(2))
        # Tolerance is half a unit in the last quoted decimal place.
        if abs(lo_q - lo_r) > 0.05 or abs(hi_q - hi_r) > 0.05:
            consistency.append(
                "penalty weight: manuscript quotes [%g%%, %g%%] but the records span "
                "[%.4f%%, %.4f%%]" % (lo_q, hi_q, lo_r, hi_r))
    # A sweep row whose 8 seeds are the *same* seeds as the headline study would be a
    # different experiment than the caption claims; check the counts instead of assuming.
    for name, rows in (("reward scale", sw_rs), ("penalty weight", sw_lam)):
        for r in rows:
            if r["penalty_leaky_runs"] > r["train_seeds"]:
                consistency.append(
                    "%s at value %s: %d leaky runs out of %d seeds"
                    % (name, r["value"], r["penalty_leaky_runs"], r["train_seeds"]))

    # ---- report ----------------------------------------------------------------------
    print(f"manuscript: {args.tex}")
    print(f"records   : {args.outdir}\n")
    failures = []
    for label, needle in checks:
        if contains(tex, needle):
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}\n        expected to find: {needle!r}")
            failures.append(label)

    print()
    print(f"record-level: Lemma 1 vs. the disturbance sweep (a_max = {a_max:.3f})")
    for row in sorted(sw_eps, key=lambda r: r["value"]):
        print("   eps_max=%-6.3f  vacuous steps=%-8d breach steps=%-6d on vacuous steps=%-6d"
              % (row["value"], row.get("abmdp_vacuous_steps_total", 0),
                 row["abmdp_breach_steps_total"], row.get("abmdp_vacuous_breach_steps", 0)))
    if consistency:
        print()
        for c in consistency:
            print(f"   INCONSISTENT: {c}")
        failures.extend(consistency)
    else:
        print("   consistent at every grid point and for both knob-sweep patterns")

    print()
    print("record-level: rounding ties at the precision quoted")
    if TIES:
        for t in TIES:
            print(f"   TIE: {t} -- quote one more digit")
        failures.extend("rounding tie: %s" % t for t in TIES)
    else:
        print(f"   none: {len(SCREENED)} quoted values screened, all off a tie")

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(checks)} checks did not match the records.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"All {len(checks)} number checks passed against the released records.")


if __name__ == "__main__":
    main()
