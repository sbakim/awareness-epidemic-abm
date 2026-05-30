"""
utils.py
========
Statistical helpers and reproducible Monte Carlo experiment utilities.

IMPORTANT (reproducibility): seeds are derived deterministically from
(seed_base, run index, topology, scenario) WITHOUT Python's hash(), whose
string hashing is randomized per process. This guarantees that re-running the
scripts reproduces identical numbers.
"""

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


# Deterministic integer codes for seeding (no hash()).
TOPO_IDX = {"BA": 0, "WS": 1, "ER": 2, "MOD": 3}
SC_IDX = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6}


def make_seed(seed_base, run, topology="BA", scenario="S1", extra=0):
    """
    Deterministic, collision-free seed.

        seed = seed_base + run*1000 + topo*100 + sc*10 + extra

    With run < 1000 and the small topo/sc codes this is unique across the
    experiment grid and fully reproducible.
    """
    t = TOPO_IDX.get(topology, 0)
    s = SC_IDX.get(scenario, 0)
    return int(seed_base + run * 1000 + t * 100 + s * 10 + extra)


def mc_run(model_cls, topology, scenario, N, T, n_runs,
           seed_base=0, params=None):
    """
    Run n_runs independent, reproducible Monte Carlo replications.

    Returns a list of summary dicts (one per run).
    """
    results = []
    for r in range(n_runs):
        seed = make_seed(seed_base, r, topology, scenario)
        m = model_cls(topology, N, scenario, T, seed=seed, params=params)
        m.run()
        results.append(m.get_summary())
    return results


def total_infections(mc_results):
    return np.array([r["total_inf"] for r in mc_results])


def peak_infections(mc_results):
    return np.array([r["peak_inf"] for r in mc_results])


def mw_test(a, b, alternative="two-sided"):
    """Mann-Whitney U test. Returns (p_value, label)."""
    _, p = mannwhitneyu(np.asarray(a), np.asarray(b), alternative=alternative)
    label = ("***" if p < 0.001 else "**" if p < 0.01 else
             "*" if p < 0.05 else "ns")
    return float(p), label


def cohen_d(a, b):
    """Cohen's d effect size for two independent samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1))
                     / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def spearman_corr(x, y):
    """Spearman rank correlation. Returns (rho, p, label)."""
    rho, p = spearmanr(np.asarray(x), np.asarray(y))
    label = ("***" if p < 0.001 else "**" if p < 0.01 else
             "*" if p < 0.05 else "ns")
    return float(rho), float(p), label


def bonferroni_alpha(alpha=0.05, n_comparisons=1):
    return alpha / n_comparisons


def summary_stats(arr):
    arr = np.asarray(arr, float)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "median": float(np.median(arr)),
        "q25":    float(np.percentile(arr, 25)),
        "q75":    float(np.percentile(arr, 75)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }
