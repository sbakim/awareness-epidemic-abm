"""
F_sensitivity.py
================
Extended Latin Hypercube Sampling (LHS) sensitivity analysis.

Addresses Reviewer #2/#3, comment 6 (extend the sensitivity analysis beyond a
narrow +-30% range). Varies five parameters simultaneously across +-30%, +-50%,
and +-70% perturbation ranges and computes Spearman correlations with total
infection under Scenario S6.

Reproduces Table 8 and Supplementary Figure S4.

Usage
-----
    python analyses/F_sensitivity.py
    python analyses/F_sensitivity.py --n_samples 50 --n_runs 5
"""
# === PAPER ROLE ===
# TABLE/VALIDATION: produces Table 8 (Spearman rho); figure kept FOR VALIDATION ONLY. Reviewer #4.6.
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================


import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.stats import spearmanr, qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import VectorizedEpidemicModel

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

TOPOS = ["BA", "WS", "ER", "MOD"]
COLORS = {"BA": "#E24B4A", "WS": "#1D9E75", "ER": "#378ADD", "MOD": "#7F77DD"}
PARAMS = ["beta_0", "alpha", "delta_local", "mu", "kappa"]
BASE = dict(beta_0=0.12, gamma=0.1, eta_m=0.65, eta_d=0.50,
            alpha=0.12, delta_local=0.35, mu=0.04,
            kappa=0.6, psi=0.35, t_start=5, n_init_inf=5, vacc_rate=0.02)


def run_lhs(pert_frac, n_samples, n_runs, N, T, seed):
    bounds = {p: (BASE[p] * (1 - pert_frac), BASE[p] * (1 + pert_frac))
              for p in PARAMS}
    sampler = qmc.LatinHypercube(d=len(PARAMS), seed=seed)
    unit = sampler.random(n=n_samples)
    scaled = qmc.scale(unit, [bounds[p][0] for p in PARAMS],
                       [bounds[p][1] for p in PARAMS])
    samples = [{PARAMS[j]: float(scaled[i, j]) for j in range(len(PARAMS))}
               for i in range(n_samples)]
    results = {topo: [] for topo in TOPOS}
    for idx, sample in enumerate(samples):
        params = {**BASE, **sample}
        for topo in TOPOS:
            totals = []
            for r in range(n_runs):
                # deterministic seed: encodes run, sample idx, topo, pert range
                seed_r = (seed * 100000 + idx * 100 + r * 7
                          + TOPOS.index(topo) + int(pert_frac * 1000))
                m = VectorizedEpidemicModel(topo, N, "S6", T, seed=seed_r,
                                            params=params)
                m.run()
                totals.append(m.get_summary()["total_inf"])
            results[topo].append({"params": sample,
                                  "mean": float(np.mean(totals))})
    return results


def compute_correlations(lhs_results):
    corr = {}
    for topo in TOPOS:
        corr[topo] = {}
        means = [r["mean"] for r in lhs_results[topo]]
        for p in PARAMS:
            vals = [r["params"][p] for r in lhs_results[topo]]
            rho, pv = spearmanr(vals, means)
            sig = ("***" if pv < 0.001 else "**" if pv < 0.01 else
                   "*" if pv < 0.05 else "ns")
            corr[topo][p] = {"rho": float(rho), "p": float(pv), "sig": sig}
    return corr


def print_table(corr, label):
    print(f"\n+-{label}%:")
    print(f"  {'Param':>12}  " + "  ".join(f"{t:>9}" for t in TOPOS))
    for p in PARAMS:
        row = f"  {p:>12}:"
        for topo in TOPOS:
            c = corr[topo][p]
            row += f"  {c['rho']:+.2f}{c['sig']:<3}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_corr = {}
    t0 = time.time()
    for pert_frac, label in [(0.30, "30"), (0.50, "50"), (0.70, "70")]:
        print(f"\nRunning +-{label}% LHS "
              f"({args.n_samples} samples x {args.n_runs} runs x 4 topo)...")
        res = run_lhs(pert_frac, args.n_samples, args.n_runs,
                      args.N, args.T, args.seed)
        corr = compute_correlations(res)
        all_corr[f"{label}pct"] = corr
        print_table(corr, label)
        print(f"  [{time.time()-t0:.0f}s]")

    with open("results/F_extended_sensitivity.json", "w") as f:
        json.dump({"correlations": all_corr}, f, indent=2)
    print("\nSaved results/F_extended_sensitivity.json")

    # ---- Figure S4 -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    labels = ["30pct", "50pct", "70pct"]
    names = ["+-30%", "+-50%", "+-70%"]
    bar_colors = ["#93C5FD", "#3B82F6", "#1D4ED8"]

    ax = axes[0]
    x = np.arange(len(PARAMS)); w = 0.25
    for pi, (lab, nm, col) in enumerate(zip(labels, names, bar_colors)):
        rhos = [all_corr[lab]["BA"][p]["rho"] for p in PARAMS]
        ax.bar(x + (pi - 1) * w, rhos, w, label=nm, color=col, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([p.replace("_", "\n") for p in PARAMS])
    ax.set_ylabel("Spearman rho (BA)")
    ax.set_title("(a) Sensitivity by perturbation range", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for topo in TOPOS:
        rho_p = [all_corr[lab][topo]["beta_0"]["rho"] for lab in labels]
        ax.plot(names, rho_p, color=COLORS[topo], marker="o", lw=2,
                markersize=8, label=topo)
    ax.set_ylabel("Spearman rho (beta_0 vs total infection)")
    ax.set_title("(b) beta_0 dominance vs perturbation range", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/_validation_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/_validation_sensitivity.png")


if __name__ == "__main__":
    main()
