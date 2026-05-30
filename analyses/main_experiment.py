"""
main_experiment.py
==================
Reproduces the main simulation results: Table 5 (scenario comparison) and
Table 6 (pairwise S4/S5/S6 comparisons), plus the infection-curve and
heatmap figures.

Runs n_runs Monte Carlo replications for each of the 24 (topology x scenario)
configurations and saves results to results/main_results.json.

Usage
-----
    python analyses/main_experiment.py
    python analyses/main_experiment.py --n_runs 30 --N 500 --T 200
"""
# === PAPER ROLE ===
# MAIN TEXT: Table 5, Table 6, Fig 1 (infection curves), Fig 2 (co-evolution). Also writes a scenario heatmap FOR VALIDATION ONLY (numbers are in Table 5).
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================


import argparse
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (VectorizedEpidemicModel, mc_run, mw_test, cohen_d,
                 bonferroni_alpha)
from src.utils import make_seed

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SC_COLORS = {"S1": "#8B0000", "S2": "#CD5C5C", "S3": "#E07B54",
             "S4": "#90EE90", "S5": "#3CB371", "S6": "#006400"}
SC_STYLE = {"S1": "--", "S2": "--", "S3": "--",
            "S4": "-", "S5": "-", "S6": "-"}


def run_experiment(topologies, scenarios, N, T, n_runs, seed_base):
    results = {topo: {sc: [] for sc in scenarios} for topo in topologies}
    t0 = time.time()
    total = len(topologies) * len(scenarios) * n_runs
    done = 0
    for topo in topologies:
        for sc in scenarios:
            results[topo][sc] = mc_run(VectorizedEpidemicModel, topo, sc,
                                       N, T, n_runs, seed_base=seed_base)
            done += n_runs
            el = time.time() - t0
            eta = (total - done) * el / done if done else 0
            mean_tot = np.mean([r["total_inf"] for r in results[topo][sc]])
            print(f"  {topo}/{sc}: total={mean_tot:.1%}  [{el:.0f}s | ETA {eta:.0f}s]")
        print(f"-- {topo} done --")
    return results


def compute_table5(results, topologies, scenarios):
    table = {}
    for topo in topologies:
        s1_tot = np.array([r["total_inf"] for r in results[topo]["S1"]])
        s1_pk = np.array([r["peak_inf"] for r in results[topo]["S1"]])
        table[topo] = {}
        for sc in scenarios:
            tot = np.array([r["total_inf"] for r in results[topo][sc]])
            pk = np.array([r["peak_inf"] for r in results[topo][sc]])
            p, sig = mw_test(s1_tot, tot, alternative="greater")
            d = cohen_d(s1_tot, tot)
            table[topo][sc] = {
                "peak_mean": float(pk.mean()), "peak_std": float(pk.std()),
                "total_mean": float(tot.mean()), "total_std": float(tot.std()),
                "peak_red": float(1 - pk.mean() / s1_pk.mean()) if sc != "S1" else None,
                "total_red": float(1 - tot.mean() / s1_tot.mean()) if sc != "S1" else None,
                "p": p, "sig": sig,
                "cohen_d": d if sc != "S1" else None,
            }
    return table


def compute_table6(results, topologies):
    """Pairwise comparisons among behavioral scenarios S4, S5, S6 (Table 6)."""
    pairs = [("S4", "S5"), ("S5", "S6"), ("S4", "S6")]
    alpha_adj = bonferroni_alpha(0.05, 12)
    table = {}
    for topo in topologies:
        table[topo] = {}
        for a, b in pairs:
            ta = np.array([r["total_inf"] for r in results[topo][a]]) * 100
            tb = np.array([r["total_inf"] for r in results[topo][b]]) * 100
            p, _ = mw_test(ta, tb, alternative="two-sided")
            d = cohen_d(tb, ta)
            sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                   "*" if p < alpha_adj else "ns")
            table[topo][f"{a}_vs_{b}"] = {
                "delta_pp": float(tb.mean() - ta.mean()),
                "p": p, "cohen_d": d, "sig": sig,
            }
    return table


def plot_infection_curves(topologies, scenarios, T, seed=42):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, topo in zip(axes.flatten(), topologies):
        for sc in scenarios:
            m = VectorizedEpidemicModel(topo, 500, sc, T,
                                        seed=make_seed(seed, 0, topo, sc))
            ts = m.run()
            ax.plot(range(1, T + 1), [v * 100 for v in ts["I"]],
                    color=SC_COLORS[sc], ls=SC_STYLE[sc], lw=1.8,
                    label=sc, alpha=0.9)
        ax.axvline(5, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.set_title(f"({'abcd'[topologies.index(topo)]}) {topo}",
                     loc="left", fontsize=12)
        ax.set_ylabel("Infected (%)"); ax.set_xlabel("Day")
        ax.set_xlim(0, 80); ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig1_infection_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/fig1_infection_curves.png")


def plot_coevolution(topologies, T, seed=2):
    """MAIN TEXT Fig 2: co-evolution of infection, mean awareness, and mask rate
    under S5 (dual spreading) across the four topologies. Uses the canonical
    recalibrated coefficients via the model defaults."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, topo in zip(axes.flatten(), topologies):
        r = VectorizedEpidemicModel(topo, 500, "S5", T,
                                    seed=make_seed(seed, 0, topo, "S5")).run()
        I = np.array(r["I"]) * 100
        aw = np.array(r["mean_awareness"]); mk = np.array(r["mask_rate"])
        days = np.arange(1, len(I) + 1)
        ax.plot(days, I, color="#d62728", lw=2, label="Infected %")
        ax.set_xlabel("Day"); ax.set_ylabel("Infected (%)", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#d62728"); ax.set_xlim(0, 80)
        ax2 = ax.twinx()
        ax2.plot(days, aw, color="#1f77b4", lw=2, label="Mean awareness")
        ax2.plot(days, mk, color="#2ca02c", lw=1.6, ls="--", label="Mask rate")
        ax2.set_ylabel("Awareness / Mask rate", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4"); ax2.set_ylim(0, 1.0)
        ax.axvline(5, color="gray", ls=":", lw=1)
        ax.set_title(f"({'abcd'[topologies.index(topo)]}) {topo}", loc="left", fontsize=11)
        if topo == topologies[0]:
            l1, lab1 = ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")
    plt.tight_layout(pad=1.3)
    plt.savefig("figures/fig2_awareness_coevolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/fig2_awareness_coevolution.png")


def plot_heatmap(table, topologies, scenarios):
    # FOR VALIDATION ONLY -- NOT a manuscript figure.
    # The manuscript reports these numbers in Table 5 (with peak, SD, p, d),
    # so the heatmap is redundant. Kept here so the result stays reproducible
    # and inspectable; do not cite it as a figure in the paper.
    data = np.array([[table[t][sc]["total_mean"] * 100 for sc in scenarios]
                     for t in topologies])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(scenarios)
    ax.set_yticks(range(len(topologies))); ax.set_yticklabels(topologies)
    ax.set_xlabel("Scenario"); ax.set_ylabel("Topology")
    for i in range(len(topologies)):
        for j in range(len(scenarios)):
            ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center",
                    fontweight="bold",
                    color="white" if data[i, j] > 50 else "black")
    plt.colorbar(im, ax=ax, label="Total infection (%)")
    plt.tight_layout()
    plt.savefig("figures/_validation_scenario_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/_validation_scenario_heatmap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="+", default=["BA", "WS", "ER", "MOD"])
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4", "S5", "S6"])
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Main experiment: {len(args.topologies)} topo x "
          f"{len(args.scenarios)} sc x {args.n_runs} runs (N={args.N}, T={args.T})")
    results = run_experiment(args.topologies, args.scenarios,
                             args.N, args.T, args.n_runs, args.seed)
    table5 = compute_table5(results, args.topologies, args.scenarios)
    table6 = compute_table6(results, args.topologies)

    with open("results/main_results.json", "w") as f:
        json.dump({"params": vars(args), "table5": table5, "table6": table6},
                  f, indent=2)
    print("Saved results/main_results.json")

    print("\n" + "=" * 78)
    print("TABLE 5 - Monte Carlo results (mean +/- SD)")
    print("=" * 78)
    print(f"{'Topo':>5} {'SC':>3} {'Peak%':>11} {'Total%':>11} "
          f"{'PeakRed':>8} {'TotRed':>8} {'p':>9} {'d':>6}")
    for topo in args.topologies:
        for sc in args.scenarios:
            r = table5[topo][sc]
            pr = f"{r['peak_red']*100:.1f}" if r["peak_red"] is not None else "-"
            tr = f"{r['total_red']*100:.1f}" if r["total_red"] is not None else "-"
            d = f"{r['cohen_d']:.2f}" if r["cohen_d"] is not None else "-"
            print(f"{topo:>5} {sc:>3} "
                  f"{r['peak_mean']*100:>5.1f}+/-{r['peak_std']*100:<4.1f} "
                  f"{r['total_mean']*100:>5.1f}+/-{r['total_std']*100:<4.1f} "
                  f"{pr:>8} {tr:>8} {r['p']:>9.1e} {d:>6}")

    print("\n" + "=" * 60)
    print("TABLE 6 - Pairwise S4/S5/S6 (Delta pp, p, Cohen d, sig)")
    print("=" * 60)
    for topo in args.topologies:
        for key, r in table6[topo].items():
            print(f"  {topo:>4} {key:>9}: d_pp={r['delta_pp']:+.1f} "
                  f"p={r['p']:.3f} d={r['cohen_d']:.2f} {r['sig']}")

    plot_infection_curves(args.topologies, args.scenarios, args.T)   # MAIN TEXT Fig 1
    plot_coevolution(args.topologies, args.T)                        # MAIN TEXT Fig 2
    plot_heatmap(table5, args.topologies, args.scenarios)            # FOR VALIDATION (not shown)


if __name__ == "__main__":
    main()
