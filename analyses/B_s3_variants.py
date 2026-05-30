"""
B_s3_variants.py
================
Addresses Reviewer #3, comment 2 (the S3-S4 gap may depend on the
aggressiveness of the exogenous baseline).

We sweep the exogenous (centrality-targeted) intervention from the paper's
baseline up to a near-maximal intervention and measure, on every topology:
  - total infection of each S3 variant,
  - the S3*-S4 gap (difference vs the autonomous-agent scenario S4),
  - whether the ordering S3* > S4 is preserved.

This shows whether the architectural advantage (S4 over S3) is an artifact of
a weak baseline or persists against strong exogenous interventions, and how
aggressive the exogenous intervention must become before the gap closes.

Variants (s3_hub_reduction, s3_hub_frac, s3_vacc_mult):
    S3_baseline : 0.80, 0.15, 2x   (the paper's S3)
    S3_strong   : 0.90, 0.20, 3x
    S3_aggr     : 0.95, 0.30, 4x
    S3_extreme  : 0.99, 0.50, 5x

Reproduces Supplementary Figure S2 and the S3-variant numbers.

Usage
-----
    python analyses/B_s3_variants.py
    python analyses/B_s3_variants.py --n_runs 30
"""
# === PAPER ROLE ===
# MAIN TEXT: Fig 6 (conditional S3 vs S4 advantage). Reviewer #3.2.
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
from src import VectorizedEpidemicModel, mw_test, cohen_d
from src.utils import make_seed

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

TOPOS = ["BA", "WS", "ER", "MOD"]
COLORS = {"BA": "#E24B4A", "WS": "#1D9E75", "ER": "#378ADD", "MOD": "#7F77DD"}

VARIANTS = {
    "S3_baseline": dict(s3_hub_reduction=0.80, s3_hub_frac=0.15, s3_vacc_mult=2.0),
    "S3_strong":   dict(s3_hub_reduction=0.90, s3_hub_frac=0.20, s3_vacc_mult=3.0),
    "S3_aggr":     dict(s3_hub_reduction=0.95, s3_hub_frac=0.30, s3_vacc_mult=4.0),
    "S3_extreme":  dict(s3_hub_reduction=0.99, s3_hub_frac=0.50, s3_vacc_mult=5.0),
}


def run_cfg(topo, scenario, N, T, n_runs, extra_params=None):
    totals = []
    for r in range(n_runs):
        seed = make_seed(2000, r, topo, scenario)
        m = VectorizedEpidemicModel(topo, N, scenario, T, seed=seed,
                                    params=extra_params)
        m.run()
        totals.append(m.get_summary()["total_inf"])
    return np.array(totals) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=200)
    args = ap.parse_args()

    out = {"variants": {k: v for k, v in VARIANTS.items()}, "by_topology": {}}
    t0 = time.time()

    print(f"S3-variant sweep ({args.n_runs} runs each)...")
    for topo in TOPOS:
        s4 = run_cfg(topo, "S4", args.N, args.T, args.n_runs)
        rec = {"S4_total": float(s4.mean()), "S4_std": float(s4.std()),
               "variants": {}}
        print(f"  {topo}: S4 (autonomous) = {s4.mean():.1f}%")
        for name, params in VARIANTS.items():
            sv = run_cfg(topo, "S3", args.N, args.T, args.n_runs,
                         extra_params=params)
            gap = sv.mean() - s4.mean()
            p, sig = mw_test(sv, s4, alternative="greater")
            rec["variants"][name] = {
                "total": float(sv.mean()), "std": float(sv.std()),
                "gap_vs_S4_pp": float(gap),
                "p_greater_than_S4": p, "sig": sig,
                "ordering_preserved": bool(sv.mean() > s4.mean()),
                "cohen_d": cohen_d(sv, s4),
            }
            print(f"    {name:>12}: {sv.mean():5.1f}%  gap_vs_S4={gap:+5.1f}pp  "
                  f"(S3>S4 p={p:.3f} {sig})  [{time.time()-t0:.0f}s]")
        out["by_topology"][topo] = rec

    with open("results/B_s3_variants.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/B_s3_variants.json")

    # ---- Figure S2 -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    vnames = list(VARIANTS.keys())

    ax = axes[0]
    x = np.arange(len(vnames)); w = 0.2
    for di, topo in enumerate(TOPOS):
        ys = [out["by_topology"][topo]["variants"][v]["total"] for v in vnames]
        ax.bar(x + (di - 1.5) * w, ys, w, label=topo, color=COLORS[topo])
    # S4 reference lines
    for topo in TOPOS:
        ax.axhline(out["by_topology"][topo]["S4_total"], color=COLORS[topo],
                   ls=":", lw=1.2, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([v.replace("S3_", "") for v in vnames])
    ax.set_xlabel("Exogenous intervention aggressiveness")
    ax.set_ylabel("Total infection (%)")
    ax.set_title("(a) S3 variants vs S4 (dotted = S4 level)", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for topo in TOPOS:
        gaps = [out["by_topology"][topo]["variants"][v]["gap_vs_S4_pp"]
                for v in vnames]
        ax.plot([v.replace("S3_", "") for v in vnames], gaps, marker="o",
                lw=2, color=COLORS[topo], label=topo)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Exogenous intervention aggressiveness")
    ax.set_ylabel("S3*-S4 gap (pp; >0 means S4 still better)")
    ax.set_title("(b) Architectural gap vs baseline strength", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/fig6_s3_variants.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/fig6_s3_variants.png")


if __name__ == "__main__":
    main()
