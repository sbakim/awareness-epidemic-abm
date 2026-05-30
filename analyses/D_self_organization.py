"""
D_self_organization.py
======================
Addresses Reviewer #3, comment 3 (self-organization claim).

Computes genuine spatial-coordination diagnostics for the behavioral scenarios
S4, S5, S6 and tests whether awareness diffusion (S5/S6) produces spatial
behavioral clustering BEYOND the stimulus-driven convergence already present in
S4. Metrics at t = t_snap (default 50), averaged over runs:

  - order parameter Phi_ord = |2<m> - 1|
  - behavioral entropy H (bits)  [over (mask, distancing)]
  - spatial synchronization S = Moran's I of the mask indicator
  - neighbor mutual information (bits) between an agent's mask and its
    neighbors' mean mask

Significance: S5-vs-S4 (and S6-vs-S4) Mann-Whitney tests on Moran's I and on
neighbor MI across runs. These produce the p-values reported in Section 4.3
(supporting the statement that S5 does NOT produce significant spatial
clustering beyond stimulus-driven convergence).

Reproduces Table 7 and the self-organization p-values.

Usage
-----
    python analyses/D_self_organization.py
    python analyses/D_self_organization.py --topology BA --n_runs 30
"""
# === PAPER ROLE ===
# TABLE/VALIDATION: produces Table 7 numbers; figure kept FOR VALIDATION ONLY (no spatial self-organization). Reviewer #3.3.
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
from src import (VectorizedEpidemicModel, morans_i,
                 neighbor_mutual_information, combination_entropy, mw_test)
from src.utils import make_seed

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SCS = ["S4", "S5", "S6"]


def snapshot_metrics(topo, scenario, seed, t_snap):
    """Run to t_snap and compute coordination metrics on the mask field."""
    m = VectorizedEpidemicModel(topo, 500, scenario, T=t_snap, seed=seed)
    m.run()
    mask = m.wearing_mask.astype(float)
    order = float(abs(2 * mask.mean() - 1))
    H = combination_entropy(m.wearing_mask, m.distancing)
    I = morans_i(m.adj_bin, mask)
    MI = neighbor_mutual_information(m.adj_bin, mask, m.degrees)
    return {"order": order, "H": H, "moransI": I, "MI": MI}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="BA")
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--t_snap", type=int, default=50)
    args = ap.parse_args()

    print(f"Self-organization diagnostics on {args.topology} "
          f"({args.n_runs} runs, t={args.t_snap})")
    data = {sc: {"order": [], "H": [], "moransI": [], "MI": []} for sc in SCS}
    t0 = time.time()

    for sc in SCS:
        for r in range(args.n_runs):
            seed = make_seed(4000, r, args.topology, sc)
            mt = snapshot_metrics(args.topology, sc, seed, args.t_snap)
            for k in data[sc]:
                data[sc][k].append(mt[k])
        print(f"  {sc}: Phi_ord={np.mean(data[sc]['order']):.2f}, "
              f"H={np.mean(data[sc]['H']):.2f}, "
              f"MoranI={np.mean(data[sc]['moransI']):.3f}, "
              f"MI={np.mean(data[sc]['MI']):.3f}  [{time.time()-t0:.0f}s]")

    # Table 7
    table7 = {sc: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                   for k, v in data[sc].items()} for sc in SCS}

    # Significance: does S5 / S6 exceed S4 in spatial clustering?
    pI_54, _ = mw_test(data["S5"]["moransI"], data["S4"]["moransI"], "greater")
    pMI_54, _ = mw_test(data["S5"]["MI"], data["S4"]["MI"], "greater")
    pI_64, _ = mw_test(data["S6"]["moransI"], data["S4"]["moransI"], "greater")
    pMI_64, _ = mw_test(data["S6"]["MI"], data["S4"]["MI"], "greater")

    tests = {
        "S5_vs_S4_moransI_p": pI_54,
        "S5_vs_S4_neighborMI_p": pMI_54,
        "S6_vs_S4_moransI_p": pI_64,
        "S6_vs_S4_neighborMI_p": pMI_64,
    }
    print("\nSignificance (one-sided, S>S4):")
    for k, v in tests.items():
        print(f"  {k} = {v:.3f}  ({'sig' if v < 0.05 else 'ns'})")

    with open("results/D_self_organization.json", "w") as f:
        json.dump({"topology": args.topology, "table7": table7, "tests": tests},
                  f, indent=2)
    print("Saved results/D_self_organization.json")

    # ---- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(SCS))

    ax = axes[0]
    w = 0.27
    for i, (key, lab) in enumerate([("order", "Order param"),
                                    ("moransI", "Moran's I"),
                                    ("MI", "Neighbor MI")]):
        ys = [np.mean(data[sc][key]) for sc in SCS]
        es = [np.std(data[sc][key]) for sc in SCS]
        ax.bar(x + (i - 1) * w, ys, w, yerr=es, capsize=3, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(SCS)
    ax.set_ylabel("Coordination metric")
    ax.set_title(f"(a) Coordination metrics ({args.topology})", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, [np.mean(data[sc]["H"]) for sc in SCS],
           yerr=[np.std(data[sc]["H"]) for sc in SCS], capsize=4,
           color="#7F77DD", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SCS)
    ax.set_ylabel("Behavioral entropy H (bits)")
    ax.set_title("(b) Behavioral entropy by scenario", loc="left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/_validation_self_organization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/_validation_self_organization.png")


if __name__ == "__main__":
    main()
