"""
E_sigmoid_calibration.py
========================
Calibration and robustness of the sigmoid decision coefficients
(Reviewer #1 c.1 / Reviewer #2 c.1).

MAIN model coefficients (src.model.MASK_COEF/DIST_COEF/VACC_COEF) are calibrated
so that simulated adoption matches behavioral-survey anchor points:

  * Baseline (no epidemic) mask/distancing adoption is low (~5-15%), consistent
    with pre-/inter-pandemic protective-behavior surveys.
  * Adoption rises to ~40-50% for a median-compliance individual once prevalence
    reaches ~25% (perceived-risk-driven uptake).
  * Vaccination uptake is governed by social conformity: ~5% with no vaccinated
    peers, rising to ~50% when ~60% of social contacts are vaccinated
    (Kabir et al. 2019; Funk et al. 2009).

This script (1) verifies the MAIN coefficients hit these anchors, and (2) shows
that all principal findings (scenario ordering, topology ordering, the positive
S3-S4 gap, and the bounds of Phi) are ROBUST when the original heuristic
coefficients are used instead. Reproduces Table (sigmoid calibration) and figS3b.
"""
# === PAPER ROLE ===
# TABLE/VALIDATION: produces calibration numbers (small table in paper) + a figure kept FOR VALIDATION ONLY. Reviewer #1.1 / #2.1.
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================

import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as Bdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (sigmoid, phi, beta_c, measure_all_lambdas, mw_test,
                 VectorizedEpidemicModel,
                 MASK_COEF, DIST_COEF, VACC_COEF,
                 MASK_COEF_HEURISTIC, DIST_COEF_HEURISTIC, VACC_COEF_HEURISTIC)

os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

_C = Bdist.rvs(2, 2, size=200000, random_state=1)
_L = Bdist.rvs(2, 3, size=200000, random_state=2)

# survey-based target ranges
TARGETS = {
    "mask_pop_theta0":   (0.05, 0.20),
    "mask_med_theta25":  (0.40, 0.55),
    "vacc_med_peer0":    (0.00, 0.10),
    "vacc_med_peer60":   (0.40, 0.60),
}

def adoption(coef, c, l, x):
    return sigmoid(coef[0]*c + coef[1]*l + coef[2]*x + coef[3])

def anchors(mask, vacc):
    return {
        "mask_pop_theta0":  float(np.mean(adoption(mask, _C, _L, 0.0))),
        "mask_med_theta25": float(adoption(mask, 0.5, 0.4, 0.25)),
        "vacc_med_peer0":   float(adoption(vacc, 0.5, 0.4, 0.0)),
        "vacc_med_peer60":  float(adoption(vacc, 0.5, 0.4, 0.60)),
    }

def in_range(v, rng): return rng[0] <= v <= rng[1]

def phi_bounds(mk, dk):
    return (phi(0, 0, mask_coef=mk, dist_coef=dk),
            phi(0, 0.4, mask_coef=mk, dist_coef=dk),
            phi(1, 1, mask_coef=mk, dist_coef=dk))

def scenario_table(coef_params, n_runs):
    SCS = ["S1", "S2", "S3", "S4", "S5", "S6"]
    out = {}
    for topo in ["BA", "WS", "ER", "MOD"]:
        row = {}
        for sc in SCS:
            v = [VectorizedEpidemicModel(topo, scenario=sc, seed=r,
                                         params=coef_params).run()["R"][-1]
                 for r in range(n_runs)]
            row[sc] = float(np.mean(v) * 100)
        out[topo] = row
    return out


def main(n_runs=8):
    MAIN = dict(mask_coef=MASK_COEF, dist_coef=DIST_COEF, vacc_coef=VACC_COEF)
    HEUR = dict(mask_coef=MASK_COEF_HEURISTIC, dist_coef=DIST_COEF_HEURISTIC,
                vacc_coef=VACC_COEF_HEURISTIC)

    print("Sigmoid coefficient calibration\n" + "-" * 60)
    a_main = anchors(MASK_COEF, VACC_COEF)
    a_heur = anchors(MASK_COEF_HEURISTIC, VACC_COEF_HEURISTIC)
    print(f"  {'anchor':18s} {'target':>12s} {'MAIN':>8s} {'heuristic':>10s}")
    checks_main = {}
    for k in TARGETS:
        ok = in_range(a_main[k], TARGETS[k]); checks_main[k] = ok
        print(f"  {k:18s} {str(TARGETS[k]):>12s} {a_main[k]:7.1%}{'*' if ok else ' '} {a_heur[k]:9.1%}")
    print(f"  MAIN coefficients meet all anchors: {all(checks_main.values())}")

    pm = phi_bounds(MASK_COEF, DIST_COEF)
    ph = phi_bounds(MASK_COEF_HEURISTIC, DIST_COEF_HEURISTIC)
    lam = measure_all_lambdas(N=500, n_seeds=20)
    bc_main = 0.1 / (lam["BA"] * pm[1]); bc_heur = 0.1 / (lam["BA"] * ph[1])
    print(f"\n  Phi (max,Phi0,min):  MAIN=({pm[0]:.3f},{pm[1]:.3f},{pm[2]:.3f})  "
          f"heuristic=({ph[0]:.3f},{ph[1]:.3f},{ph[2]:.3f})")
    print(f"  beta_c(BA):          MAIN={bc_main:.4f}   heuristic={bc_heur:.4f}")

    print(f"\nScenario robustness ({n_runs} runs/cell):")
    tbl_main = scenario_table(MAIN, n_runs)
    tbl_heur = scenario_table(HEUR, n_runs)
    SCS = ["S1", "S2", "S3", "S4", "S5", "S6"]
    for label, tbl in [("MAIN(recalibrated)", tbl_main), ("heuristic", tbl_heur)]:
        print(f"  [{label}]")
        for topo in tbl:
            row = tbl[topo]
            major = all(row[a] >= row[b] for a, b in
                        zip(["S1", "S2", "S3", "S4"], ["S2", "S3", "S4", "S6"]))
            print(f"    {topo}: " + " ".join(f"{sc}={row[sc]:4.1f}" for sc in SCS)
                  + f" | S3-S4={row['S3']-row['S4']:+.1f}pp order={'OK' if major else 'BROKEN'}")

    out = {
        "main_coef": {"mask": list(MASK_COEF), "dist": list(DIST_COEF), "vacc": list(VACC_COEF)},
        "heuristic_coef": {"mask": list(MASK_COEF_HEURISTIC), "dist": list(DIST_COEF_HEURISTIC),
                           "vacc": list(VACC_COEF_HEURISTIC)},
        "targets": TARGETS, "anchors_main": a_main, "anchors_heuristic": a_heur,
        "checks_main": checks_main,
        "phi_main": {"max": pm[0], "phi0": pm[1], "min": pm[2]},
        "phi_heuristic": {"max": ph[0], "phi0": ph[1], "min": ph[2]},
        "beta_c_BA_main": bc_main, "beta_c_BA_heuristic": bc_heur,
        "scenarios_main": tbl_main, "scenarios_heuristic": tbl_heur,
    }
    json.dump(out, open("results/E_sigmoid_calibration.json", "w"), indent=2)
    print("\nSaved results/E_sigmoid_calibration.json")

    # figure: adoption curves (main vs heuristic) + scenario comparison
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    th = np.linspace(0, 0.5, 100)
    axes[0].plot(th, [adoption(MASK_COEF, 0.5, 0.4, t) for t in th], "-", color="#E24B4A", lw=2, label="mask (main)")
    axes[0].plot(th, [adoption(MASK_COEF_HEURISTIC, 0.5, 0.4, t) for t in th], "--", color="#E24B4A", lw=1.5, label="mask (heuristic)")
    axes[0].plot(th, [adoption(DIST_COEF, 0.5, 0.4, t) for t in th], "-", color="#1D9E75", lw=2, label="dist (main)")
    axes[0].plot(th, [adoption(DIST_COEF_HEURISTIC, 0.5, 0.4, t) for t in th], "--", color="#1D9E75", lw=1.5, label="dist (heuristic)")
    axes[0].axhspan(0.05, 0.20, xmax=0.02, alpha=0); axes[0].scatter([0,0.25],[0.10,0.45], color="k", zorder=5, s=30, label="survey anchors")
    axes[0].set_xlabel(r"Prevalence $\theta$"); axes[0].set_ylabel("Adoption (median agent)")
    axes[0].set_title("(a) Calibrated adoption curves", loc="left"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    pe = np.linspace(0, 1, 100)
    axes[1].plot(pe, [adoption(VACC_COEF, 0.5, 0.4, p) for p in pe], "-", color="#378ADD", lw=2, label="vacc (main)")
    axes[1].plot(pe, [adoption(VACC_COEF_HEURISTIC, 0.5, 0.4, p) for p in pe], "--", color="#378ADD", lw=1.5, label="vacc (heuristic)")
    axes[1].scatter([0,0.6],[0.05,0.50], color="k", zorder=5, s=30, label="survey anchors")
    axes[1].set_xlabel("Vaccinated-peer fraction"); axes[1].set_ylabel("Vaccination uptake")
    axes[1].set_title("(b) Vaccination conformity", loc="left"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

    x = np.arange(6); w = 0.38
    for topo, mk in [("BA", "o")]:
        axes[2].bar(x - w/2, [tbl_main["BA"][s] for s in SCS], w, label="main", color="#E24B4A", alpha=.85)
        axes[2].bar(x + w/2, [tbl_heur["BA"][s] for s in SCS], w, label="heuristic", color="#888", alpha=.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(SCS); axes[2].set_ylabel("Total infection % (BA)")
    axes[2].set_title("(c) Scenario robustness (BA)", loc="left"); axes[2].legend(fontsize=9); axes[2].grid(alpha=.3, axis="y")
    plt.tight_layout(pad=1.5)
    plt.savefig("figures/_validation_sigmoid_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(); print("Saved figures/_validation_sigmoid_calibration.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n_runs", type=int, default=8)
    main(**vars(ap.parse_args()))
