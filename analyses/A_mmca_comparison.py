"""
A_mmca_comparison.py
====================
Addresses Reviewer #3, comment 1 (novelty vs. a single-channel nonlinear rate).

The reviewer notes that a single nonlinear transition rate "could in principle
reproduce similar AGGREGATE behavior." This script tests that claim directly
and quantifies the ARCHITECTURAL difference that survives it:

  1. A single-channel baseline is calibrated so that its population-mean
     behavioral reduction factor Phi_single(theta, l) matches the multi-channel
     Phi(theta, l) as closely as possible (least squares over a (theta, l)
     grid). This gives the single-channel model "comparable degrees of freedom."

  2. Both models are simulated under S5 on all four topologies. We report:
       - aggregate total infection (expected to be SIMILAR -> we concede this),
       - behavioral-combination entropy at t=50 (multi-channel > 0; single
         channel is structurally capped at 1 bit),
       - the number of distinct per-agent protection levels (multi-channel
         produces a non-degenerate distribution; single channel is two-point).

Conclusion supported: the multi-channel architecture's contribution is the
heterogeneous distribution of individual protection states, NOT the magnitude
of aggregate suppression.

Reproduces Supplementary Figure S1 and the MMCA-comparison numbers.

Usage
-----
    python analyses/A_mmca_comparison.py
    python analyses/A_mmca_comparison.py --n_runs 20
"""
# === PAPER ROLE ===
# MAIN TEXT: Fig 7 (multi- vs single-channel, 2 panels). Reviewer #3.1.
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================


import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (VectorizedEpidemicModel, sigmoid, phi, mw_test,
                 combination_entropy, protection_level_distribution)
from src.model import MASK_COEF, DIST_COEF, VACC_COEF
from src.model import VectorizedEpidemicModel as Base
from src.networks import generate_network  # noqa: F401
from src.utils import make_seed
from scipy.stats import beta as beta_dist

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

TOPOS = ["BA", "WS", "ER", "MOD"]
COLORS = {"BA": "#E24B4A", "WS": "#1D9E75", "ER": "#378ADD", "MOD": "#7F77DD"}

ETA_M, ETA_D = 0.65, 0.50
_C = beta_dist.rvs(2, 2, size=20000, random_state=0)


# --------------------------------------------------------------------------
# 1. Calibrate the single-channel model to the multi-channel aggregate Phi
# --------------------------------------------------------------------------
def phi_single(params, theta, l_bar, c=_C):
    """Single-channel behavioral reduction factor (one decision, one efficacy)."""
    b1, b2, b3, b4, eta = params
    P_s = sigmoid(b1 * c + b2 * l_bar + b3 * theta + b4)
    return float(((1 - P_s * eta) ** 2).mean())


def calibrate_single_channel():
    """Least-squares fit of the single-channel Phi to the multi-channel Phi."""
    thetas = [0.0, 0.05, 0.10, 0.20, 0.30]
    lbars = [0.0, 0.2, 0.4, 0.6, 0.8]
    targets = {(t, l): phi(t, l) for t in thetas for l in lbars}

    def loss(params):
        b1, b2, b3, b4, eta = params
        if not (0.05 < eta < 0.99):
            return 1e3
        err = 0.0
        for (t, l), tgt in targets.items():
            err += (phi_single(params, t, l) - tgt) ** 2
        return err

    # init near the mask channel
    x0 = np.array([1.9, 1.35, 3.5, -2.25, 0.62])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10})
    params = res.x

    # fit quality
    preds = np.array([phi_single(params, t, l) for (t, l) in targets])
    tgts = np.array(list(targets.values()))
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    ss_res = float(np.sum((preds - tgts) ** 2))
    ss_tot = float(np.sum((tgts - tgts.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return params, {"rmse": rmse, "r2": r2}


# --------------------------------------------------------------------------
# 2. Single-channel model (one protective behavior, calibrated efficacy)
# --------------------------------------------------------------------------
class SingleChannelModel(Base):
    """Single-channel analog: one binary protective decision per agent."""

    def __init__(self, topology, N, scenario, T, seed, sc_params):
        super().__init__(topology, N, scenario, T, seed=seed)
        self.b1, self.b2, self.b3, self.b4, self.eta_eff = sc_params
        self.protected = np.zeros(self.N, dtype=bool)

    def _behavioral_update(self, prev):
        c, l, n = self.compliance, self.info_level, self.N
        p_s = sigmoid(self.b1 * c + self.b2 * l + self.b3 * prev + self.b4)
        self.protected = self.rng.random(n) < p_s
        # Mirror into mask for recording (distancing stays False -> 1-bit cap)
        self.wearing_mask = self.protected
        self.distancing = np.zeros(n, dtype=bool)
        # Vaccination identical to multi-channel (isolates the behavioral channel)
        frac_vacc = (self.adj.dot(self.vaccinated.astype(np.float64))
                     / np.maximum(self.degrees, 1))
        p_vacc = sigmoid(VACC_COEF[0]*c + VACC_COEF[1]*l + VACC_COEF[2]*frac_vacc + VACC_COEF[3])
        eligible = (self.health == 0) & (~self.vaccinated)
        self.vaccinated |= eligible & (self.rng.random(n) < p_vacc)

    def _transmission(self):
        p = self.p
        is_S = (self.health == 0) & (~self.vaccinated)
        is_I = (self.health == 1)
        if not is_I.any() or not is_S.any():
            return
        prot_f = np.where(self.protected, 1.0 - self.eta_eff, 1.0)
        inf_strength = is_I.astype(np.float64) * p["beta_0"] * prot_f
        force = self.W.dot(inf_strength) * prot_f
        self.health[is_S & (self.rng.random(self.N) < 1 - np.exp(-force))] = 1


# --------------------------------------------------------------------------
# 3. Snapshot helpers (capture behavioral diversity at t = t_snap)
# --------------------------------------------------------------------------
def snapshot_multi(topo, seed, t_snap=50):
    m = VectorizedEpidemicModel(topo, 500, "S5", T=t_snap, seed=seed)
    m.run()
    total = m.get_summary()["total_inf"]
    H = combination_entropy(m.wearing_mask, m.distancing)
    pl = protection_level_distribution(m.wearing_mask, m.distancing, ETA_M, ETA_D)
    return total, H, pl["n_levels"], pl["entropy_bits"]


def snapshot_single(topo, seed, sc_params, eta_eff, t_snap=50):
    m = SingleChannelModel(topo, 500, "S5", T=t_snap, seed=seed,
                           sc_params=sc_params)
    m.run()
    total = m.get_summary()["total_inf"]
    H = combination_entropy(m.protected, np.zeros(m.N, bool))
    # single-channel protection levels: (1 - prot*eta_eff), two-point
    factor = np.where(m.protected, 1.0 - eta_eff, 1.0)
    n_levels = int(np.unique(np.round(factor, 6)).size)
    vals, counts = np.unique(np.round(factor, 6), return_counts=True)
    pp = counts / counts.sum()
    ent = float(-np.sum(pp * np.log2(pp))) if pp.size > 1 else 0.0
    return total, H, n_levels, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--t_snap", type=int, default=50)
    ap.add_argument("--T", type=int, default=200)
    args = ap.parse_args()

    print("Calibrating single-channel model to multi-channel aggregate Phi...")
    sc_params, fit = calibrate_single_channel()
    b1, b2, b3, b4, eta_eff = sc_params
    print(f"  fitted single-channel params: b1={b1:.3f} b2={b2:.3f} "
          f"b3={b3:.3f} b4={b4:.3f} eta_eff={eta_eff:.3f}")
    print(f"  fit quality: R^2={fit['r2']:.4f}, RMSE={fit['rmse']:.4f}")
    print("  (high R^2 => single channel reproduces the AGGREGATE Phi well)\n")

    out = {"calibration": {"params": [float(x) for x in sc_params], **fit},
           "by_topology": {}}
    t0 = time.time()

    print(f"Simulating both architectures (S5, {args.n_runs} runs each)...")
    for topo in TOPOS:
        multi = {"total": [], "H": [], "nlev": [], "lent": []}
        single = {"total": [], "H": [], "nlev": [], "lent": []}
        for r in range(args.n_runs):
            seed = make_seed(1000, r, topo, "S5")
            tm, Hm, nm, lm = snapshot_multi(topo, seed, args.t_snap)
            ts, Hs, ns, ls = snapshot_single(topo, seed, sc_params, eta_eff,
                                             args.t_snap)
            multi["total"].append(tm); multi["H"].append(Hm)
            multi["nlev"].append(nm); multi["lent"].append(lm)
            single["total"].append(ts); single["H"].append(Hs)
            single["nlev"].append(ns); single["lent"].append(ls)

        tm_a = np.array(multi["total"]) * 100
        ts_a = np.array(single["total"]) * 100
        p_tot, sig = mw_test(tm_a, ts_a, alternative="two-sided")

        rec = {
            "total_multi_mean": float(tm_a.mean()),
            "total_single_mean": float(ts_a.mean()),
            "total_diff_pp": float(tm_a.mean() - ts_a.mean()),
            "total_p": p_tot, "total_sig": sig,
            "entropy_multi": float(np.mean(multi["H"])),
            "entropy_single": float(np.mean(single["H"])),
            "protection_levels_multi": float(np.mean(multi["nlev"])),
            "protection_levels_single": float(np.mean(single["nlev"])),
            "level_entropy_multi": float(np.mean(multi["lent"])),
            "level_entropy_single": float(np.mean(single["lent"])),
        }
        out["by_topology"][topo] = rec
        print(f"  {topo}: total multi={rec['total_multi_mean']:.1f}% vs "
              f"single={rec['total_single_mean']:.1f}% "
              f"(diff={rec['total_diff_pp']:+.1f}pp, p={p_tot:.3f} {sig}) | "
              f"combo-H multi={rec['entropy_multi']:.2f} vs "
              f"single={rec['entropy_single']:.2f} bits | "
              f"levels multi={rec['protection_levels_multi']:.1f} vs "
              f"single={rec['protection_levels_single']:.1f}  "
              f"[{time.time()-t0:.0f}s]")

    with open("results/A_mmca_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/A_mmca_comparison.json")

    # ---- Fig 7 (compact 2-panel): aggregate identical, diversity differs ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(TOPOS)); w = 0.38

    # (a) total infection multi vs single (aggregate outcome identical)
    ax = axes[0]
    tm = [out["by_topology"][t]["total_multi_mean"] for t in TOPOS]
    tsg = [out["by_topology"][t]["total_single_mean"] for t in TOPOS]
    ax.bar(x - w / 2, tm, w, label="multi-channel", color="#377eb8")
    ax.bar(x + w / 2, tsg, w, label="single-channel (calibrated)", color="#e41a1c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TOPOS)
    ax.set_ylabel("Total infection (%)")
    ax.set_title(f"(a) Aggregate outcome identical ($R^2$={fit['r2']:.4f})", loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # (b) behavioral combination entropy multi vs single (diversity differs)
    ax = axes[1]
    Hm = [out["by_topology"][t]["entropy_multi"] for t in TOPOS]
    Hs = [out["by_topology"][t]["entropy_single"] for t in TOPOS]
    ax.bar(x - w / 2, Hm, w, label="multi-channel", color="#377eb8")
    ax.bar(x + w / 2, Hs, w, label="single-channel", color="#e41a1c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TOPOS)
    ax.set_ylabel("Behavioral combination entropy (bits)")
    ax.set_title("(b) Behavioral diversity differs", loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(pad=1.2)
    plt.savefig("figures/fig7_mmca_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/fig7_mmca_comparison.png")


if __name__ == "__main__":
    main()
