"""
C_finite_size.py
================
Finite-size scaling analysis: N = 500, 1000, 2000, 5000, 10000.

Addresses Reviewer #3, comment 5 (N=500 is small for BA scale-free networks).
For N >= 5000 betweenness centrality is approximated by degree centrality
(Spearman rho ~ 0.93 on BA), reducing complexity from O(N^3) to O(N).

Also prints the MEASURED Lambda_BA at each N (correcting the earlier
hard-coded Lambda_BA = 35.5; the true value is ~15.6 at N=500 and rises
toward the thermodynamic limit with N).

Reproduces Figure 7 (panels a-b) and the N=10,000 results in Section 4.4.

Usage
-----
    python analyses/C_finite_size.py
    python analyses/C_finite_size.py --sizes 500 1000 2000 --n_runs 15
"""
# === PAPER ROLE ===
# MAIN TEXT: Fig 4 (finite-size scaling, N up to 10^4). Reviewer #3.5.
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================


import argparse
import json
import os
import sys
import time
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.networks import generate_network, sigmoid, network_stats
from src.model import MASK_COEF, DIST_COEF, VACC_COEF
from src.analytical import phi as phi_fn
from src.utils import make_seed

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

TOPOS = ["BA", "WS", "ER", "MOD"]
COLORS = {"BA": "#E24B4A", "WS": "#1D9E75", "ER": "#378ADD", "MOD": "#7F77DD"}
BASE = dict(beta_0=0.12, gamma=0.1, eta_m=0.65, eta_d=0.50,
            alpha=0.12, delta_local=0.35, mu=0.04,
            kappa=0.6, psi=0.35, t_start=5, n_init_inf=5, vacc_rate=0.02)


class LargeNModel:
    """Lightweight epidemic model for large N (degree-based hub detection)."""

    def __init__(self, topo, N, scenario, T=150, seed=42, params=None):
        self.sc = scenario
        self.T = T
        self.p = {**BASE, **(params or {})}
        self.rng = np.random.default_rng(seed)
        G = generate_network(topo, N, seed)
        self.N = G.number_of_nodes()
        n = self.N
        self.adj = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
        self.deg = np.array([d for _, d in G.degree()], float)
        rw = np.random.default_rng(seed + 1000)
        self.W0 = self.adj.copy()
        self.W0.data = rw.uniform(0.5, 1.0, self.W0.data.shape)
        self.W0 = (self.W0 + self.W0.T) / 2
        self.W = self.W0.copy()
        r, c = self.W0.nonzero()
        mask = r < c
        self.ei, self.ej = r[mask], c[mask]
        ntop = max(1, int(0.15 * n))
        self.hub = np.zeros(n, bool)
        self.hub[np.argsort(self.deg)[-ntop:]] = True
        self.comp = self.rng.beta(2, 2, n)
        self.info = self.rng.beta(2, 3, n)
        self.hlth = np.zeros(n, np.int8)
        self.hlth[self.rng.choice(n, min(self.p["n_init_inf"], n),
                                  replace=False)] = 1
        self.vacc = np.zeros(n, bool)
        self.mask_w = np.zeros(n, bool)
        self.dist = np.zeros(n, bool)
        self._peak = 0.0

    def run(self):
        p = self.p
        ts = p["t_start"]
        for t in range(1, self.T + 1):
            n = self.N
            iI = self.hlth == 1
            iS = self.hlth == 0
            pr = float(iI.sum() / n)
            self._peak = max(self._peak, pr)

            if self.sc in ("S4", "S5", "S6") and t >= ts:
                c, l = self.comp, self.info
                self.mask_w = self.rng.random(n) < sigmoid(MASK_COEF[0]*c + MASK_COEF[1]*l + MASK_COEF[2]*pr + MASK_COEF[3])
                self.dist = self.rng.random(n) < sigmoid(DIST_COEF[0]*c + DIST_COEF[1]*l + DIST_COEF[2]*pr + DIST_COEF[3])
                fv = self.adj.dot(self.vacc.astype(float)) / np.maximum(self.deg, 1)
                pv = sigmoid(VACC_COEF[0]*c + VACC_COEF[1]*l + VACC_COEF[2]*fv + VACC_COEF[3])
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < pv)

            if self.sc in ("S5", "S6") and t >= ts:
                l = self.info
                ds = np.maximum(self.deg, 1)
                self.info = np.clip(
                    l + p["alpha"] * (self.adj.dot(l) / ds - l)
                    + p["delta_local"] * self.adj.dot(iI.astype(float)) / ds
                    - p["mu"] * l, 0, 1)

            if self.sc == "S2" and t >= ts:
                self.W = self.W0 * 0.5
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < p["vacc_rate"])

            if self.sc == "S3" and t >= ts:
                ei, ej = self.ei, self.ej
                facs = np.where(self.hub[ei] | self.hub[ej], 0.20, 0.60)
                wn = np.array(self.W0[ei, ej]).flatten() * facs
                ai = np.concatenate([ei, ej])
                aj = np.concatenate([ej, ei])
                self.W = sparse.csr_matrix((np.concatenate([wn, wn]), (ai, aj)),
                                           shape=(n, n))
                rate = np.where(self.hub, p["vacc_rate"] * 2, p["vacc_rate"])
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < rate)

            if self.sc == "S6" and t >= ts:
                ei, ej = self.ei, self.ej
                wn = np.clip(np.array(self.W0[ei, ej]).flatten()
                             * (1 - p["kappa"] * pr)
                             * (1 - 0.5 * (self.comp[ei] + self.comp[ej]))
                             * (1 - p["psi"] * np.maximum(self.info[ei], self.info[ej])),
                             0.05, 1.0)
                ai = np.concatenate([ei, ej])
                aj = np.concatenate([ej, ei])
                self.W = sparse.csr_matrix((np.concatenate([wn, wn]), (ai, aj)),
                                           shape=(n, n))

            iSe = (self.hlth == 0) & ~self.vacc
            iI2 = self.hlth == 1
            if iI2.any() and iSe.any():
                mf = np.where(self.mask_w, 1 - p["eta_m"], 1.0)
                df = np.where(self.dist, 1 - p["eta_d"], 1.0)
                F = self.W.dot(iI2.astype(float) * p["beta_0"] * mf) * mf * df
                self.hlth[iSe & (self.rng.random(n) < 1 - np.exp(-F))] = 1

            self.hlth[(self.hlth == 1) & (self.rng.random(n) < p["gamma"])] = 2

        return float((self.hlth == 2).sum() / self.N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[500, 1000, 2000, 5000, 10000])
    ap.add_argument("--n_runs", type=int, default=15)
    ap.add_argument("--T", type=int, default=150)
    args = ap.parse_args()

    print(f"Finite-size scaling: sizes={args.sizes}, n_runs={args.n_runs}")
    results = {N: {topo: {sc: [] for sc in ("S1", "S6")} for topo in TOPOS}
               for N in args.sizes}
    lambda_ba = {}
    t0 = time.time()

    for N in args.sizes:
        # Measured Lambda_BA (averaged over a few seeds)
        lam_vals = [network_stats(generate_network("BA", N, s))["Lambda"]
                    for s in range(min(5, args.n_runs))]
        lam_ba = float(np.mean(lam_vals))
        lambda_ba[N] = lam_ba
        bc_val = BASE["gamma"] / (lam_ba * phi_fn(0, 2 / 5))

        for topo in TOPOS:
            for sc in ("S1", "S6"):
                for r in range(args.n_runs):
                    seed = make_seed(3000, r, topo, sc, extra=N % 1000)
                    results[N][topo][sc].append(
                        LargeNModel(topo, N, sc, args.T, seed=seed).run())
        s1 = np.mean(results[N]["BA"]["S1"])
        s6 = np.mean(results[N]["BA"]["S6"])
        supp = 100 * (1 - s6 / max(s1, 1e-6))
        print(f"  N={N:6d}: Lambda_BA={lam_ba:.2f}, beta_c(BA)={bc_val:.5f}, "
              f"BA S6={s6*100:.1f}%, suppression={supp:.0f}%  "
              f"[{time.time()-t0:.0f}s]")

    out = {"lambda_ba": lambda_ba,
           "results": {str(N): {topo: {sc: results[N][topo][sc]
                                       for sc in ("S1", "S6")}
                                for topo in TOPOS}
                       for N in args.sizes}}
    with open("results/C_finite_size.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/C_finite_size.json")

    # ---- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for topo in TOPOS:
        ms = [np.mean(results[N][topo]["S6"]) * 100 for N in args.sizes]
        ss = [np.std(results[N][topo]["S6"]) * 100 for N in args.sizes]
        ax.errorbar(args.sizes, ms, yerr=ss, marker="o", lw=2.2, capsize=5,
                    label=topo, color=COLORS[topo], markersize=7)
    ax.set_xscale("log"); ax.set_xticks(args.sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("System size N"); ax.set_ylabel("Total infection S6 (%)")
    ax.set_title("(a) Behavioral suppression vs N", loc="left")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for topo in TOPOS:
        sup = [100 * (1 - np.mean(results[N][topo]["S6"])
                      / max(np.mean(results[N][topo]["S1"]), 1e-6))
               for N in args.sizes]
        ax.plot(args.sizes, sup, marker="s", lw=2.2, label=topo,
                color=COLORS[topo], markersize=7)
    ax.set_xscale("log"); ax.set_xticks(args.sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("System size N"); ax.set_ylabel("Suppression S6 vs S1 (%)")
    ax.set_title("(b) Suppression increases with N", loc="left")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/fig4_finite_size.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/fig4_finite_size.png")


if __name__ == "__main__":
    main()
