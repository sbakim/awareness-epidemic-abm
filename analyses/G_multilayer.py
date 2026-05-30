"""
G_multilayer.py
===============
Two-layer (multiplex) extension addressing Reviewer #2 c.3:
disease spreads on a PHYSICAL contact layer, while awareness diffuses on a
distinct SOCIAL/information layer. We sweep the topological OVERLAP between the
two layers and measure its effect on epidemic outcome.

Layer routing
-------------
  Physical layer A_phys :  disease transmission (Eq. 4) and the local-epidemic
                           feedback theta_loc that drives awareness (you see your
                           physical contacts fall ill).
  Information layer A_info: awareness social-learning diffusion (the Delta_l term
                           of Eq. 5) and vaccination social conformity (f_vacc).

Overlap construction
--------------------
The information layer has the same node set and (approximately) the same number
of edges as the physical layer. A fraction `overlap` of physical edges are reused
as information edges; the remaining information edges are placed at random. The
realized overlap (Jaccard of edge sets) is measured and reported. overlap=1
reduces to the single-layer model used in the paper; overlap=0 is a fully
decoupled information layer.

Note: at epidemic onset (theta -> 0) there is no awareness to diffuse, so the
analytical threshold beta_c is INVARIANT to the overlap (consistent with
Section 3.2). The overlap therefore acts only on the above-threshold trajectory,
which is what this experiment quantifies.

Outputs: results/G_multilayer.json, figures/_validation_multilayer.png
"""
# === PAPER ROLE ===
# INLINE/VALIDATION: result reported as 1-2 sentences (overlap has no effect); figure kept FOR VALIDATION ONLY. Reviewer #2.3.
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================

import json, os, sys, time
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.networks import generate_network, sigmoid
from src.model import MASK_COEF, DIST_COEF, VACC_COEF

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

BASE = dict(beta_0=0.12, gamma=0.1, eta_m=0.65, eta_d=0.50,
            alpha=0.12, delta_local=0.35, mu=0.04,
            kappa=0.6, psi=0.35, t_start=5, n_init_inf=5, vacc_rate=0.02)


def build_info_layer(G_phys, overlap, seed):
    """Information layer: reuse a fraction `overlap` of physical edges, fill the
    rest with random edges (matched edge count). Returns (G_info, realized_overlap)."""
    rng = np.random.default_rng(seed)
    n = G_phys.number_of_nodes()
    phys_edges = [tuple(sorted(e)) for e in G_phys.edges()]
    m = len(phys_edges)
    phys_set = set(phys_edges)

    n_keep = int(round(overlap * m))
    keep_idx = rng.choice(m, size=n_keep, replace=False) if n_keep > 0 else []
    info_edges = set(phys_edges[i] for i in keep_idx)

    # fill remaining with random edges not already present
    trials = 0
    while len(info_edges) < m and trials < 50 * m:
        a, b = rng.integers(0, n, 2)
        trials += 1
        if a != b:
            info_edges.add(tuple(sorted((int(a), int(b)))))
    G_info = nx.Graph(); G_info.add_nodes_from(range(n)); G_info.add_edges_from(info_edges)
    realized = len(info_edges & phys_set) / max(len(info_edges), 1)
    return G_info, realized


class MultilayerModel:
    def __init__(self, G_phys, G_info, scenario, T=200, seed=42, params=None):
        self.sc, self.T = scenario, T
        self.p = {**BASE, **(params or {})}
        self.rng = np.random.default_rng(seed)
        self.N = n = G_phys.number_of_nodes()
        self.Ap = nx.to_scipy_sparse_array(G_phys, nodelist=range(n), format="csr", dtype=float)
        self.Ai = nx.to_scipy_sparse_array(G_info, nodelist=range(n), format="csr", dtype=float)
        self.degp = np.array(self.Ap.sum(1)).ravel(); self.degp[self.degp == 0] = 1
        self.degi = np.array(self.Ai.sum(1)).ravel(); self.degi[self.degi == 0] = 1
        rng_w = np.random.default_rng(seed + 1000)
        W0 = self.Ap.copy(); W0.data = rng_w.uniform(0.5, 1.0, W0.data.shape)
        self.W0 = (W0 + W0.T) / 2; self.W = self.W0.copy()
        r, c = self.W0.nonzero(); mk = r < c; self.ei, self.ej = r[mk], c[mk]
        self.comp = self.rng.beta(2, 2, n); self.info = self.rng.beta(2, 3, n)
        self.hlth = np.zeros(n, np.int8)
        self.hlth[self.rng.choice(n, min(self.p["n_init_inf"], n), replace=False)] = 1
        self.vacc = np.zeros(n, bool); self.mask_w = np.zeros(n, bool); self.dist = np.zeros(n, bool)

    def run(self):
        p, ts, n = self.p, self.p["t_start"], self.N
        for t in range(1, self.T + 1):
            iI = self.hlth == 1; iS = self.hlth == 0; pr = float(iI.sum() / n)
            if self.sc in ("S4", "S5", "S6") and t >= ts:
                c, l = self.comp, self.info
                self.mask_w = self.rng.random(n) < sigmoid(MASK_COEF[0]*c + MASK_COEF[1]*l + MASK_COEF[2]*pr + MASK_COEF[3])
                self.dist = self.rng.random(n) < sigmoid(DIST_COEF[0]*c + DIST_COEF[1]*l + DIST_COEF[2]*pr + DIST_COEF[3])
                # vaccination conformity over INFORMATION layer
                fv = self.Ai.dot(self.vacc.astype(float)) / self.degi
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < sigmoid(VACC_COEF[0]*c + VACC_COEF[1]*l + VACC_COEF[2]*fv + VACC_COEF[3]))
            if self.sc in ("S5", "S6") and t >= ts:
                l = self.info
                # social-learning diffusion over INFORMATION layer:
                diffusion = p["alpha"] * (self.Ai.dot(l) / self.degi - l)
                # local epidemic feedback over PHYSICAL layer:
                local_prev = self.Ap.dot(iI.astype(float)) / self.degp
                self.info = np.clip(l + diffusion + p["delta_local"]*local_prev - p["mu"]*l, 0, 1)
            if self.sc == "S6" and t >= ts:
                ei, ej = self.ei, self.ej
                wn = np.clip(np.array(self.W0[ei, ej]).flatten()
                             * (1 - p["kappa"]*pr) * (1 - .5*(self.comp[ei] + self.comp[ej]))
                             * (1 - p["psi"]*np.maximum(self.info[ei], self.info[ej])), 0.05, 1)
                ai, aj = np.concatenate([ei, ej]), np.concatenate([ej, ei])
                self.W = sparse.csr_matrix((np.concatenate([wn, wn]), (ai, aj)), shape=(n, n))
            # transmission on PHYSICAL layer
            iSe = (self.hlth == 0) & ~self.vacc; iI2 = self.hlth == 1
            if iI2.any() and iSe.any():
                mf = np.where(self.mask_w, 1 - p["eta_m"], 1.0); df = np.where(self.dist, 1 - p["eta_d"], 1.0)
                F = self.W.dot(iI2.astype(float) * p["beta_0"] * mf) * mf * df
                self.hlth[iSe & (self.rng.random(n) < 1 - np.exp(-F))] = 1
            self.hlth[(self.hlth == 1) & (self.rng.random(n) < p["gamma"])] = 2
        return float((self.hlth == 2).sum() / self.N)


def main(topologies=("BA", "WS", "ER", "MOD"),
         overlaps=(0.0, 0.25, 0.5, 0.75, 1.0), n_runs=20, N=500):
    out = {"overlaps": list(overlaps), "n_runs": n_runs, "N": N, "data": {}}
    t0 = time.time()
    for topo in topologies:
        G_phys = generate_network(topo, N, seed=0)
        # S1 baseline (no awareness; info layer irrelevant)
        s1 = np.mean([MultilayerModel(G_phys, G_phys, "S1", seed=r).run() for r in range(n_runs)])
        rows = {}
        for o in overlaps:
            real_ov = []
            s5, s6 = [], []
            for r in range(n_runs):
                G_info, ro = build_info_layer(G_phys, o, seed=100 + r)
                real_ov.append(ro)
                s5.append(MultilayerModel(G_phys, G_info, "S5", seed=r).run())
                s6.append(MultilayerModel(G_phys, G_info, "S6", seed=r).run())
            rows[o] = {"realized_overlap": float(np.mean(real_ov)),
                       "S5": float(np.mean(s5)), "S5_std": float(np.std(s5)),
                       "S6": float(np.mean(s6)), "S6_std": float(np.std(s6)),
                       "S6_suppression": float(100*(1 - np.mean(s6)/max(s1, 1e-9)))}
        out["data"][topo] = {"S1": float(s1), "rows": rows}
        print(f"{topo}: S1={s1:.1%} | " + " ".join(
            f"o={o}:S6={rows[o]['S6']:.1%}(sup{rows[o]['S6_suppression']:.0f}%)" for o in overlaps)
            + f"  [{time.time()-t0:.0f}s]")

    # --- regime robustness: does overlap matter if awareness were diffusion-dominated? ---
    print("\nRegime robustness (S6, o=0 vs o=1, 15 runs):")
    regimes = {"feedback_dominated": dict(alpha=0.12, delta_local=0.35, mu=0.04),
               "diffusion_dominated": dict(alpha=0.60, delta_local=0.05, mu=0.02)}
    reg_out = {}
    for topo in ("BA", "WS"):
        Gp = generate_network(topo, N, seed=0)
        reg_out[topo] = {}
        for rname, rp in regimes.items():
            d = {}
            for o in (0.0, 1.0):
                vals = []
                for r in range(15):
                    Gi, _ = build_info_layer(Gp, o, seed=100 + r)
                    vals.append(MultilayerModel(Gp, Gi, "S6", seed=r, params=rp).run())
                d[o] = float(np.mean(vals))
            reg_out[topo][rname] = {"o0": d[0.0], "o1": d[1.0], "delta_pp": (d[0.0]-d[1.0])*100}
            print(f"  {topo} {rname:20s}: o=0 S6={d[0.0]:.1%}  o=1 S6={d[1.0]:.1%}  delta={(d[0.0]-d[1.0])*100:+.1f}pp")
    out["regime_robustness"] = reg_out

    with open("results/G_multilayer.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/G_multilayer.json")

    # figure: S6 suppression vs overlap
    colors = {"BA": "#E24B4A", "WS": "#1D9E75", "ER": "#378ADD", "MOD": "#9B59B6"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for topo in topologies:
        ov = [out["data"][topo]["rows"][o]["realized_overlap"] for o in overlaps]
        sup = [out["data"][topo]["rows"][o]["S6_suppression"] for o in overlaps]
        s6 = [out["data"][topo]["rows"][o]["S6"]*100 for o in overlaps]
        axes[0].plot(ov, sup, "o-", color=colors[topo], lw=2, label=topo)
        axes[1].plot(ov, s6, "o-", color=colors[topo], lw=2, label=topo)
    axes[0].set_xlabel("Layer overlap (Jaccard of edge sets)")
    axes[0].set_ylabel("S6 suppression vs S1 (%)")
    axes[0].set_title("(a) Suppression vs awareness-layer overlap", loc="left"); axes[0].set_ylim(0, 100)
    axes[1].set_xlabel("Layer overlap (Jaccard of edge sets)")
    axes[1].set_ylabel("S6 total infection (%)")
    axes[1].set_title("(b) Total infection vs overlap", loc="left"); axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
    plt.tight_layout(pad=1.5)
    plt.savefig("figures/_validation_multilayer.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/_validation_multilayer.png")


if __name__ == "__main__":
    main()
