"""
empirical_validation.py  (REVISED)
==================================
Validation on REAL high-resolution contact networks (Reviewer #1 c.3,
Reviewer #2 c.2). Three public SocioPatterns datasets are used directly:

    data/primaryschool.csv     Stehle et al. 2011,  PLoS ONE 6(8):e23176
    data/highschool_2013.csv   Fournet & Barrat 2014, PLoS ONE 9:e107878
    data/hospital_ward.csv     Vanhems et al. 2013, PLoS ONE 8:e73970

Run `python prepare_data.py` once to download+build these files
(or place them in data/ manually). If a file is missing the script falls back
to a clearly-labeled calibrated-synthetic surrogate.

KEY REVISIONS vs the previous version
-------------------------------------
* Edge weights are NO LONGER U[0.5,1] placeholders. They are set to the
  measured cumulative contact duration  w_ij = clip(count_ij / p95, 0, 1),
  so transmission is proportional to real interaction strength and fleeting
  20-second contacts contribute negligibly (the heavy-tailed contact-strength
  distribution is preserved). This is the physically correct meaning of w_ij
  in beta_eff = beta0 * (...) * w_ij.
* A contact-duration threshold sensitivity sweep is added: the static network
  is rebuilt keeping only edges with cumulative duration >= theta_min intervals.
  This shows how the conclusions depend on the operational definition of a
  "contact" (a standard robustness check for high-resolution proximity data).
* beta_c uses the MEASURED Lambda of each real network (no hard-coding).

Produces results/empirical_validation.json and figures/fig5_empirical_validation.png
"""
# === PAPER ROLE ===
# MAIN TEXT: Fig 5 (three real SocioPatterns contact networks). Reviewer #1.3 / #2.2.
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================


import json, os, sys, time
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.networks import sigmoid, network_stats
from src.model import MASK_COEF, DIST_COEF, VACC_COEF
from src.analytical import phi as phi_fn

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

BASE = dict(beta_0=0.12, gamma=0.1, eta_m=0.65, eta_d=0.50,
            alpha=0.12, delta_local=0.35, mu=0.04,
            kappa=0.6, psi=0.35, t_start=5, n_init_inf=5, vacc_rate=0.02)
SCS = ["S1", "S2", "S3", "S4", "S5", "S6"]

REFERENCES = {
    "PrimarySchool": "Stehle et al. 2011, PLoS ONE 6(8):e23176",
    "HighSchool":    "Fournet & Barrat 2014, PLoS ONE 9:e107878",
    "HospitalWard":  "Vanhems et al. 2013, PLoS ONE 8:e73970",
}
DATA_FILES = {
    "PrimarySchool": "primaryschool.csv",
    "HighSchool":    "highschool_2013.csv",
    "HospitalWard":  "hospital_ward.csv",
}


# ----------------------------------------------------------------------
#  Data loading: weighted contact network from `i j [w]` edge list
# ----------------------------------------------------------------------
def load_contact_counts(path):
    """Return {(i,j): cumulative_duration}. Sums duplicate pairs; if no weight
    column is present, each row counts as one 20-second contact interval."""
    from collections import Counter
    c = Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            p = line.replace(",", " ").split()
            if len(p) < 2:
                continue
            i, j = p[0], p[1]
            if i == j:
                continue
            w = float(p[2]) if len(p) >= 3 else 1.0
            c[tuple(sorted((i, j)))] += w
    return dict(c)


def graph_from_counts(counts, theta_min=1):
    """Build a connected weighted graph keeping edges with duration>=theta_min.
    Edge attribute 'count' holds cumulative duration; returns relabeled graph."""
    items = [(a, b, w) for (a, b), w in counts.items() if w >= theta_min]
    G = nx.Graph()
    G.add_weighted_edges_from([(a, b, w) for a, b, w in items], weight="count")
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def duration_weight_matrix(G, floor=0.0, cap_pct=95):
    """Edge weights = clip(count / percentile(count,cap_pct), floor, 1).
    Transmission proportional to real contact duration."""
    adj = nx.to_scipy_sparse_array(G, format="csr", weight="count", dtype=float)
    scale = np.percentile(adj.data, cap_pct) if adj.data.size else 1.0
    W = adj.copy()
    W.data = np.clip(adj.data / max(scale, 1e-9), floor, 1.0)
    return (W + W.T) / 2


def build_synthetic(name):
    """Calibrated-synthetic fallback (labeled) if a real file is absent."""
    if name == "PrimarySchool":
        G = nx.stochastic_block_model([24] * 10,
            [[0.55 if i == j else 0.02 for j in range(10)] for i in range(10)], seed=42)
    elif name == "HighSchool":
        G = nx.stochastic_block_model([35] * 9,
            [[0.48 if i == j else 0.04 for j in range(9)] for i in range(9)], seed=44)
    else:
        G = nx.stochastic_block_model([29, 46], [[0.10, 0.40], [0.40, 0.25]], seed=43)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(7)
    adj = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    W = adj.copy(); W.data = rng.uniform(0.5, 1.0, W.data.shape)
    return G, (W + W.T) / 2


def get_networks():
    """Return {name: (Graph, W0, is_real, counts_or_None)}."""
    nets = {}
    for name, fname in DATA_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.isfile(path):
            counts = load_contact_counts(path)
            G = graph_from_counts(counts, theta_min=1)
            nets[name] = (G, duration_weight_matrix(G), True, counts)
        else:
            G, W = build_synthetic(name)
            nets[name] = (G, W, False, None)
    return nets


# ----------------------------------------------------------------------
#  Epidemic model on an external weighted graph (identical dynamics to src.model)
# ----------------------------------------------------------------------
class EmpiricalEpiModel:
    def __init__(self, G, W0, scenario, T=200, seed=42, params=None):
        self.sc, self.T = scenario, T
        self.p = {**BASE, **(params or {})}
        self.rng = np.random.default_rng(seed)
        self.N = G.number_of_nodes(); n = self.N
        self.adj = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
        self.deg = np.array([d for _, d in G.degree()], float)
        self.W0 = W0.copy(); self.W = self.W0.copy()
        r, c = self.W0.nonzero(); mk = r < c
        self.ei, self.ej = r[mk], c[mk]
        self.hub = np.zeros(n, bool)
        self.hub[np.argsort(self.deg)[-max(1, int(0.15 * n)):]] = True
        self.comp = self.rng.beta(2, 2, n)
        self.info = self.rng.beta(2, 3, n)
        self.hlth = np.zeros(n, np.int8)
        self.hlth[self.rng.choice(n, min(self.p["n_init_inf"], n), replace=False)] = 1
        self.vacc = np.zeros(n, bool); self.mask_w = np.zeros(n, bool)
        self.dist = np.zeros(n, bool)

    def run(self):
        p, ts, n = self.p, self.p["t_start"], self.N
        for t in range(1, self.T + 1):
            iI = self.hlth == 1; iS = self.hlth == 0; pr = float(iI.sum() / n)
            if self.sc in ("S4", "S5", "S6") and t >= ts:
                c, l = self.comp, self.info
                self.mask_w = self.rng.random(n) < sigmoid(MASK_COEF[0]*c + MASK_COEF[1]*l + MASK_COEF[2]*pr + MASK_COEF[3])
                self.dist = self.rng.random(n) < sigmoid(DIST_COEF[0]*c + DIST_COEF[1]*l + DIST_COEF[2]*pr + DIST_COEF[3])
                fv = self.adj.dot(self.vacc.astype(float)) / np.maximum(self.deg, 1)
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < sigmoid(VACC_COEF[0]*c + VACC_COEF[1]*l + VACC_COEF[2]*fv + VACC_COEF[3]))
            if self.sc in ("S5", "S6") and t >= ts:
                l = self.info; ds = np.maximum(self.deg, 1)
                self.info = np.clip(l + p["alpha"]*(self.adj.dot(l)/ds - l)
                                    + p["delta_local"]*self.adj.dot(iI.astype(float))/ds
                                    - p["mu"]*l, 0, 1)
            if self.sc == "S2" and t >= ts:
                self.W = self.W0 * 0.5
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < p["vacc_rate"])
            if self.sc == "S3" and t >= ts:
                ei, ej = self.ei, self.ej
                facs = np.where(self.hub[ei] | self.hub[ej], 0.20, 0.60)
                wn = np.array(self.W0[ei, ej]).flatten() * facs
                ai, aj = np.concatenate([ei, ej]), np.concatenate([ej, ei])
                self.W = sparse.csr_matrix((np.concatenate([wn, wn]), (ai, aj)), shape=(n, n))
                rate = np.where(self.hub, p["vacc_rate"]*2, p["vacc_rate"])
                self.vacc |= (iS & ~self.vacc) & (self.rng.random(n) < rate)
            if self.sc == "S6" and t >= ts:
                ei, ej = self.ei, self.ej
                wn = np.clip(np.array(self.W0[ei, ej]).flatten()
                             * (1 - p["kappa"]*pr)
                             * (1 - .5*(self.comp[ei] + self.comp[ej]))
                             * (1 - p["psi"]*np.maximum(self.info[ei], self.info[ej])),
                             0.05, 1)
                ai, aj = np.concatenate([ei, ej]), np.concatenate([ej, ei])
                self.W = sparse.csr_matrix((np.concatenate([wn, wn]), (ai, aj)), shape=(n, n))
            iSe = (self.hlth == 0) & ~self.vacc; iI2 = self.hlth == 1
            if iI2.any() and iSe.any():
                mf = np.where(self.mask_w, 1 - p["eta_m"], 1.0)
                df = np.where(self.dist, 1 - p["eta_d"], 1.0)
                F = self.W.dot(iI2.astype(float) * p["beta_0"] * mf) * mf * df
                self.hlth[iSe & (self.rng.random(n) < 1 - np.exp(-F))] = 1
            self.hlth[(self.hlth == 1) & (self.rng.random(n) < p["gamma"])] = 2
        return float((self.hlth == 2).sum() / self.N)


def run_scenarios(G, W0, n_runs, name_idx):
    res = {sc: [] for sc in SCS}
    for sc in SCS:
        for r in range(n_runs):
            seed = 5000 + r*13 + name_idx*1000 + SCS.index(sc)
            res[sc].append(EmpiricalEpiModel(G, W0, sc, T=200, seed=seed).run())
    return res


# ----------------------------------------------------------------------
def main(n_runs=25, thresholds=(1, 3, 5, 10)):
    networks = get_networks()
    any_real = any(v[2] for v in networks.values())

    print("Network statistics (real contact networks, duration-weighted):")
    net_stats = {}
    for name, (G, W0, is_real, _) in networks.items():
        st = network_stats(G)
        bc = BASE["gamma"] / (st["Lambda"] * phi_fn(0, 2/5))
        net_stats[name] = {"N": st["N"], "k_mean": round(st["k_mean"], 1),
                           "Lambda": round(st["Lambda"], 2),
                           "clustering": round(st["clustering"], 3),
                           "beta_c": round(bc, 5), "is_real": is_real}
        tag = "REAL" if is_real else "calibrated-synthetic"
        print(f"  {name:13s}[{tag}] N={st['N']:3d} <k>={st['k_mean']:.1f} "
              f"Lambda={st['Lambda']:.1f} clust={st['clustering']:.3f} beta_c={bc:.5f}")

    # ---- main run: full duration-weighted networks ----------------------
    results, t0 = {}, time.time()
    for ni, (name, (G, W0, _, _)) in enumerate(networks.items()):
        results[name] = run_scenarios(G, W0, n_runs, ni)
        print(f"  {name} done [{time.time()-t0:.0f}s]")

    summary, scen = {}, {}
    for name in results:
        means = {sc: float(np.mean(results[name][sc])) for sc in SCS}
        ordered = all(means[SCS[i]] >= means[SCS[i+1]] for i in range(len(SCS)-1))
        s3, s4 = np.array(results[name]["S3"]), np.array(results[name]["S4"])
        _, p34 = mannwhitneyu(s3, s4, alternative="greater")
        supp = 100*(1 - means["S6"]/max(means["S1"], 1e-6))
        scen[name] = {"ordered": ordered, "s3_s4_gap_pp": float((s3.mean()-s4.mean())*100),
                      "p34": float(p34), "S6_suppression": float(supp)}
        summary[name] = {sc: {"mean": float(np.mean(results[name][sc])),
                              "std": float(np.std(results[name][sc]))} for sc in SCS}
        print(f"  {name}: ordering {'preserved' if ordered else 'DISRUPTED'}, "
              f"S3-S4 gap={(s3.mean()-s4.mean())*100:.1f}pp (p={p34:.1e}), "
              f"S6 suppression={supp:.0f}%")

    # ---- threshold sensitivity (contact-duration definition) ------------
    print("\nContact-duration threshold sensitivity (S1 & S6, 12 runs):")
    thr_sweep = {}
    for name, (_, _, is_real, counts) in networks.items():
        if not is_real:
            continue
        thr_sweep[name] = {}
        for tm in thresholds:
            G = graph_from_counts(counts, theta_min=tm)
            W0 = duration_weight_matrix(G)
            st = network_stats(G)
            s1 = np.mean([EmpiricalEpiModel(G, W0, "S1", seed=s).run() for s in range(12)])
            s6 = np.mean([EmpiricalEpiModel(G, W0, "S6", seed=s).run() for s in range(12)])
            thr_sweep[name][tm] = {"N": st["N"], "k_mean": round(st["k_mean"], 1),
                                   "Lambda": round(st["Lambda"], 2),
                                   "S1": float(s1), "S6": float(s6),
                                   "suppression": float(100*(1 - s6/max(s1, 1e-6)))}
            print(f"  {name:13s} thr>={tm:2d}: <k>={st['k_mean']:4.1f} Lam={st['Lambda']:5.1f} "
                  f"S1={s1:.0%} S6={s6:.0%} supp={100*(1-s6/max(s1,1e-6)):.0f}%")

    out = {"mode": "real" if any_real else "calibrated-synthetic",
           "weighting": "duration  w=clip(count/p95,0,1)",
           "references": REFERENCES, "network_stats": net_stats,
           "scenario_summary": scen, "summary": summary,
           "threshold_sensitivity": thr_sweep}
    with open("results/empirical_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/empirical_validation.json")

    # ---- figure ---------------------------------------------------------
    colors = {"PrimarySchool": "#E24B4A", "HighSchool": "#1D9E75", "HospitalWard": "#378ADD"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]; x = np.arange(len(SCS)); w = 0.25
    for di, name in enumerate(results):
        ms = [summary[name][sc]["mean"]*100 for sc in SCS]
        ss = [summary[name][sc]["std"]*100 for sc in SCS]
        ax.bar(x + (di-1)*w*1.05, ms, w, label=name, color=colors.get(name, "gray"),
               alpha=.85, edgecolor="white")
        ax.errorbar(x + (di-1)*w*1.05, ms, yerr=ss, fmt="none", color="black",
                    capsize=3, elinewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(SCS); ax.set_ylabel("Total infection (%)")
    ax.set_title("(a) Six scenarios on REAL contact networks", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=.3, axis="y")

    ax = axes[1]
    for name in results:
        lam = net_stats[name]["Lambda"]; supp = scen[name]["S6_suppression"]
        ax.scatter(lam, supp, s=200, color=colors.get(name, "gray"), marker="*", zorder=6)
        ax.annotate(name, (lam, supp), xytext=(5, 4), textcoords="offset points",
                    fontsize=9, color=colors.get(name, "gray"), weight="bold")
    ax.set_xlabel(r"Network heterogeneity ratio $\Lambda$")
    ax.set_ylabel("S6 suppression vs S1 (%)")
    ax.set_title("(b) Suppression vs heterogeneity", loc="left"); ax.grid(True, alpha=.3)

    ax = axes[2]
    for name in thr_sweep:
        tms = sorted(thr_sweep[name])
        supp = [thr_sweep[name][t]["suppression"] for t in tms]
        ax.plot(tms, supp, "o-", color=colors.get(name, "gray"), label=name, lw=2)
    ax.set_xlabel("Contact-duration threshold (20s intervals)")
    ax.set_ylabel("S6 suppression vs S1 (%)")
    ax.set_title("(c) Robustness to contact definition", loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/fig5_empirical_validation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/fig5_empirical_validation.png")


if __name__ == "__main__":
    main()
