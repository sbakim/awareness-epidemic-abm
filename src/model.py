"""
model.py
========
Vectorized coupled awareness-epidemic model with autonomous agent decisions.

Reference:
    Bakim, S. (2025). Phase transitions and emergent behavioral coordination
    in coupled awareness-epidemic dynamics on complex networks.
    Physica A: Statistical Mechanics and its Applications.

Model equations:
    Mask decision    : P(mask_i,t)  = sigma(a1*c_i + a2*l_i(t) + a3*theta_t + a4)   [coef in MASK_COEF]
    Distance decision: P(dist_i,t)  = sigma(a1*c_i + a2*l_i(t) + a3*theta_t + a4)   [coef in DIST_COEF]
    Vaccine decision : P(vacc_i,t)  = sigma(1.5*c_i + 2.0*l_i(t) + 2.5*f_vacc_i - 3.0)
    Transmission     : beta_eff(i,j)= beta0*(1-m_i*eta_m)(1-m_j*eta_m)(1-d_i*eta_d)*w_ij(t)
    Awareness update : l_i(t+1)     = l_i(t) + alpha*Dl_i + delta_loc*theta_i^loc - mu*l_i(t)
    Network rewiring : w_ij(t+1)    = w_ij(0)*(1-kappa*theta)*(1-0.5*(c_i+c_j))*(1-psi*max(l_i,l_j))

Scenarios:
    S1 - No intervention (passive nodes, static network)
    S2 - Homogeneous intervention (uniform edge reduction + vaccination)
    S3 - Centrality-targeted intervention (betweenness-based)
    S4 - Endogenous behavioral (sigmoid decisions, static awareness)
    S5 - Dual spreading (S4 + dynamic awareness diffusion)
    S6 - Full co-evolution (S5 + awareness-dependent network rewiring)
"""

import numpy as np
import networkx as nx
from scipy import sparse

from .networks import generate_network, sigmoid
from .metrics import morans_i


# -- Canonical sigmoid decision coefficients (single source of truth) -------
# (a1=compliance, a2=awareness, a3=prevalence/peer, a4=bias)
# MAIN model: recalibrated so adoption matches behavioral-survey anchors
#   (~10% baseline masking at zero prevalence, ~45% at 25% prevalence;
#    vaccination uptake ~5% with no vaccinated peers, ~50% at 60% peer uptake).
MASK_COEF = (2.0, 1.5, 8.44, -3.91)
DIST_COEF = (1.8, 1.2, 9.1, -4.06)
VACC_COEF = (1.5, 2.0, 4.91, -4.49)
# Original heuristic set retained for the robustness comparison (analysis E).
MASK_COEF_HEURISTIC = (2.0, 1.5, 3.0, -2.0)
DIST_COEF_HEURISTIC = (1.8, 1.2, 4.0, -2.5)
VACC_COEF_HEURISTIC = (1.5, 2.0, 2.5, -3.0)

# -- Default parameters (Table 1 in paper) ---------------------------------
DEFAULT_PARAMS = {
    "beta_0":      0.12,   # Base transmission rate
    "gamma":       0.10,   # Recovery rate (mean 10-day illness)
    "eta_m":       0.65,   # Mask efficacy (bilateral)
    "eta_d":       0.50,   # Distancing efficacy
    "alpha":       0.12,   # Awareness diffusion rate
    "delta_local": 0.35,   # Local infection feedback strength
    "mu":          0.04,   # Awareness decay rate
    "kappa":       0.60,   # Prevalence-contact sensitivity (S6)
    "psi":         0.35,   # Awareness-contact reduction (S6)
    "t_start":     5,      # Intervention onset day
    "n_init_inf":  5,      # Initial infected agents
    "vacc_rate":   0.02,   # Uniform vaccination rate (S2)
    # Sigmoid decision coefficients (a1=compliance, a2=awareness, a3=prevalence/peer, a4=bias).
    # Defaults are the published values; override with the recalibrated set to
    # test robustness to the heuristic coefficient choice (Reviewer #1.1 / #2.1).
    "mask_coef": MASK_COEF,
    "dist_coef": DIST_COEF,
    "vacc_coef": VACC_COEF,
    # S3 aggressiveness (defaults reproduce the paper's S3):
    "s3_hub_frac": 0.15,   # fraction of top-centrality nodes targeted
    "s3_hub_reduction": 0.80,   # edge-weight reduction on hub edges
    "s3_other_reduction": 0.40,  # edge-weight reduction on other edges
    "s3_vacc_mult": 2.0,   # vaccination-rate multiplier on hub nodes
}

VALID_TOPOLOGIES = ("BA", "WS", "ER", "MOD")
VALID_SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")


class VectorizedEpidemicModel:
    """
    Coupled awareness-epidemic agent-based model.

    Parameters
    ----------
    topology : str  - one of 'BA','WS','ER','MOD'
    N : int         - number of agents (default 500)
    scenario : str  - one of 'S1'..'S6' (default 'S5')
    T : int         - simulation days (default 200)
    seed : int      - random seed (default 42)
    params : dict or None - overrides for DEFAULT_PARAMS
    G : networkx.Graph or None - optional external graph (overrides topology)
    """

    def __init__(self, topology="BA", N=500, scenario="S5",
                 T=200, seed=42, params=None, G=None):

        if topology not in VALID_TOPOLOGIES and G is None:
            raise ValueError(f"topology must be one of {VALID_TOPOLOGIES}")
        if scenario not in VALID_SCENARIOS:
            raise ValueError(f"scenario must be one of {VALID_SCENARIOS}")

        self.topology = topology
        self.scenario = scenario
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.p = {**DEFAULT_PARAMS, **(params or {})}

        # -- Build network ------------------------------------------------
        if G is None:
            G = generate_network(topology, N, seed)
        self.N = G.number_of_nodes()
        n = self.N

        self.adj = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
        # binary adjacency for spatial metrics
        self.adj_bin = self.adj.copy()
        self.adj_bin.data[:] = 1.0
        self.degrees = np.array([d for _, d in G.degree()], dtype=np.float64)

        # Initial edge weights ~ Uniform[0.5, 1.0]
        rng_w = np.random.default_rng(seed + 1000)
        self.W0 = self.adj.copy()
        self.W0.data = rng_w.uniform(0.5, 1.0, size=self.W0.data.shape)
        self.W0 = (self.W0 + self.W0.T) / 2          # symmetrize
        self.W = self.W0.copy()

        # Edge index list (upper triangle only)
        rows, cols = self.W0.nonzero()
        mask_tri = rows < cols
        self.edge_i = rows[mask_tri]
        self.edge_j = cols[mask_tri]

        # Hub detection for S3: top fraction by betweenness centrality
        bc = nx.betweenness_centrality(G)
        bc_arr = np.array([bc[i] for i in range(n)])
        n_top = max(1, int(self.p["s3_hub_frac"] * n))
        self.high_bc = np.zeros(n, dtype=bool)
        self.high_bc[np.argsort(bc_arr)[-n_top:]] = True

        # -- Initialize agent states --------------------------------------
        self.compliance = self.rng.beta(2, 2, size=n)        # ~ Beta(2,2)
        self.info_level = self.rng.beta(2, 3, size=n)        # ~ Beta(2,3), mean 0.4

        self.health = np.zeros(n, dtype=np.int8)             # 0=S,1=I,2=R
        init_idx = self.rng.choice(n, size=min(self.p["n_init_inf"], n),
                                   replace=False)
        self.health[init_idx] = 1

        self.vaccinated = np.zeros(n, dtype=bool)
        self.wearing_mask = np.zeros(n, dtype=bool)
        self.distancing = np.zeros(n, dtype=bool)

        # -- Data collection ----------------------------------------------
        self._ts = {k: [] for k in [
            "S", "I", "R",
            "mask_rate", "dist_rate", "vacc_frac",
            "mean_awareness", "std_awareness",
            "order_param", "entropy_bits", "spatial_sync",
        ]}

    # -- Public interface --------------------------------------------------

    def run(self):
        """Run the full simulation for T steps; return time-series dict."""
        for t in range(1, self.T + 1):
            self._step(t)
        return self._ts

    def get_summary(self):
        """Return scalar summary statistics after run()."""
        I = np.array(self._ts["I"])
        return {
            "topology":      self.topology,
            "scenario":      self.scenario,
            "N":             self.N,
            "peak_inf":      float(I.max()),
            "peak_day":      int(I.argmax()) + 1,
            "total_inf":     float(self._ts["R"][-1]),
            "final_S":       float(self._ts["S"][-1]),
            "vacc_frac":     float(self._ts["vacc_frac"][-1]),
            "max_mask_rate": float(max(self._ts["mask_rate"])),
            "max_awareness": float(max(self._ts["mean_awareness"])),
        }

    # -- Private step methods ---------------------------------------------

    def _step(self, t):
        n = self.N
        ts = self.p["t_start"]
        is_I = (self.health == 1)
        is_S = (self.health == 0)
        prev = float(is_I.sum() / n)

        if self.scenario in ("S4", "S5", "S6") and t >= ts:
            self._behavioral_update(prev)
        if self.scenario in ("S5", "S6") and t >= ts:
            self._awareness_diffusion(is_I)
        if self.scenario in ("S2", "S3") and t >= ts:
            self._exogenous_intervention(is_S)
        if self.scenario == "S6" and t >= ts:
            self._network_adaptation(prev)

        self._transmission()
        self._recovery()
        self._record(prev)

    def _behavioral_update(self, prev):
        c, l, n = self.compliance, self.info_level, self.N
        mk, dk, vk = self.p["mask_coef"], self.p["dist_coef"], self.p["vacc_coef"]
        p_mask = sigmoid(mk[0]*c + mk[1]*l + mk[2]*prev + mk[3])         # Eq. 1
        self.wearing_mask = self.rng.random(n) < p_mask
        p_dist = sigmoid(dk[0]*c + dk[1]*l + dk[2]*prev + dk[3])         # Eq. 2
        self.distancing = self.rng.random(n) < p_dist
        frac_vacc = (self.adj.dot(self.vaccinated.astype(np.float64))
                     / np.maximum(self.degrees, 1))
        p_vacc = sigmoid(vk[0]*c + vk[1]*l + vk[2]*frac_vacc + vk[3])    # Eq. 3
        eligible = (self.health == 0) & (~self.vaccinated)
        self.vaccinated |= eligible & (self.rng.random(n) < p_vacc)

    def _awareness_diffusion(self, is_I):
        p, l = self.p, self.info_level
        deg_s = np.maximum(self.degrees, 1)
        diffusion = p["alpha"] * (self.adj.dot(l) / deg_s - l)
        local_prev = self.adj.dot(is_I.astype(np.float64)) / deg_s
        feedback = p["delta_local"] * local_prev
        decay = -p["mu"] * l
        self.info_level = np.clip(l + diffusion + feedback + decay, 0.0, 1.0)

    def _exogenous_intervention(self, is_S):
        p = self.p
        if self.scenario == "S2":
            self.W = self.W0 * 0.5
            eligible = is_S & (~self.vaccinated)
            self.vaccinated |= eligible & (self.rng.random(self.N) < p["vacc_rate"])
        elif self.scenario == "S3":
            ei, ej = self.edge_i, self.edge_j
            is_hub_e = self.high_bc[ei] | self.high_bc[ej]
            hub_keep = 1.0 - p["s3_hub_reduction"]
            other_keep = 1.0 - p["s3_other_reduction"]
            factors = np.where(is_hub_e, hub_keep, other_keep)
            w0v = np.array(self.W0[ei, ej]).flatten()
            w_new = w0v * factors
            all_i = np.concatenate([ei, ej])
            all_j = np.concatenate([ej, ei])
            self.W = sparse.csr_matrix(
                (np.concatenate([w_new, w_new]), (all_i, all_j)),
                shape=(self.N, self.N))
            rate = np.where(self.high_bc,
                            p["vacc_rate"] * p["s3_vacc_mult"], p["vacc_rate"])
            eligible = is_S & (~self.vaccinated)
            self.vaccinated |= eligible & (self.rng.random(self.N) < rate)

    def _network_adaptation(self, prev):
        p = self.p
        ei, ej = self.edge_i, self.edge_j
        prev_fac = 1.0 - p["kappa"] * prev
        comp_fac = 1.0 - 0.5 * (self.compliance[ei] + self.compliance[ej])
        aware_fac = 1.0 - p["psi"] * np.maximum(
            self.info_level[ei], self.info_level[ej])
        w0v = np.array(self.W0[ei, ej]).flatten()
        w_new = np.clip(w0v * prev_fac * comp_fac * aware_fac, 0.05, 1.0)
        all_i = np.concatenate([ei, ej])
        all_j = np.concatenate([ej, ei])
        self.W = sparse.csr_matrix(
            (np.concatenate([w_new, w_new]), (all_i, all_j)),
            shape=(self.N, self.N))

    def _transmission(self):
        p = self.p
        is_S = (self.health == 0) & (~self.vaccinated)
        is_I = (self.health == 1)
        if not is_I.any() or not is_S.any():
            return
        mask_f = np.where(self.wearing_mask, 1.0 - p["eta_m"], 1.0)
        dist_f = np.where(self.distancing, 1.0 - p["eta_d"], 1.0)
        inf_strength = is_I.astype(np.float64) * p["beta_0"] * mask_f
        force = self.W.dot(inf_strength) * mask_f * dist_f
        p_infect = 1.0 - np.exp(-force)
        new_infected = is_S & (self.rng.random(self.N) < p_infect)
        self.health[new_infected] = 1

    def _recovery(self):
        is_I = (self.health == 1)
        self.health[is_I & (self.rng.random(self.N) < self.p["gamma"])] = 2

    def _record(self, prev):
        n, ts = self.N, self._ts
        ts["S"].append(float((self.health == 0).sum() / n))
        ts["I"].append(float((self.health == 1).sum() / n))
        ts["R"].append(float((self.health == 2).sum() / n))
        ts["mask_rate"].append(float(self.wearing_mask.mean()))
        ts["dist_rate"].append(float(self.distancing.mean()))
        ts["vacc_frac"].append(float(self.vaccinated.mean()))
        ts["mean_awareness"].append(float(self.info_level.mean()))
        ts["std_awareness"].append(float(self.info_level.std()))

        # Order parameter: |2<m> - 1|
        mean_m = self.wearing_mask.mean()
        ts["order_param"].append(float(abs(2 * mean_m - 1)))

        # Behavioral combination entropy over (mask, distancing): max 2 bits
        m, d = self.wearing_mask, self.distancing
        counts = np.array([
            (m & d).sum(), (m & ~d).sum(), (~m & d).sum(), (~m & ~d).sum()
        ], dtype=float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ts["entropy_bits"].append(float(-np.sum(probs * np.log2(probs))))

        # Spatial synchronization = Moran's I of the mask indicator
        # (genuine spatial autocorrelation; 0 if no masking variation)
        ts["spatial_sync"].append(morans_i(self.adj_bin,
                                            self.wearing_mask.astype(float)))
