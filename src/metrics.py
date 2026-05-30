"""
metrics.py
==========
Spatial-coordination and behavioral-diversity metrics used in the
self-organization (Section 4.3) and MMCA-comparison (Section 4.x) analyses.

All functions take NumPy arrays / SciPy sparse adjacency matrices and return
plain floats, so they are safe to call inside the simulation loop.
"""

import numpy as np


def morans_i(adj, x):
    """
    Moran's I spatial autocorrelation of a node-level signal x over a graph.

        I = (N / W) * (z^T A z) / (z^T z),   z = x - mean(x),  W = sum(A)

    Parameters
    ----------
    adj : scipy.sparse matrix (N x N), binary adjacency.
    x   : array of length N (e.g. mask indicator in {0,1}).

    Returns
    -------
    float
        Moran's I. ~0 means no spatial autocorrelation; positive means
        neighbors are more alike than chance. Returns 0.0 if x is constant.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    z = x - x.mean()
    den = float(z @ z)
    W = float(adj.sum())
    if den == 0.0 or W == 0.0:
        return 0.0
    num = float(z @ adj.dot(z))
    return (n / W) * (num / den)


def morans_i_pvalue(adj, x, n_perm=499, seed=0):
    """
    Permutation p-value for Moran's I (one-sided, testing positive autocorr).

    Returns
    -------
    tuple (I_observed, p_value)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    I_obs = morans_i(adj, x)
    if np.allclose(x, x[0]):
        return I_obs, 1.0
    count = 0
    xp = x.copy()
    for _ in range(n_perm):
        rng.shuffle(xp)
        if morans_i(adj, xp) >= I_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(I_obs), float(p)


def neighbor_mutual_information(adj, x, deg=None, n_bins=4):
    """
    Mutual information (bits) between a node's binary state x_i and the binned
    mean state of its neighbors. Captures local agent-agent coordination that
    is independent of the global stimulus.

    Returns
    -------
    float
        MI in bits (>= 0). 0 means neighbor state is uninformative about x_i.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if deg is None:
        deg = np.asarray(adj.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1)
    nbr_mean = np.asarray(adj.dot(x)).ravel() / deg_safe

    # Discretize: x is binary (2 states); neighbor mean into n_bins
    xb = (x > 0.5).astype(int)
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    yb = np.clip(np.digitize(nbr_mean, edges) - 1, 0, n_bins - 1)

    # Joint and marginal distributions
    joint = np.zeros((2, n_bins))
    for xi, yi in zip(xb, yb):
        joint[xi, yi] += 1
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    mi = 0.0
    for i in range(2):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i, 0] > 0 and py[0, j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i, 0] * py[0, j]))
    return float(max(mi, 0.0))


def neighbor_mi_pvalue(adj, x, deg=None, n_perm=499, n_bins=4, seed=0):
    """Permutation p-value for neighbor mutual information."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    mi_obs = neighbor_mutual_information(adj, x, deg, n_bins)
    if np.allclose(x, x[0]):
        return mi_obs, 1.0
    count = 0
    xp = x.copy()
    for _ in range(n_perm):
        rng.shuffle(xp)
        if neighbor_mutual_information(adj, xp, deg, n_bins) >= mi_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(mi_obs), float(p)


def combination_entropy(*binary_arrays):
    """
    Joint Shannon entropy (bits) of the combination of several binary
    behavioral channels.

    For two channels (mask, distancing) the maximum is 2 bits; for three
    channels (mask, distancing, vaccination) the maximum is 3 bits.

    Returns
    -------
    float
        Entropy in bits. 0 means all agents share an identical combination
        (degenerate behavioral landscape).
    """
    arrays = [np.asarray(a).astype(int) for a in binary_arrays]
    n = arrays[0].size
    if n == 0:
        return 0.0
    # Encode each agent's combination as an integer code
    code = np.zeros(n, dtype=int)
    for k, a in enumerate(arrays):
        code += (a > 0).astype(int) << k
    _, counts = np.unique(code, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def protection_level_distribution(mask, dist, eta_m=0.65, eta_d=0.50):
    """
    Per-agent transmission-survival factor (1 - m*eta_m)*(1 - d*eta_d) and its
    diversity. A single-channel model collapses this to a two-point
    distribution; a multi-channel model produces several distinct levels.

    Returns
    -------
    dict with keys:
        levels      - sorted unique protection-survival factors,
        n_levels    - number of distinct levels,
        entropy_bits- Shannon entropy of the level distribution,
        mean        - population-mean survival factor.
    """
    mask = np.asarray(mask).astype(bool)
    dist = np.asarray(dist).astype(bool)
    factor = (1.0 - mask * eta_m) * (1.0 - dist * eta_d)
    vals, counts = np.unique(np.round(factor, 6), return_counts=True)
    p = counts / counts.sum()
    ent = float(-np.sum(p * np.log2(p))) if p.size > 1 else 0.0
    return {
        "levels":       [float(v) for v in vals],
        "n_levels":     int(vals.size),
        "entropy_bits": ent,
        "mean":         float(factor.mean()),
    }
