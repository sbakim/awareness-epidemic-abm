"""
networks.py
===========
Network generation, the logistic sigmoid decision function, and structural
statistics (including the network heterogeneity ratio Lambda = <k^2>/<k>).
"""

import numpy as np
import networkx as nx


def sigmoid(x):
    """
    Numerically stable logistic sigmoid sigma(x) = 1 / (1 + e^{-x}).

    Uses the positive branch for x >= 0 and the negative branch for x < 0
    to avoid overflow in exp().
    """
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
    )


def generate_network(topology, N=500, seed=42):
    """
    Generate a connected network with <k> ~ 8.

    Parameters
    ----------
    topology : str
        'BA'  - Barabasi-Albert (m=4, power-law degree, scale-free hubs)
        'WS'  - Watts-Strogatz (k=8, p=0.1, small-world, clustered)
        'ER'  - Erdos-Renyi (p = 8/(N-1), Poisson degree, homogeneous)
        'MOD' - Stochastic block model (5 communities, p_in = 10*p_out)
    N : int
        Target number of nodes.
    seed : int
        Random seed.

    Returns
    -------
    networkx.Graph
        Connected graph with integer node labels 0, ..., N-1.
    """
    if topology == "BA":
        G = nx.barabasi_albert_graph(N, m=4, seed=seed)

    elif topology == "WS":
        G = nx.watts_strogatz_graph(N, k=8, p=0.1, seed=seed)

    elif topology == "ER":
        G = nx.erdos_renyi_graph(N, p=8.0 / (N - 1), seed=seed)

    elif topology == "MOD":
        bs = N // 5
        sizes = [bs] * 5
        sizes[-1] += N - sum(sizes)                       # remainder in last block
        # Solve for p_out so that the expected degree is ~8 with p_in = 10*p_out:
        #   <k> = p_in*(bs-1) + p_out*(N-bs) = p_out*(10*(bs-1) + (N-bs))
        p_out = 8.0 / (10 * (bs - 1) + (N - bs))
        p_in = 10 * p_out
        probs = [[p_in if i == j else p_out for j in range(5)]
                 for i in range(5)]
        G = nx.stochastic_block_model(sizes, probs, seed=seed)

    else:
        raise ValueError(
            f"Unknown topology '{topology}'. Choose from: BA, WS, ER, MOD")

    # Ensure connectivity - keep largest connected component
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


def network_stats(G):
    """
    Compute structural statistics of a graph.

    Returns
    -------
    dict with keys: N, E, k_mean, k2_mean, Lambda, clustering, diameter.
    """
    degs = [d for _, d in G.degree()]
    k1 = float(np.mean(degs))
    k2 = float(np.mean([d ** 2 for d in degs]))
    try:
        diam = nx.diameter(G)
    except Exception:
        diam = -1

    return {
        "N":          G.number_of_nodes(),
        "E":          G.number_of_edges(),
        "k_mean":     k1,
        "k2_mean":    k2,
        "Lambda":     k2 / k1,
        "clustering": float(nx.average_clustering(G)),
        "diameter":   diam,
    }


def measure_lambda(topology, N=500, n_seeds=30):
    """
    Empirically measure the network heterogeneity ratio Lambda = <k^2>/<k>
    by averaging over n_seeds independent realizations of a topology.

    This replaces hard-coded Lambda values. For N=500 the measured values are
    approximately:  BA ~ 15.6,  WS ~ 8.1,  ER ~ 9.0,  MOD ~ 9.6.

    Returns
    -------
    dict with keys 'mean' and 'std'.
    """
    vals = [network_stats(generate_network(topology, N, seed=s))["Lambda"]
            for s in range(n_seeds)]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def measure_all_lambdas(N=500, n_seeds=30, topologies=("BA", "WS", "ER", "MOD")):
    """Return {topology: mean Lambda} measured over n_seeds realizations."""
    return {t: measure_lambda(t, N, n_seeds)["mean"] for t in topologies}
