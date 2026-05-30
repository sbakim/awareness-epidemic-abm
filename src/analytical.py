"""
analytical.py
=============
Analytical framework: behavioral reduction factor Phi, epidemic threshold
beta_c, and effective reproduction number R_eff.

Reference: Section 3 of the paper.

NOTE on Lambda: earlier drafts hard-coded Lambda_BA = 35.5, which is incorrect
for a Barabasi-Albert network with m=4 at N=500. The measured value is
Lambda_BA ~ 15.6 (see networks.measure_lambda). All defaults below use the
measured values; threshold_table() measures Lambda directly from generated
networks unless explicit values are supplied.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from .networks import sigmoid, measure_all_lambdas
from .model import MASK_COEF, DIST_COEF


# Monte Carlo sample for Phi integration over c ~ Beta(2,2)
_C_MC = beta_dist.rvs(2, 2, size=20000, random_state=0)


def phi(theta, l_bar, c_samples=None, mask_coef=MASK_COEF, dist_coef=DIST_COEF):
    """
    Behavioral reduction factor Phi(theta, l_bar)  [Equation 7].

        Phi = E_c[(1 - P_m*eta_m)^2 * (1 - P_d*eta_d)]

    Expectation over compliance c ~ Beta(2,2).
    """
    c = _C_MC if c_samples is None else np.asarray(c_samples)
    mk, dk = mask_coef, dist_coef
    P_m = sigmoid(mk[0]*c + mk[1]*l_bar + mk[2]*theta + mk[3])   # Eq. 1
    P_d = sigmoid(dk[0]*c + dk[1]*l_bar + dk[2]*theta + dk[3])   # Eq. 2
    eta_m, eta_d = 0.65, 0.50
    return float(((1 - P_m * eta_m) ** 2 * (1 - P_d * eta_d)).mean())


def phi_bounds(n_samples=20000, seed=0):
    """
    Bounds of Phi (Proposition 1, part ii), evaluated by MC integration.

    Returns dict {'phi_min': Phi(1,1), 'phi_max': Phi(0,0)}.
    With the paper's sigmoid coefficients: phi_max ~ 0.61, phi_min ~ 0.07-0.10.
    """
    c = beta_dist.rvs(2, 2, size=n_samples, random_state=seed)
    return {"phi_min": phi(1.0, 1.0, c), "phi_max": phi(0.0, 0.0, c)}


def beta_c(Lambda, gamma=0.1, l_bar_0=0.4):
    """
    Analytical epidemic threshold beta_c = gamma / (Lambda * Phi(0, l_bar_0)).
    Default l_bar_0 = 0.4 = E[Beta(2,3)].
    """
    return gamma / (Lambda * phi(0.0, l_bar_0))


def R_eff(beta_0, gamma, Lambda, theta, l_bar):
    """Effective reproduction number R_eff = (beta_0/gamma)*Lambda*Phi(theta,l_bar)."""
    return (beta_0 / gamma) * Lambda * phi(theta, l_bar)


def awareness_steady_state(theta, delta_local=0.35, mu=0.04):
    """Homogeneous-mixing awareness steady state l*(theta) = min(1, delta*theta/mu)."""
    return np.minimum(1.0, delta_local * np.asarray(theta, float) / mu)


def saturation_prevalence(delta_local=0.35, mu=0.04):
    """Prevalence above which the awareness channel saturates: theta_sat = mu/delta."""
    return mu / delta_local


def phi_table(theta_values=(0.00, 0.05, 0.10, 0.20),
              l_bar_values=(0.0, 0.3, 0.5, 0.8),
              n_samples=20000, seed=42):
    """Compute the Phi(theta, l_bar) table (Table 2)."""
    c = beta_dist.rvs(2, 2, size=n_samples, random_state=seed)
    return {(th, lb): phi(th, lb, c)
            for th in theta_values for lb in l_bar_values}


def phi_elasticity(theta, l_bar, delta=1e-4):
    """Elasticity |dPhi/dl_bar| / Phi (Proposition 1, iii)."""
    phi_0 = phi(theta, l_bar)
    phi_up = phi(theta, min(l_bar + delta, 1.0))
    phi_dn = phi(theta, max(l_bar - delta, 0.0))
    dPhi_dl = (phi_up - phi_dn) / (2 * delta)
    return abs(dPhi_dl) / phi_0 if phi_0 > 0 else 0.0


def threshold_table(topologies=None, gamma=0.1, l_bar_0=0.4,
                    N=500, n_seeds=30):
    """
    Compute beta_c for each topology (Table 3).

    Parameters
    ----------
    topologies : dict or None
        Mapping {name: Lambda}. If None, Lambda is MEASURED from generated
        networks (recommended) rather than hard-coded.

    Returns
    -------
    dict mapping topology name -> {'Lambda': ..., 'beta_c': ...}.
    """
    if topologies is None:
        topologies = measure_all_lambdas(N=N, n_seeds=n_seeds)
    return {name: {"Lambda": float(lam), "beta_c": beta_c(lam, gamma, l_bar_0)}
            for name, lam in topologies.items()}
