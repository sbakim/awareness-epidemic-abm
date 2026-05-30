"""
H_phase_diagrams.py
===================
Mean-field phase diagrams of the final epidemic size rho_inf for the coupled
awareness-epidemic system (recovers the structure of Fig. 4 in the original
manuscript, recomputed with the canonical recalibrated coefficients and the
corrected Lambda values).

For each (beta_0, awareness-parameter) pair we solve the HMF self-consistency
    rho* :  rho = 1 - exp(-R_eff(rho) * rho),   R_eff(rho) = (beta_0/gamma) Lambda Phi(rho, l*(rho))
with the awareness steady state  l*(rho) = min(1, delta_local*rho/mu)  (Eq. 14).
Top row sweeps the awareness diffusion proxy via delta_local-scaled feedback
(alpha axis, through its effect on realized awareness); bottom row sweeps mu.

Outputs results/H_phase_diagrams.json and figures/fig3_phase_diagrams.png
"""
# === PAPER ROLE ===
# MAIN TEXT: Fig 3 (mean-field phase diagrams).
# See FIGURE_REGISTER.md for the full map of analyses -> outputs -> paper role.
# Canonical coefficients: src/model.py (MASK_COEF/DIST_COEF/VACC_COEF).
# =================

import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import phi as _phi_raw, measure_all_lambdas
# Precompute Phi on a (theta,l) grid and bilinearly interpolate -- avoids
# repeated Monte-Carlo integration inside the fixed-point loop.
_TH = np.linspace(0,1,101); _LB = np.linspace(0,1,101)
_PHI = None
def _build_phi():
    global _PHI
    _PHI = np.array([[_phi_raw(t,l) for l in _LB] for t in _TH])
def phi(theta, l_bar):
    if _PHI is None: _build_phi()
    ti = min(100, max(0, int(round(theta*100))))
    li = min(100, max(0, int(round(l_bar*100))))
    return _PHI[ti, li]

os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
GAMMA = 0.1

def rho_star(beta_0, Lambda, alpha_eff, mu, delta_local=0.35, n_iter=120):
    """Fixed-point final size. alpha_eff scales how efficiently local feedback
    translates into population awareness (proxy for diffusion rate alpha)."""
    rho = 0.05
    for _ in range(n_iter):
        l_star = min(1.0, alpha_eff * delta_local * rho / max(mu, 1e-6))
        Reff = (beta_0 / GAMMA) * Lambda * phi(rho, l_star)
        new = 1.0 - np.exp(-Reff * rho) if Reff > 1 else 0.0
        rho = 0.5 * rho + 0.5 * new
        if rho < 1e-5:
            return 0.0
    return rho

def main(n_grid=45):
    lam = measure_all_lambdas(N=500, n_seeds=20)
    topos = ["BA", "WS", "ER", "MOD"]
    beta = np.linspace(0.005, 0.30, n_grid)
    alpha = np.linspace(0.0, 0.5, n_grid)   # diffusion-efficiency proxy
    mu = np.linspace(0.005, 0.30, n_grid)   # decay

    data = {"beta": beta.tolist(), "alpha": alpha.tolist(), "mu": mu.tolist(),
            "Lambda": {t: float(lam[t]) for t in topos}, "grids": {}}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    levels = [0.01, 0.10, 0.30, 0.50]
    for j, t in enumerate(topos):
        L = lam[t]
        # top: beta vs alpha (mu fixed at 0.04)
        Za = np.array([[rho_star(b, L, a, 0.04) for b in beta] for a in alpha])
        # bottom: beta vs mu (alpha_eff fixed at 1.0)
        Zm = np.array([[rho_star(b, L, 1.0, m) for b in beta] for m in mu])
        data["grids"][t] = {"beta_alpha": Za.tolist(), "beta_mu": Zm.tolist()}

        for row, (Z, yax, ylab, tag) in enumerate(
            [(Za, alpha, r"$\alpha$ (awareness diffusion)", "abcd"[j]),
             (Zm, mu, r"$\mu$ (awareness decay)", "efgh"[j])]):
            ax = axes[row, j]
            im = ax.contourf(beta, yax, Z, levels=20, cmap="YlOrRd", vmin=0, vmax=1)
            cs = ax.contour(beta, yax, Z, levels=levels, colors="k", linewidths=0.7)
            ax.clabel(cs, fmt="%.2f", fontsize=7)
            ax.axvline(GAMMA/(L*phi(0,0.4)), color="cyan", ls="--", lw=1.2)  # analytical beta_c
            ax.set_xlabel(r"$\beta_0$"); ax.set_ylabel(ylab)
            ax.set_title(f"({tag}) {t}" + ("" if row==0 else ""), loc="left", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label=r"final epidemic size $\rho_\infty$")
    plt.savefig("figures/fig3_phase_diagrams.png", dpi=170, bbox_inches="tight")
    plt.close()
    json.dump(data, open("results/H_phase_diagrams.json", "w"))
    print("beta_c (cyan lines):", {t: round(GAMMA/(lam[t]*phi(0,0.4)),4) for t in topos})
    print("Saved figures/fig3_phase_diagrams.png and results/H_phase_diagrams.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_grid", type=int, default=45)
    main(**vars(ap.parse_args()))
