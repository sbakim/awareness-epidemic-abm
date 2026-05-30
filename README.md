# Phase Transitions and Emergent Behavioral Coordination in Coupled Awareness–Epidemic Dynamics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Complete simulation code, analysis scripts, and supplementary results for:

> **Bakım, S.** (2025). *Phase transitions and emergent behavioral coordination in coupled awareness–epidemic dynamics on complex networks: An agent-based approach with autonomous decision-making.* 

---

---

## Which analysis produces which figure/table?

See **`FIGURE_REGISTER.md`** for the single-source-of-truth map of
analysis script -> output file -> role in the paper (MAIN TEXT figure,
TABLE, or VALIDATION-only). Quick summary:

- Main-text figures: `figures/fig1_…` through `figures/fig7_…`
  (infection curves, awareness co-evolution, phase diagrams, finite-size,
  empirical validation, S3 variants, MMCA comparison).
- Files prefixed `figures/_validation_…` are produced for reproducibility/
  inspection but are **not** manuscript figures; their results are reported
  as tables (Tables 7, 8, calibration table) or as inline sentences
  (multilayer). The scenario heatmap is likewise validation-only (numbers
  are in Table 5).
- The canonical (survey-recalibrated) sigmoid coefficients live in
  `src/model.py` (`MASK_COEF`, `DIST_COEF`, `VACC_COEF`); the original
  heuristic set is kept alongside (`…_HEURISTIC`) only for the robustness
  comparison in `analyses/E_sigmoid_calibration.py`.

---

## Overview

Each network node is an autonomous agent that simultaneously decides on
mask-wearing, social distancing, and vaccination through sigmoid-based functions
driven by individual compliance, dynamic awareness, and local epidemic state.
The framework couples three dynamical processes:

- **SIR disease transmission** with behavior-dependent effective rates
- **Continuous awareness diffusion** with local epidemic feedback and decay
- **Prevalence- and awareness-dependent adaptive network rewiring** (Scenario S6)

### Key findings

| Finding | Details |
|---------|---------|
| Epidemic threshold | β_c = γ / (Λ · Φ); topology ordering BA < MOD < ER < WS is invariant under behavioral parameters |
| Heterogeneity ratio | Measured Λ ≈ 15.6 (BA), 8.1 (WS), 9.0 (ER), 9.6 (MOD) at N = 500 — BA threshold ≈ 2× lower than WS |
| Behavioral suppression | Full co-evolution (S6) achieves ~88–94% reduction across topologies |
| Architecture vs aggregate | Single- and multi-channel models give similar aggregate suppression, but only the multi-channel ABM produces a non-degenerate distribution of individual protection states (combination entropy > 0) |
| Finite-size scaling | Suppression increases monotonically toward N = 10,000 |
| Robustness | β₀ dominance confirmed across ±30%, ±50%, ±70% LHS perturbation ranges |

---

## Repository structure

```
.
├── src/
│   ├── model.py            # VectorizedEpidemicModel (core ABM)
│   ├── networks.py         # network generation + Lambda measurement
│   ├── analytical.py       # Phi, beta_c, R_eff (Lambda measured, not hard-coded)
│   ├── metrics.py          # Moran's I, neighbor MI, combination entropy
│   └── utils.py            # reproducible seeding + statistical helpers
│
├── analyses/
│   ├── main_experiment.py        # Tables 5 & 6, infection curves, heatmap
│   ├── A_mmca_comparison.py      # multi-channel ABM vs single-channel baseline (R3.1)
│   ├── B_s3_variants.py          # S3 aggressiveness sweep (R3.2)
│   ├── C_finite_size.py          # finite-size scaling N = 500 → 10,000 (R3.5)
│   ├── D_self_organization.py    # Moran's I + neighbor MI, Table 7 (R3.3)
│   ├── E_sigmoid_calibration.py  # sigmoid vs survey ranges (R1.1)
│   ├── F_sensitivity.py          # extended ±30/50/70% LHS, Table 8 (R2/R3.6)
│   └── empirical_validation.py   # real or calibrated-synthetic contact nets
│
├── data/                   # SocioPatterns data (not redistributed; see data/README.md)
├── prepare_data.py         # downloads + builds the empirical edge lists
├── run_all.py              # runs the full pipeline (use --quick to smoke-test)
├── requirements.txt
└── README.md
```

Running any analysis writes JSON to `results/` and PNGs to `figures/`
(both created automatically).

---

## Installation

```bash
git clone https://github.com/sbakim/awareness-epidemic-abm.git
cd awareness-epidemic-abm
pip install -r requirements.txt
```

Requirements: Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.8, NetworkX ≥ 2.8, Matplotlib ≥ 3.5.

---

## Quick start

```python
from src.model import VectorizedEpidemicModel

model = VectorizedEpidemicModel(topology="BA", N=500, scenario="S6", T=200, seed=42)
model.run()
s = model.get_summary()
print(f"Peak {s['peak_inf']:.1%}, total {s['total_inf']:.1%}")
```

Run the whole pipeline:

```bash
python run_all.py --quick   # ~minutes: verify everything runs
python run_all.py           # full run (~30–60 min on Colab)
```

Or a single analysis, e.g.:

```bash
python analyses/main_experiment.py --n_runs 30
python analyses/A_mmca_comparison.py
```

---

## Empirical data

The empirical validation (Section 4.5) uses three real SocioPatterns contact
networks. These are **not** redistributed here; the analysis MIT licence does
not cover them. To fetch and build them:

```bash
python prepare_data.py
```

This downloads the raw records (primary school, high school, hospital ward) and
writes weighted edge lists into `data/`. See [data/README.md](data/README.md)
for sources, citations, and the manual-download fallback.

---

## Reproducibility

Seeds are derived deterministically from `(seed_base, run, topology, scenario)`
in `src/utils.py` (no use of Python's process-randomized `hash()`), so every
script reproduces identical numbers on re-run. For full determinism set
`PYTHONHASHSEED=0` in the environment.

---

## Scenarios

| Scenario | Mechanisms |
|----------|-----------|
| S1 | No intervention |
| S2 | Homogeneous (uniform edge reduction + vaccination) |
| S3 | Centrality-targeted (configurable aggressiveness) |
| S4 | Autonomous sigmoid decisions (static awareness) |
| S5 | S4 + dynamic awareness diffusion |
| S6 | S5 + awareness-dependent network rewiring |

---

## Contact

Sümeyye Bakım — sumeyye.bakim@karatay.edu.tr
KTO Karatay University, Konya, Türkiye

## License

The code in this repository is released under the MIT licence — see [LICENSE](LICENSE).

The empirical contact-network data used in Section 4.5 are **not** part of this
repository and are **not** covered by the MIT licence. They belong to the
SocioPatterns initiative and are subject to their own terms (CC-BY-NC-SA / CC0);
see [data/README.md](data/README.md) and please cite the original dataset papers.
