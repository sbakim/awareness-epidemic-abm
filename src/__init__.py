"""
Coupled awareness-epidemic ABM package.

Usage
-----
from src import VectorizedEpidemicModel, phi, beta_c, R_eff
"""

from .model import (VectorizedEpidemicModel, DEFAULT_PARAMS,
                        MASK_COEF, DIST_COEF, VACC_COEF,
                        MASK_COEF_HEURISTIC, DIST_COEF_HEURISTIC, VACC_COEF_HEURISTIC)
from .networks import (generate_network, sigmoid, network_stats,
                       measure_lambda, measure_all_lambdas)
from .analytical import (phi, beta_c, R_eff, phi_bounds, phi_table,
                         phi_elasticity, threshold_table,
                         awareness_steady_state, saturation_prevalence)
from .metrics import (morans_i, morans_i_pvalue, neighbor_mutual_information,
                      neighbor_mi_pvalue, combination_entropy,
                      protection_level_distribution)
from .utils import (mc_run, make_seed, mw_test, cohen_d, spearman_corr,
                    summary_stats, total_infections, peak_infections,
                    bonferroni_alpha)

__all__ = [
    "VectorizedEpidemicModel", "DEFAULT_PARAMS",
    "MASK_COEF", "DIST_COEF", "VACC_COEF",
    "MASK_COEF_HEURISTIC", "DIST_COEF_HEURISTIC", "VACC_COEF_HEURISTIC",
    "generate_network", "sigmoid", "network_stats",
    "measure_lambda", "measure_all_lambdas",
    "phi", "beta_c", "R_eff", "phi_bounds", "phi_table", "phi_elasticity",
    "threshold_table", "awareness_steady_state", "saturation_prevalence",
    "morans_i", "morans_i_pvalue", "neighbor_mutual_information",
    "neighbor_mi_pvalue", "combination_entropy",
    "protection_level_distribution",
    "mc_run", "make_seed", "mw_test", "cohen_d", "spearman_corr",
    "summary_stats", "total_infections", "peak_infections",
    "bonferroni_alpha",
]
