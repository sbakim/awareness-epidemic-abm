# Figure & Analysis Register

This file is the single source of truth for which analysis produces which output,
and whether that output appears in the MAIN TEXT, is reported as a TABLE/inline
number, or is kept only for VALIDATION (run but not shown as a figure).

Canonical model = recalibrated sigmoid coefficients (survey-anchored), defined in
`src/model.py`: MASK_COEF, DIST_COEF, VACC_COEF. All figures/tables below are
generated with these defaults. The original heuristic coefficients are retained
in the same file (……_HEURISTIC) only for the robustness comparison in analysis E.

| Analysis script            | Output file                       | Role in paper            | Notes |
|----------------------------|-----------------------------------|--------------------------|-------|
| main_experiment.py         | fig1_infection_curves.png         | MAIN TEXT  Fig 1         | I(t), 6 scenarios x 4 topo |
| main_experiment.py         | fig2_awareness_coevolution.png    | MAIN TEXT  Fig 2         | S5 infection+awareness+mask co-evolution |
| main_experiment.py         | _validation_scenario_heatmap.png  | VALIDATION ONLY          | NOT a manuscript figure; numbers are in Table 5 |
| main_experiment.py         | -> Table 5, Table 6               | MAIN TEXT tables         | total/peak + pairwise S4/S5/S6 |
| H_phase_diagrams.py        | fig3_phase_diagrams.png           | MAIN TEXT  Fig 3         | mean-field rho_inf, beta0 x (alpha, mu) |
| C_finite_size.py           | fig4_finite_size.png              | MAIN TEXT  Fig 4         | N=500..10000, suppression vs N |
| empirical_validation.py    | fig5_empirical_validation.png     | MAIN TEXT  Fig 5         | 3 real SocioPatterns networks |
| B_s3_variants.py           | fig6_s3_variants.png              | MAIN TEXT  Fig 6         | conditional architecture advantage |
| A_mmca_comparison.py       | fig7_mmca_comparison.png          | MAIN TEXT  Fig 7         | multi- vs single-channel (2 panel) |
| E_sigmoid_calibration.py   | _validation_sigmoid_calibration.png | TABLE + VALIDATION     | anchors + robustness; numbers go in a small calibration table |
| D_self_organization.py     | _validation_self_organization.png | TABLE (Table 7)          | Phi_ord/Moran's I/H; numbers in Table 7 |
| F_sensitivity.py           | _validation_sensitivity.png       | TABLE (Table 8)          | Spearman rho; numbers in Table 8 |
| G_multilayer.py            | _validation_multilayer.png        | INLINE sentence          | result is "no effect"; 1-2 sentences only |

## What each analysis answers (reviewer mapping)
- main_experiment : core results (Table 5/6, Fig 1/2)
- H_phase_diagrams: phase structure / second-order transition (Fig 3)
- C_finite_size   : Reviewer #3.5 (N>=10^4)               -> Fig 4
- empirical_validation: Reviewer #1.3 / #2.2 (real networks) -> Fig 5
- B_s3_variants   : Reviewer #3.2 (S3-S4 gap conditionality) -> Fig 6
- A_mmca_comparison: Reviewer #3.1 (novelty vs single-channel) -> Fig 7
- E_sigmoid_calibration: Reviewer #1.1 / #2.1 (parameter justification) -> calibration table
- D_self_organization: Reviewer #3.3 (self-organization claim)  -> Table 7 (+ no spatial org.)
- F_sensitivity   : Reviewer #4.6 (wider sensitivity)       -> Table 8
- G_multilayer    : Reviewer #2.3 (multilayer coupling)      -> inline result

## Figures generated but NOT shown in the paper (kept for validation only)
These scripts still SAVE a .png so the result is reproducible and inspectable,
but the manuscript reports the result as a table or a sentence (see column above):
  - figS1.. naming is legacy; current canonical names are in the table above.
  - self-organization, sigmoid-calibration, sensitivity, multilayer figures:
    look at them to verify, but do not include them as manuscript figures.
