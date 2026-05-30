"""
run_all.py
==========
Driver that runs the full analysis pipeline and writes all results to
results/ and all figures to figures/.

On Google Colab:
    !git clone <your-repo-url> && cd <repo> && pip install -r requirements.txt
    !python run_all.py            # full run (can take ~30-60 min)
    !python run_all.py --quick    # fast smoke run with reduced parameters

The --quick flag drastically reduces sample sizes so you can verify everything
runs end-to-end in a few minutes before launching the full run.
"""

import argparse
import subprocess
import sys
import time

FULL = {
    "analyses/main_experiment.py":      ["--n_runs", "30"],
    "analyses/A_mmca_comparison.py":    ["--n_runs", "20"],
    "analyses/B_s3_variants.py":        ["--n_runs", "30"],
    "analyses/C_finite_size.py":        [],
    "analyses/D_self_organization.py":  ["--n_runs", "30"],
    "analyses/E_sigmoid_calibration.py": ["--n_runs", "20"],
    "analyses/F_sensitivity.py":        ["--n_samples", "50", "--n_runs", "5"],
    "analyses/empirical_validation.py": [],
    "analyses/G_multilayer.py":         [],
    "analyses/H_phase_diagrams.py":     [],
}

QUICK = {
    "analyses/main_experiment.py":      ["--n_runs", "4", "--T", "120"],
    "analyses/A_mmca_comparison.py":    ["--n_runs", "4"],
    "analyses/B_s3_variants.py":        ["--n_runs", "4", "--T", "120"],
    "analyses/C_finite_size.py":        ["--sizes", "500", "1000",
                                         "--n_runs", "3", "--T", "100"],
    "analyses/D_self_organization.py":  ["--n_runs", "6"],
    "analyses/E_sigmoid_calibration.py": ["--n_runs", "4"],
    "analyses/F_sensitivity.py":        ["--n_samples", "12", "--n_runs", "2",
                                         "--T", "100"],
    "analyses/empirical_validation.py": [],
    "analyses/G_multilayer.py":         [],
    "analyses/H_phase_diagrams.py":     ["--n_grid", "25"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run with reduced parameters")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only the named scripts (basename match)")
    args = ap.parse_args()

    plan = QUICK if args.quick else FULL
    print(f"=== run_all.py ({'QUICK' if args.quick else 'FULL'}) ===\n")

    t0 = time.time()
    failures = []
    for script, scr_args in plan.items():
        if args.only and not any(o in script for o in args.only):
            continue
        print(f"\n{'#'*70}\n# {script}\n{'#'*70}")
        ts = time.time()
        ret = subprocess.run([sys.executable, script] + scr_args)
        dt = time.time() - ts
        if ret.returncode != 0:
            failures.append(script)
            print(f"!! {script} FAILED (exit {ret.returncode})")
        else:
            print(f"-- {script} done in {dt:.0f}s")

    print(f"\n=== finished in {time.time()-t0:.0f}s ===")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("All analyses completed. See results/ and figures/.")


if __name__ == "__main__":
    main()
