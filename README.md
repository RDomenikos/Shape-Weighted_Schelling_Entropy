# Shape-Weighted Entropy for Generalized Schelling Models

This repository contains the simulation, diagnostic, robustness, and plotting pipeline for the paper *Beyond Shannon Entropy in Schelling Segregation: Shape-Weighted Diagnostics of Bimodality, Polarization, and Economic Freezing*.

The code compares four generalized Schelling scenarios and evaluates their affiliation distributions using Shannon entropy, a shape-weighted diagnostic, independent measures of bimodality and polarization, and established measures of spatial segregation.

## Repository files

- `simulation_model.py` — implements Scenarios 1–4 and the lattice dynamics.
- `entropy_metrics.py` — calculates the B/A/T microstates, Shannon and shape-weighted entropy, distribution weights, bimodality and polarization indices, and spatial segregation measures.
- `run_paper_batch.py` — runs repeated simulations and saves the numerical results.
- `run_robustness_validation.py` — reproduces the mask-threshold, enlarged-lattice, and one-factor model-parameter robustness checks.
- `plot_paper_figures.py` — generates the paper figures from the saved results.
- `run_full_paper.sh` — runs the complete numerical and plotting workflow with the reported settings.
- `requirements.txt` — Python dependencies.

## Installation

Python 3.9 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick test

Run a small test before starting the full simulations:

```bash
python run_paper_batch.py \
  --runs 2 \
  --steps 5 \
  --N 16 \
  --scenarios 1 2 3 4 \
  --out test_results \
  --n_jobs 2

python plot_paper_figures.py --results test_results
```

## Reproduce the paper simulations

```bash
python run_paper_batch.py \
  --runs 30 \
  --steps 100 \
  --N 64 \
  --scenarios 1 2 3 4 \
  --out results_paper \
  --n_jobs 8

python run_robustness_validation.py \
  --main-results results_paper \
  --out results_robustness_validation \
  --mask-runs 30 \
  --sweep-runs 10 \
  --steps 100 \
  --checkpoint-every 10 \
  --n-jobs 8

python plot_paper_figures.py \
  --results results_paper \
  --validation-results results_robustness_validation
```

The batch runner uses the parameter defaults reported in the paper, including the vacancy fraction, movement radius, happiness threshold, affiliation range, monetary range, movement and interaction costs, entropy parameters, and random-seed construction.

The main paper batch uses 30 independent runs for each scenario. The robustness pipeline reuses those paired histories for the threshold checks, evaluates lattice sizes 64, 96, and 128, and varies the happiness threshold, vacancy fraction, number of long-range calls, and common cost scale one factor at a time. Adjust `--n_jobs` or `--n-jobs` to match the available CPU cores.

To inspect or change any option:

```bash
python run_paper_batch.py --help
python run_robustness_validation.py --help
python plot_paper_figures.py --help
```

To additionally export the representative simulations to Excel, add:

```bash
--export_excel_representatives
```

## Main outputs

`run_paper_batch.py` creates:

- `all_runs_metrics.csv` — every run and simulation step;
- `summary_timeseries.csv` — ensemble means, standard deviations, sample sizes, and 95% confidence intervals;
- `correlation_summary.csv` — correlations among entropy, distributional, and spatial diagnostics;
- `per_run/` — individual-run trajectories;
- `representative_npz/` — representative lattice histories;
- `run_parameters.json` — complete parameters and seed information.

`run_robustness_validation.py` creates:

- `mask_sensitivity_per_run.csv` and `mask_sensitivity_summary.csv` — paired threshold robustness results;
- `sweep_checkpoints.csv` and `sweep_summary.csv` — enlarged-lattice and model-parameter runs;
- `sweep_comparisons.csv` — direction-preservation and trajectory-correlation comparisons against the baseline.

`plot_paper_figures.py` saves the figures in:

```text
results_paper/paper_figures/
```

Generated results are not required in the source repository because they can be recreated with the commands above.

## Reproducibility notes

- The main batch uses a fixed base seed and independent seed streams for vacancy placement, initial affiliation, initial money, and stochastic dynamics.
- `run_parameters.json` records the exact parameter and seed configuration for each run.
- Vacancies are stored as `NaN`; occupied agents with neutral affiliation `C = 0` remain valid observations.
- The distributional measures do not use spatial coordinates. Moran's I, sign assortativity, and the same-sign edge fraction are calculated separately from occupied Moore-neighborhood edges.
- The mask-threshold analysis evaluates all nine combinations of tail thresholds `{6, 7, 8}` and center thresholds `{3, 4, 5}` on paired seeded histories.
- The enlarged-lattice and one-factor sweeps are evaluated at common checkpoints from step 0 through step 100.

## Citation and license

Please cite the associated paper when using this code or its outputs. The project license is provided in `LICENSE`.
