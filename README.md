# Schelling Shape-Weighted Entropy Pipeline

Code used to simulate the four generalized Schelling scenarios and calculate the entropy and structural diagnostics reported in the paper.

## Repository files

- `simulation_model.py` — implements Scenarios 1–4 and the lattice dynamics.
- `entropy_metrics.py` — calculates the B/A/T microstates, Shannon and shape-weighted entropy, distribution weights, and structural metrics.
- `run_paper_batch.py` — runs repeated simulations and saves the numerical results.
- `plot_paper_figures.py` — generates the paper figures from the saved results.
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
  --out test_results

python plot_paper_figures.py --results test_results
```

## Reproduce the paper simulations

```bash
python run_paper_batch.py \
  --runs 30 \
  --steps 100 \
  --N 64 \
  --scenarios 1 2 3 4 \
  --out results_paper

python plot_paper_figures.py --results results_paper
```

The batch runner uses the parameter defaults reported in the paper, including the vacancy fraction, movement radius, happiness threshold, affiliation range, monetary range, movement and interaction costs, entropy parameters, and random-seed construction.

To inspect or change any option:

```bash
python run_paper_batch.py --help
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
- `correlation_summary.csv` — correlations between entropy diagnostics and structural metrics;
- `per_run/` — individual-run trajectories;
- `representative_npz/` — representative lattice histories;
- `run_parameters.json` — complete parameters and seed information.

`plot_paper_figures.py` saves the figures in:

```text
results_paper/paper_figures/
```

Generated results are not required in the source repository because they can be recreated with the commands above.
