#!/usr/bin/env bash
set -euo pipefail

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
