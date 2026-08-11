# Local sensitivity analysis

## Scope

The shape-weighted lattice entropy was recomputed on the same 120 seeded
simulation histories used in the paper (30 runs for each of four scenarios).
The analysis used the full factorial grid

- `q = {1.4, 1.5, 1.6}`;
- `beta_tail = {0.9, 1.0, 1.1}`; and
- `beta_bimodal = {0.9, 1.0, 1.1}`.

This gives 27 parameter combinations. Because these parameters enter only in
the entropy post-processing, each combination was evaluated on identical agent
histories. The comparison therefore isolates parameter sensitivity from
stochastic simulation variability.

## Results

| Scenario | Minimum Pearson correlation with reported mean trajectory | Maximum absolute final-step change | Minimum lattice entropy over the grid |
|---:|---:|---:|---:|
| 1 | 0.9989 | 5.30% | 0.2987 |
| 2 | 0.9901 | 6.68% | 0.6237 |
| 3 | 0.9921 | 6.91% | 0.6406 |
| 4 | 0.9891 | 6.61% | 0.9779 |

Across all scenarios and parameter combinations:

- the minimum Pearson correlation with the reported ensemble-mean trajectory
  was 0.9891;
- the direction of the start-to-final change was preserved in every case;
- the largest absolute change in the final-step mean was 6.91%; and
- every lattice-level entropy value remained positive.

The fitted normal, heavy-tail, and bimodality weights are unchanged because
they are determined from the empirical histograms before `q`, `beta_tail`, and
`beta_bimodal` enter the scalar diagnostic. The independent polarization,
central-depletion, tail-mass, and segregation indices are likewise unchanged.
The occasional interchange of Scenarios 2 and 3 in their exact final-step rank
reflects their near-tied reported values and does not alter their common
polarized classification or any conclusion of the paper.

The central setting reproduces the original 12,120 run-step values with a
maximum absolute difference of `1.29e-14`.

## Reproduction

From the repository root, run:

```bash
python run_entropy_sensitivity.py \
  --runs 30 \
  --workers 8 \
  --scenarios 1 2 3 4 \
  --out results_sensitivity
```

The script writes the ensemble trajectories, run-level values, qualitative
checks, structural correlations, final-step ordering checks, exact parameters,
and the sensitivity figure to `results_sensitivity/`.

The outcomes can be repricated by hte user, and are readily available also in the figure fig_parameter_sensitivity.png in the git. 

