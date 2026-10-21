#!/usr/bin/env python3
"""Robustness and validation analyses for the Schelling entropy study.

The script performs two independent checks:

1. A 3 x 3 mask-threshold sensitivity analysis on the original 120 seeded
   histories (c_tail in {6, 7, 8}; c_0 in {3, 4, 5}).
2. Larger-lattice and one-factor-at-a-time model-parameter sweeps, evaluated at
   eleven common checkpoints from step 0 through step 100.

The original N=64, 30-run ensemble is read from ``--main-results`` and is not
rerun for the model sweeps.  The mask analysis deliberately recreates the same
seeded histories so all diagnostic variants are paired within run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from entropy_metrics import (
    build_masks,
    classify_microstates,
    compute_step_metrics,
    fit_reference_weights,
    histogram_C_by_state,
    shape_entropy_components,
    shape_weighted_entropy,
)
from run_paper_batch import aggregate_runs, make_params
from simulation_model import run_simulation


MASK_TAILS = (6, 7, 8)
MASK_CENTERS = (3, 4, 5)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan
    xx, yy = x[mask], y[mask]
    if float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _direction(x: np.ndarray, tolerance: float = 1e-12) -> int:
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return 0
    delta = float(finite[-1] - finite[0])
    if abs(delta) <= tolerance:
        return 0
    return 1 if delta > 0 else -1


def _mask_worker(payload):
    args_dict, scenario, run_idx = payload
    args = argparse.Namespace(**args_dict)
    params = make_params(args, scenario=int(scenario), run_idx=int(run_idx))
    C_hist, M_hist, _stats = run_simulation(params)

    grid = np.arange(params.C_range[0], params.C_range[1] + 1)
    masks = {
        (tail, center): build_masks(grid, tail, center, kappa_center=1.0)
        for tail in MASK_TAILS
        for center in MASK_CENTERS
    }
    trajectories = {key: [] for key in masks}

    for t in range(C_hist.shape[0]):
        st = classify_microstates(C_hist[t], M_hist[t], params)
        hists = histogram_C_by_state(
            C_hist[t], st, params.C_range[0], params.C_range[1], alpha=0.5
        )
        total = max(1, sum(int(hists[state]["n"]) for state in "BAT"))
        state_cache = {}
        for state in "BAT":
            n_state = int(hists[state]["n"])
            if n_state == 0:
                continue
            p = hists[state]["p"]
            weights, _meta = fit_reference_weights(p, grid)
            state_cache[state] = (p, weights, n_state / total)

        for key, (m_tail, m_bi) in masks.items():
            value = 0.0
            for p, weights, fraction in state_cache.values():
                comps = shape_entropy_components(
                    p,
                    grid,
                    q_tsallis=1.5,
                    beta_tail=1.0,
                    beta_bimodal=1.0,
                    m_tail=m_tail,
                    m_bi=m_bi,
                )
                value += fraction * shape_weighted_entropy(comps, weights)
            trajectories[key].append(value)

    base = np.asarray(trajectories[(7, 4)], dtype=float)
    rows: List[Dict] = []
    for tail in MASK_TAILS:
        for center in MASK_CENTERS:
            variant = np.asarray(trajectories[(tail, center)], dtype=float)
            base_final = float(base[-1])
            rel_final = (
                100.0 * abs(float(variant[-1]) - base_final) / max(abs(base_final), 1e-12)
            )
            rows.append({
                "scenario": int(scenario),
                "run": int(run_idx),
                "tail_threshold": int(tail),
                "center_threshold": int(center),
                "trajectory_r": _pearson(base, variant),
                "final_relative_change_pct": rel_final,
                "direction_preserved": int(_direction(base) == _direction(variant)),
                "positive_all_steps": int(bool(np.all(variant > 0.0))),
            })
    return rows


def sweep_configurations() -> List[Dict]:
    return [
        {"config": "size_N96", "family": "lattice size", "parameter": "N", "value": 96, "N": 96},
        {"config": "size_N128", "family": "lattice size", "parameter": "N", "value": 128, "N": 128},
        {"config": "happy_h3", "family": "satisfaction", "parameter": "h", "value": 3, "happy_threshold": 3},
        {"config": "happy_h5", "family": "satisfaction", "parameter": "h", "value": 5, "happy_threshold": 5},
        {"config": "vacancy_v020", "family": "vacancy", "parameter": "v", "value": 0.20, "vacancy": 0.20},
        {"config": "vacancy_v040", "family": "vacancy", "parameter": "v", "value": 0.40, "vacancy": 0.40},
        {"config": "calls_n1", "family": "long-range contacts", "parameter": "n_call", "value": 1, "calls_per_agent": 1},
        {"config": "calls_n3", "family": "long-range contacts", "parameter": "n_call", "value": 3, "calls_per_agent": 3},
        {"config": "cost_scale075", "family": "economic costs", "parameter": "cost scale", "value": 0.75, "X_move": 0.75, "X_call": 0.1875},
        {"config": "cost_scale125", "family": "economic costs", "parameter": "cost scale", "value": 1.25, "X_move": 1.25, "X_call": 0.3125},
    ]


def _configured_namespace(args_dict: Dict, config: Dict) -> argparse.Namespace:
    d = dict(args_dict)
    for name in ["N", "happy_threshold", "vacancy", "calls_per_agent", "X_move", "X_call"]:
        if name in config:
            d[name] = config[name]
    return argparse.Namespace(**d)


def _sweep_worker(payload):
    args_dict, config, scenario, run_idx, checkpoints = payload
    args = _configured_namespace(args_dict, config)
    params = make_params(args, scenario=int(scenario), run_idx=int(run_idx))
    C_hist, M_hist, stats_df = run_simulation(params)
    stats = stats_df.set_index("step")
    rows: List[Dict] = []
    for step in checkpoints:
        row = compute_step_metrics(C_hist[step], M_hist[step], params=params)
        row.update({
            "config": config["config"],
            "family": config["family"],
            "parameter": config["parameter"],
            "value": config["value"],
            "scenario": int(scenario),
            "run": int(run_idx),
            "step": int(step),
            "N": int(params.N),
            "happy_threshold": int(params.happy_threshold),
            "vacancy": float(params.vacancy),
            "calls_per_agent": int(params.calls_per_agent),
            "X_move": float(params.X_move),
            "X_call": float(params.X_call),
            "moves_made": float(stats.loc[step, "moves_made"]),
            "mean_money": float(stats.loc[step, "mean_money"]),
        })
        rows.append(row)
    return rows


def _aggregate_sweep(df: pd.DataFrame) -> pd.DataFrame:
    group = ["config", "family", "parameter", "value", "scenario", "step"]
    excluded = set(group) | {"run", "N", "happy_threshold", "vacancy", "calls_per_agent", "X_move", "X_call"}
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    agg = df.groupby(group)[numeric].agg(["mean", "std", "count"]).reset_index()
    agg.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{'sd' if col[1] == 'std' else ('n' if col[1] == 'count' else col[1])}"
        for col in agg.columns
    ]
    return agg


def _baseline_rows(main_results: Path, checkpoints: Iterable[int]) -> pd.DataFrame:
    df = pd.read_csv(main_results / "all_runs_metrics.csv")
    df = df[df["step"].isin(list(checkpoints))].copy()
    df["config"] = "baseline"
    df["family"] = "baseline"
    df["parameter"] = "reported"
    df["value"] = 0.0
    df["N"] = 64
    df["happy_threshold"] = 4
    df["vacancy"] = 0.30
    df["calls_per_agent"] = 2
    df["X_move"] = 1.0
    df["X_call"] = 0.25
    return df


def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "S_full_lattice",
        "bimodal_score_lattice",
        "wB_lattice",
        "bimodality_coefficient",
        "esteban_ray_polarization",
        "morans_I",
        "sign_assortativity",
        "qB",
        "qT",
    ]
    rows: List[Dict] = []
    for scenario in sorted(summary["scenario"].unique()):
        baseline = summary[(summary["config"] == "baseline") & (summary["scenario"] == scenario)].sort_values("step")
        for config in sorted(c for c in summary["config"].unique() if c != "baseline"):
            variant = summary[(summary["config"] == config) & (summary["scenario"] == scenario)].sort_values("step")
            if baseline.empty or variant.empty:
                continue
            meta = variant.iloc[0]
            for metric in metrics:
                col = f"{metric}_mean"
                if col not in baseline or col not in variant:
                    continue
                merged = baseline[["step", col]].merge(variant[["step", col]], on="step", suffixes=("_base", "_variant"))
                x = merged[f"{col}_base"].to_numpy(dtype=float)
                y = merged[f"{col}_variant"].to_numpy(dtype=float)
                base_final = float(x[-1])
                rows.append({
                    "config": config,
                    "family": meta["family"],
                    "parameter": meta["parameter"],
                    "value": meta["value"],
                    "scenario": int(scenario),
                    "metric": metric,
                    "trajectory_r": _pearson(x, y),
                    "direction_preserved": int(_direction(x) == _direction(y)),
                    "baseline_final": base_final,
                    "variant_final": float(y[-1]),
                    "final_absolute_change": abs(float(y[-1]) - base_final),
                })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = list(range(0, int(args.steps) + 1, int(args.checkpoint_every)))
    if checkpoints[-1] != int(args.steps):
        checkpoints.append(int(args.steps))

    common = vars(args).copy()
    common.update({
        "N": 64,
        "vacancy": 0.30,
        "R_move": 4,
        "happy_threshold": 4,
        "C_min": -10,
        "C_max": 10,
        "M_min": 5.0,
        "M_max": 15.0,
        "X_move": 1.0,
        "X_call": 0.25,
        "calls_per_agent": 2,
        "base_seed": int(args.base_seed),
        "representative_run": -1,
        "alpha_smooth": 0.5,
        "q_tsallis": 1.5,
        "beta_tail": 1.0,
        "beta_bimodal": 1.0,
        "tail_threshold": 7,
        "center_threshold": 4,
        "kappa_center": 1.0,
        "export_excel_representatives": False,
    })

    mask_tasks = [
        (common, scenario, run_idx)
        for scenario in (1, 2, 3, 4)
        for run_idx in range(int(args.mask_runs))
    ]
    mask_rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as pool:
        for done, rows in enumerate(pool.map(_mask_worker, mask_tasks), start=1):
            mask_rows.extend(rows)
            print(f"Mask histories completed: {done}/{len(mask_tasks)}", flush=True)
    mask_df = pd.DataFrame(mask_rows)
    mask_df.to_csv(out / "mask_sensitivity_per_run.csv", index=False)
    mask_summary = mask_df.groupby(["scenario", "tail_threshold", "center_threshold"], as_index=False).agg(
        trajectory_r_mean=("trajectory_r", "mean"),
        trajectory_r_min=("trajectory_r", "min"),
        final_relative_change_pct_mean=("final_relative_change_pct", "mean"),
        final_relative_change_pct_max=("final_relative_change_pct", "max"),
        direction_preserved_fraction=("direction_preserved", "mean"),
        positive_fraction=("positive_all_steps", "mean"),
    )
    mask_summary.to_csv(out / "mask_sensitivity_summary.csv", index=False)

    configs = sweep_configurations()
    sweep_tasks = [
        (common, config, scenario, run_idx, checkpoints)
        for config in configs
        for scenario in (1, 2, 3, 4)
        for run_idx in range(int(args.sweep_runs))
    ]
    sweep_rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as pool:
        for done, rows in enumerate(pool.map(_sweep_worker, sweep_tasks), start=1):
            sweep_rows.extend(rows)
            print(f"Sweep simulations completed: {done}/{len(sweep_tasks)}", flush=True)
    sweep_df = pd.DataFrame(sweep_rows)
    baseline = _baseline_rows(Path(args.main_results), checkpoints)
    common_columns = sorted(set(sweep_df.columns) & set(baseline.columns))
    combined = pd.concat([baseline[common_columns], sweep_df[common_columns]], ignore_index=True)
    combined.to_csv(out / "sweep_checkpoints.csv", index=False)
    summary = _aggregate_sweep(combined)
    summary.to_csv(out / "sweep_summary.csv", index=False)
    comparisons = _comparison_table(summary)
    comparisons.to_csv(out / "sweep_comparisons.csv", index=False)
    print(f"Validation outputs written to: {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run robustness and validation checks.")
    p.add_argument("--main-results", default="results_r3")
    p.add_argument("--out", default="results_robustness_validation")
    p.add_argument("--mask-runs", type=int, default=30)
    p.add_argument("--sweep-runs", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=20260519)
    p.add_argument("--n-jobs", type=int, default=8)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
