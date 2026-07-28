#!/usr/bin/env python3
"""
Run the full paper pipeline up to CSV outputs.

This script performs genuine multi-seed simulations for Scenarios 1-4, computes
shape-weighted entropy diagnostics, computes independent structural metrics, and
writes per-run and averaged outputs.

Typical use
-----------
python run_paper_batch.py --runs 30 --steps 100 --N 64 --out results_paper

Quick smoke test
----------------
python run_paper_batch.py --runs 2 --steps 5 --N 16 --out test_results --scenarios 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from simulation_model import SimulationParams, run_simulation, save_npz_run, export_history_to_excel
from entropy_metrics import compute_metrics_for_history


NUMERIC_EXCLUDE = {"scenario", "run", "seed_positions", "seed_affil", "seed_money", "seed_dynamics"}


def make_params(args: argparse.Namespace, scenario: int, run_idx: int) -> SimulationParams:
    """Create reproducible but genuinely different seeds for each scenario/run."""
    base = int(args.base_seed)
    offset = base + 100_000 * int(scenario) + 1_000 * int(run_idx)
    return SimulationParams(
        scenario=int(scenario),
        N=int(args.N),
        steps=int(args.steps),
        vacancy=float(args.vacancy),
        R_move=int(args.R_move),
        happy_threshold=int(args.happy_threshold),
        C_range=(int(args.C_min), int(args.C_max)),
        M_range=(float(args.M_min), float(args.M_max)),
        X_move=float(args.X_move),
        X_call=float(args.X_call),
        calls_per_agent=int(args.calls_per_agent),
        seed_positions=offset + 11,
        seed_affil=offset + 23,
        seed_money=offset + 37,
        seed_dynamics=offset + 51,
    )


def aggregate_runs(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all numeric columns by scenario and step."""
    group_cols = ["scenario", "step"]
    numeric_cols = [
        c for c in all_runs.select_dtypes(include=[np.number]).columns
        if c not in NUMERIC_EXCLUDE and c not in group_cols
    ]

    grouped = all_runs.groupby(group_cols)[numeric_cols]
    agg = grouped.agg(["mean", "std", "count"]).reset_index()

    # Flatten columns: (S_full_lattice, mean) -> S_full_lattice_mean
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                suffix = "sd" if col[1] == "std" else ("n" if col[1] == "count" else col[1])
                flat_cols.append(f"{col[0]}_{suffix}")
        else:
            flat_cols.append(col)
    agg.columns = flat_cols

    ci_cols = {}
    for c in numeric_cols:
        mean_col = f"{c}_mean"
        sd_col = f"{c}_sd"
        n_col = f"{c}_n"
        if mean_col in agg and sd_col in agg and n_col in agg:
            ci = 1.96 * agg[sd_col] / np.sqrt(agg[n_col].clip(lower=1))
            ci_cols[f"{c}_ci95_low"] = agg[mean_col] - ci
            ci_cols[f"{c}_ci95_high"] = agg[mean_col] + ci

    if ci_cols:
        agg = pd.concat([agg, pd.DataFrame(ci_cols)], axis=1)
    return agg


def correlation_table(all_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate candidate diagnostics with independent structure metrics.

    Correlations are computed within each scenario and run over time, then averaged
    across runs. This avoids a single long run dominating the estimate.
    """
    diagnostics = [
        "S_shannon_lattice",
        "S_tail_ablation_lattice",
        "S_bimodal_ablation_lattice",
        "S_full_lattice",
        "wN_lattice",
        "wT_lattice",
        "wB_lattice",
    ]
    targets = ["polarization", "central_depletion", "tail_mass", "segregation"]

    rows: List[Dict[str, float | str | int]] = []
    for (scenario, run_idx), dfr in all_runs.groupby(["scenario", "run"]):
        for diag in diagnostics:
            for target in targets:
                if diag not in dfr.columns or target not in dfr.columns:
                    continue
                x = dfr[diag]
                y = dfr[target]
                mask = x.notna() & y.notna()
                if int(mask.sum()) < 3:
                    r = np.nan
                elif float(np.nanstd(x[mask])) == 0.0 or float(np.nanstd(y[mask])) == 0.0:
                    r = np.nan
                else:
                    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                rows.append({
                    "scenario": scenario,
                    "run": run_idx,
                    "diagnostic": diag,
                    "target": target,
                    "r": r,
                })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    summary = raw.groupby(["scenario", "diagnostic", "target"], as_index=False).agg(
        r_mean=("r", "mean"),
        r_sd=("r", "std"),
        n_runs=("r", "count"),
    )
    return summary


def run_pipeline(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_root / "per_run"
    rep_dir = out_root / "representative_npz"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []
    params_records: List[Dict] = []

    for scenario in args.scenarios:
        scenario = int(scenario)
        print(f"\n=== Scenario {scenario} ===")
        for run_idx in range(int(args.runs)):
            params = make_params(args, scenario=scenario, run_idx=run_idx)
            print(f"Scenario {scenario}, run {run_idx + 1}/{args.runs}")

            C_hist, M_hist, stats_df = run_simulation(params)
            metrics_df = compute_metrics_for_history(
                C_hist,
                M_hist,
                params=params,
                stats_df=stats_df,
                alpha_smooth=float(args.alpha_smooth),
                q_tsallis=float(args.q_tsallis),
                beta_tail=float(args.beta_tail),
                beta_bimodal=float(args.beta_bimodal),
                tail_threshold=args.tail_threshold,
                center_threshold=args.center_threshold,
                kappa_center=float(args.kappa_center),
            )
            metrics_df.insert(0, "scenario", scenario)
            metrics_df.insert(1, "run", run_idx)
            metrics_df["seed_positions"] = params.seed_positions
            metrics_df["seed_affil"] = params.seed_affil
            metrics_df["seed_money"] = params.seed_money
            metrics_df["seed_dynamics"] = params.seed_dynamics

            run_csv = per_run_dir / f"scenario_{scenario}_run_{run_idx:03d}.csv"
            metrics_df.to_csv(run_csv, index=False)
            all_rows.append(metrics_df)
            params_records.append(params.to_jsonable() | {"run": run_idx})

            if run_idx == int(args.representative_run):
                save_npz_run(
                    rep_dir / f"scenario_{scenario}_representative.npz",
                    C_hist,
                    M_hist,
                    params=params,
                    stats_df=stats_df,
                )
                if args.export_excel_representatives:
                    export_history_to_excel(
                        C_hist,
                        M_hist,
                        params=params,
                        out_dir=out_root / "representative_excel" / f"scenario_{scenario}",
                    )

    all_runs = pd.concat(all_rows, ignore_index=True)
    all_runs.to_csv(out_root / "all_runs_metrics.csv", index=False)

    summary = aggregate_runs(all_runs)
    summary.to_csv(out_root / "summary_timeseries.csv", index=False)

    corr = correlation_table(all_runs)
    corr.to_csv(out_root / "correlation_summary.csv", index=False)

    with open(out_root / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump({
            "command_arguments": vars(args),
            "run_params": params_records,
            "notes": [
                "Empty cells are np.nan; C=0 is a valid neutral affiliation bin.",
                "summary_timeseries.csv contains means, SDs, run counts, and 95% CI bounds by scenario and step.",
                "correlation_summary.csv correlates entropy diagnostics with independent structural metrics within each run, then averages by scenario.",
            ],
        }, f, indent=2)

    print(f"\nDone. Outputs written to: {out_root}")
    print("Next step: python plot_paper_figures.py --results", out_root)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run paper-ready Schelling entropy simulations.")
    p.add_argument("--out", default="results_paper", help="Output directory.")
    p.add_argument("--scenarios", nargs="+", type=int, default=[1, 2, 3, 4], help="Scenarios to run.")
    p.add_argument("--runs", type=int, default=30, help="Number of stochastic runs per scenario.")
    p.add_argument("--representative_run", type=int, default=0, help="Which run to save as representative history.")

    # Simulation parameters
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--vacancy", type=float, default=0.30)
    p.add_argument("--R_move", type=int, default=4)
    p.add_argument("--happy_threshold", type=int, default=4)
    p.add_argument("--C_min", type=int, default=-10)
    p.add_argument("--C_max", type=int, default=10)
    p.add_argument("--M_min", type=float, default=5.0)
    p.add_argument("--M_max", type=float, default=15.0)
    p.add_argument("--X_move", type=float, default=1.0)
    p.add_argument("--X_call", type=float, default=0.25)
    p.add_argument("--calls_per_agent", type=int, default=2)
    p.add_argument("--base_seed", type=int, default=20260519)

    # Entropy parameters
    p.add_argument("--alpha_smooth", type=float, default=0.5)
    p.add_argument("--q_tsallis", type=float, default=1.5)
    p.add_argument("--beta_tail", type=float, default=1.0)
    p.add_argument("--beta_bimodal", type=float, default=1.0)
    p.add_argument("--tail_threshold", type=int, default=None)
    p.add_argument("--center_threshold", type=int, default=None)
    p.add_argument("--kappa_center", type=float, default=1.0)

    # Compatibility / inspection
    p.add_argument("--export_excel_representatives", action="store_true", help="Also export representative runs to the old Excel sheet-per-step format.")
    return p


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
