#!/usr/bin/env python3
"""
Simulation engine for the paper-ready generalized Schelling model.

Main fixes relative to the older exploratory scripts
----------------------------------------------------
1. Empty cells are represented only by np.nan.
2. C = 0 is treated as a valid neutral affiliation/conviction value, not as an empty cell.
3. The C clipping bounds are taken from C_range, not hard-coded to [-10, 10].
4. Dynamic random seeds vary across runs, so multi-seed analyses are genuine stochastic runs.
5. Scenario 4 money costs are tracked explicitly through step-level statistics.

Scenarios
---------
1. Pure Schelling movement with limited mobility.
2. Scenario 1 + local affiliation/conviction interaction.
3. Scenario 2 + distant phone-call-like interactions.
4. Scenario 3 + money costs for movement and calls.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


@dataclass
class SimulationParams:
    scenario: int = 1
    N: int = 64
    steps: int = 100
    vacancy: float = 0.30
    R_move: int = 4
    happy_threshold: int = 4
    C_range: Tuple[int, int] = (-10, 10)
    M_range: Tuple[float, float] = (5.0, 15.0)
    X_move: float = 1.0
    X_call: float = 0.25
    calls_per_agent: int = 2
    seed_positions: int = 1
    seed_affil: int = 2
    seed_money: int = 3
    seed_dynamics: int = 4

    def to_jsonable(self) -> Dict:
        d = asdict(self)
        d["C_range"] = list(self.C_range)
        d["M_range"] = list(self.M_range)
        return d


def occupied_mask(C: np.ndarray) -> np.ndarray:
    """True for occupied cells. Empty cells are np.nan only."""
    return ~np.isnan(C)


def initialize_grids(params: SimulationParams) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize affiliation C and money M grids with independent random seeds."""
    N = params.N
    rng_pos = np.random.default_rng(params.seed_positions)
    rng_aff = np.random.default_rng(params.seed_affil)
    rng_money = np.random.default_rng(params.seed_money)

    num_cells = N * N
    num_empty = int(round(params.vacancy * num_cells))
    all_idx = np.arange(num_cells)
    empty_flat = rng_pos.choice(all_idx, size=num_empty, replace=False)

    occ_flat = np.ones(num_cells, dtype=bool)
    occ_flat[empty_flat] = False

    C_flat = np.full(num_cells, np.nan, dtype=float)
    C_flat[occ_flat] = rng_aff.integers(
        params.C_range[0], params.C_range[1] + 1, size=int(occ_flat.sum())
    )

    M_flat = np.full(num_cells, np.nan, dtype=float)
    M_flat[occ_flat] = rng_money.uniform(
        params.M_range[0], params.M_range[1], size=int(occ_flat.sum())
    )

    return C_flat.reshape(N, N), M_flat.reshape(N, N)


def neighbors_moore(N: int, i: int, j: int) -> Iterable[Tuple[int, int]]:
    """Yield valid Moore-neighborhood coordinates around cell (i, j)."""
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < N and 0 <= jj < N:
                yield ii, jj


def count_agreeing_neighbors(C: np.ndarray, i: int, j: int) -> int:
    """
    Count neighboring agents with the same non-zero sign of C.

    Neutral agents (C = 0) are valid occupied agents, but they do not contribute
    to same-sign agreement. This preserves the interpretation of C=0 as a neutral
    or weakly affiliated state rather than an empty site.
    """
    if np.isnan(C[i, j]) or C[i, j] == 0:
        return 0
    N = C.shape[0]
    s0 = np.sign(C[i, j])
    agree = 0
    for ii, jj in neighbors_moore(N, i, j):
        if np.isnan(C[ii, jj]) or C[ii, jj] == 0:
            continue
        if np.sign(C[ii, jj]) == s0:
            agree += 1
    return agree


def step_schelling(
    C: np.ndarray,
    M: np.ndarray,
    params: SimulationParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run one simulation step and return updated grids plus step statistics."""
    N = C.shape[0]
    scenario = params.scenario

    stats = {
        "moves_attempted": 0,
        "moves_made": 0,
        "calls_attempted": 0,
        "calls_made": 0,
        "call_cost_total": 0.0,
        "move_cost_total": 0.0,
        "n_occupied": int(occupied_mask(C).sum()),
        "n_unhappy_before_move": 0,
    }

    occ = occupied_mask(C)
    coords = np.column_stack(np.where(occ))
    rng.shuffle(coords)

    # ------------------------------------------------------------------
    # Interaction updates: local persuasion in scenarios 2-4 and distant
    # phone-like calls in scenarios 3-4.
    # ------------------------------------------------------------------
    if scenario >= 2 and len(coords) > 0:
        C_prev = C.copy()
        M_prev = M.copy()
        delta_local = np.zeros_like(C_prev, dtype=float)
        delta_dist = np.zeros_like(C_prev, dtype=float)

        # Local sign influence from Moore neighborhood.
        for i, j in coords:
            s = 0.0
            for ii, jj in neighbors_moore(N, int(i), int(j)):
                if np.isnan(C_prev[ii, jj]):
                    continue
                s += np.sign(C_prev[ii, jj])
            delta_local[i, j] = s

        # Distant calls.
        if scenario >= 3:
            occupied = np.column_stack(np.where(occupied_mask(C_prev)))
            n_occ = len(occupied)
            for i, j in coords:
                call_cost = params.calls_per_agent * params.X_call
                stats["calls_attempted"] += params.calls_per_agent

                if scenario >= 4:
                    if np.isnan(M_prev[i, j]) or M_prev[i, j] < call_cost:
                        continue
                    # Pay once for the set of distant calls in this step.
                    M[i, j] -= call_cost
                    stats["call_cost_total"] += float(call_cost)

                for _ in range(params.calls_per_agent):
                    k = int(rng.integers(n_occ))
                    ii, jj = occupied[k]
                    delta_dist[i, j] += np.sign(C_prev[ii, jj])
                    stats["calls_made"] += 1

        C = np.clip(C + delta_local + delta_dist, params.C_range[0], params.C_range[1])

    # ------------------------------------------------------------------
    # Movement update.
    # ------------------------------------------------------------------
    occ = occupied_mask(C)
    coords = np.column_stack(np.where(occ))
    rng.shuffle(coords)
    empties = set(zip(*np.where(~occ)))

    for i_raw, j_raw in coords:
        i, j = int(i_raw), int(j_raw)
        if np.isnan(C[i, j]):
            # The agent may have moved earlier in this same sweep.
            continue

        agree = count_agreeing_neighbors(C, i, j)
        unhappy = agree < params.happy_threshold
        if not unhappy:
            continue

        stats["n_unhappy_before_move"] += 1
        stats["moves_attempted"] += 1

        candidates: List[Tuple[int, int]] = []
        for di in range(-params.R_move, params.R_move + 1):
            for dj in range(-params.R_move, params.R_move + 1):
                if abs(di) + abs(dj) > params.R_move:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N and (ni, nj) in empties:
                    candidates.append((ni, nj))

        if not candidates:
            continue

        if scenario >= 4 and params.X_move > 0:
            if np.isnan(M[i, j]) or M[i, j] < params.X_move:
                continue
            M[i, j] -= params.X_move
            stats["move_cost_total"] += float(params.X_move)

        ni, nj = candidates[int(rng.integers(len(candidates)))]

        C[ni, nj] = C[i, j]
        C[i, j] = np.nan
        M[ni, nj] = M[i, j]
        M[i, j] = np.nan
        empties.remove((ni, nj))
        empties.add((i, j))
        stats["moves_made"] += 1

    occ_after = occupied_mask(C)
    stats["n_occupied_after"] = int(occ_after.sum())
    stats["mean_money"] = float(np.nanmean(M)) if np.any(occ_after) else np.nan
    stats["min_money"] = float(np.nanmin(M)) if np.any(occ_after) else np.nan
    stats["max_money"] = float(np.nanmax(M)) if np.any(occ_after) else np.nan
    return C, M, stats


def run_simulation(params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Run one complete simulation.

    Returns
    -------
    C_history, M_history:
        Arrays with shape (steps + 1, N, N).
    stats_df:
        Step-level simulation statistics. Step 0 is included with zero movements/calls.
    """
    C, M = initialize_grids(params)
    rng = np.random.default_rng(params.seed_dynamics)

    C_history = np.full((params.steps + 1, params.N, params.N), np.nan, dtype=float)
    M_history = np.full((params.steps + 1, params.N, params.N), np.nan, dtype=float)
    C_history[0] = C.copy()
    M_history[0] = M.copy()

    rows: List[Dict[str, float]] = []
    rows.append({
        "step": 0,
        "moves_attempted": 0,
        "moves_made": 0,
        "calls_attempted": 0,
        "calls_made": 0,
        "call_cost_total": 0.0,
        "move_cost_total": 0.0,
        "n_occupied": int(occupied_mask(C).sum()),
        "n_occupied_after": int(occupied_mask(C).sum()),
        "n_unhappy_before_move": int(sum(
            count_agreeing_neighbors(C, i, j) < params.happy_threshold
            for i, j in np.column_stack(np.where(occupied_mask(C)))
        )),
        "mean_money": float(np.nanmean(M)),
        "min_money": float(np.nanmin(M)),
        "max_money": float(np.nanmax(M)),
    })

    for step in range(1, params.steps + 1):
        C, M, stats = step_schelling(C, M, params=params, rng=rng)
        C_history[step] = C.copy()
        M_history[step] = M.copy()
        stats["step"] = step
        rows.append(stats)

    return C_history, M_history, pd.DataFrame(rows).sort_values("step")


def save_npz_run(
    out_path: str | Path,
    C_history: np.ndarray,
    M_history: np.ndarray,
    params: SimulationParams,
    stats_df: Optional[pd.DataFrame] = None,
) -> None:
    """Save a simulation history in compressed NumPy format plus sidecar metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, C_history=C_history, M_history=M_history)
    meta_path = out_path.with_suffix(".params.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(params.to_jsonable(), f, indent=2)
    if stats_df is not None:
        stats_df.to_csv(out_path.with_suffix(".stats.csv"), index=False)


def export_history_to_excel(
    C_history: np.ndarray,
    M_history: np.ndarray,
    params: SimulationParams,
    out_dir: str | Path,
) -> None:
    """
    Export a simulation history to the old Excel format.

    This is useful for visual inspection or compatibility with old scripts. The new
    batch pipeline does not need Excel internally.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fileC = out_dir / "C_all_steps.xlsx"
    fileM = out_dir / "M_all_steps.xlsx"
    params_df = pd.DataFrame([params.to_jsonable()])

    with pd.ExcelWriter(fileC) as writerC, pd.ExcelWriter(fileM) as writerM:
        params_df.to_excel(writerC, sheet_name="parameters", index=False)
        params_df.to_excel(writerM, sheet_name="parameters", index=False)
        for t in range(C_history.shape[0]):
            pd.DataFrame(C_history[t]).to_excel(writerC, sheet_name=f"step_{t:03d}", header=False, index=False)
            pd.DataFrame(M_history[t]).to_excel(writerM, sheet_name=f"step_{t:03d}", header=False, index=False)
