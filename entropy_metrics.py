#!/usr/bin/env python3
"""
Entropy and structural metrics for generalized Schelling simulations.

This file is shared by the batch runner and figure scripts. It deliberately
contains no plotting and no simulation code.

Critical convention
-------------------
Empty cells are np.nan. The affiliation value C = 0 is a valid neutral state
and is included in all histograms over C.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from simulation_model import SimulationParams, count_agreeing_neighbors, neighbors_moore, occupied_mask


MICROSTATE_CODES = {"B": 0.0, "A": 1.0, "T": 2.0}


def classify_microstates(C: np.ndarray, M: np.ndarray, params: SimulationParams) -> np.ndarray:
    """
    Classify occupied cells into B/A/T microstates.

    B = bound/satisfied: enough same-sign neighbors.
    A = activated/mobile: dissatisfied but has enough money to move.
    T = trapped: dissatisfied and cannot afford movement.

    For scenarios 1-3, money does not constrain movement dynamically, but M is still
    initialized and usually exceeds X_move. Thus A/T remain interpretable.
    """
    N = C.shape[0]
    st = np.full_like(C, np.nan, dtype=float)
    occ = occupied_mask(C)

    for i, j in np.column_stack(np.where(occ)):
        i, j = int(i), int(j)
        agree = count_agreeing_neighbors(C, i, j)
        unhappy = agree < params.happy_threshold
        can_move = (not np.isnan(M[i, j])) and (M[i, j] >= params.X_move)
        if not unhappy:
            st[i, j] = MICROSTATE_CODES["B"]
        elif can_move:
            st[i, j] = MICROSTATE_CODES["A"]
        else:
            st[i, j] = MICROSTATE_CODES["T"]
    return st


def histogram_C_by_state(
    C: np.ndarray,
    st: np.ndarray,
    C_min: int,
    C_max: int,
    alpha: float = 0.5,
) -> Dict[str, Dict[str, np.ndarray | int]]:
    """
    Build smoothed pmfs over C for each microstate.

    C=0 is included as a valid bin whenever C_min <= 0 <= C_max.
    """
    grid = np.arange(C_min, C_max + 1)
    out: Dict[str, Dict[str, np.ndarray | int]] = {}

    for key, code in MICROSTATE_CODES.items():
        mask = st == code
        vals = C[mask].astype(float)
        counts = np.zeros(len(grid), dtype=float)
        if vals.size > 0:
            vals = np.rint(vals).astype(int)
            idx = vals - C_min
            idx = idx[(idx >= 0) & (idx < len(grid))]
            np.add.at(counts, idx, 1.0)
        p = (counts + alpha) / (counts.sum() + alpha * len(grid))
        out[key] = {"p": p, "counts": counts, "n": int(counts.sum())}
    return out


# ----------------------------------------------------------------------
# Reference distributions and shape weights
# ----------------------------------------------------------------------

def discrete_normalized(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(arr, dtype=float), 0.0)
    z = float(arr.sum())
    if z <= 0:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / z


def pmf_gaussian(grid: np.ndarray, sigma: float) -> np.ndarray:
    return discrete_normalized(np.exp(-(grid ** 2) / (2.0 * sigma * sigma + 1e-12)))


def pmf_student_t(grid: np.ndarray, nu: float) -> np.ndarray:
    return discrete_normalized((1.0 + (grid ** 2) / float(nu)) ** (-(nu + 1.0) / 2.0))


def pmf_bimodal_sym(grid: np.ndarray, mu: float, sigma: float, pi: float = 0.5) -> np.ndarray:
    g1 = np.exp(-((grid - mu) ** 2) / (2.0 * sigma * sigma + 1e-12))
    g2 = np.exp(-((grid + mu) ** 2) / (2.0 * sigma * sigma + 1e-12))
    return discrete_normalized(pi * g1 + (1.0 - pi) * g2)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def fit_reference_weights(
    p: np.ndarray,
    grid: np.ndarray,
    q_tail_candidates: Tuple[int, ...] = (2, 3, 5, 10),
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """Return KL-derived (wN, wT, wB) and metadata."""
    p = np.asarray(p, dtype=float)
    mean = float((grid * p).sum())
    var = float(((grid - mean) ** 2 * p).sum())
    sigma = max(math.sqrt(max(var, 0.0)), 0.5)

    rN = pmf_gaussian(grid, sigma)

    rT_candidates = [pmf_student_t(grid, nu) for nu in q_tail_candidates]
    dT_list = [kl_divergence(p, r) for r in rT_candidates]
    ix = int(np.argmin(dT_list))
    rT = rT_candidates[ix]
    nu_chosen = q_tail_candidates[ix]

    mode_idx = int(np.argmax(p))
    mu_guess = max(1, abs(int(grid[mode_idx])))
    rB = pmf_bimodal_sym(grid, mu=mu_guess, sigma=max(sigma, 1.0), pi=0.5)

    rU = np.ones_like(p, dtype=float) / len(p)
    dN = kl_divergence(p, rN)
    dT = kl_divergence(p, rT)
    dB = kl_divergence(p, rB)
    dU = kl_divergence(p, rU)

    def fit_pct(dk: float) -> float:
        return max(0.0, 1.0 - dk / (dU + 1e-12))

    fN, fT, fB = fit_pct(dN), fit_pct(dT), fit_pct(dB)
    denom = fN + fT + fB
    if denom <= 0:
        wN = wT = wB = 1.0 / 3.0
    else:
        wN, wT, wB = fN / denom, fT / denom, fB / denom

    meta = {
        "sigma_gauss": sigma,
        "nu_student": nu_chosen,
        "mu_bimodal": mu_guess,
        "KL_N": dN,
        "KL_T": dT,
        "KL_B": dB,
        "KL_U": dU,
        "fit_N": fN,
        "fit_T": fT,
        "fit_B": fB,
    }
    return (float(wN), float(wT), float(wB)), meta


# ----------------------------------------------------------------------
# Entropy cores and masks
# ----------------------------------------------------------------------

def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.sum(-p * np.log(p + eps)))


def tsallis_entropy(p: np.ndarray, q: float = 1.5) -> float:
    p = np.asarray(p, dtype=float)
    if abs(q - 1.0) < 1e-9:
        return shannon_entropy(p)
    return float(np.sum((p - p ** q) / (q - 1.0)))


def build_masks(
    grid: np.ndarray,
    tail_threshold: int,
    center_threshold: int,
    kappa_center: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    abs_grid = np.abs(grid)
    m_tail = (abs_grid >= tail_threshold).astype(float)
    m_bi = np.where(abs_grid >= center_threshold, 1.0, -float(kappa_center))
    return m_tail, m_bi


def shape_entropy_components(
    p: np.ndarray,
    grid: np.ndarray,
    q_tsallis: float,
    beta_tail: float,
    beta_bimodal: float,
    m_tail: np.ndarray,
    m_bi: np.ndarray,
) -> Dict[str, float]:
    HN = shannon_entropy(p)
    Hq = tsallis_entropy(p, q=q_tsallis)
    tail_score = float(beta_tail * np.sum(m_tail * p))
    bimodal_score = float(beta_bimodal * np.sum(m_bi * p))
    return {
        "H_shannon": HN,
        "H_tsallis": Hq,
        "tail_score": tail_score,
        "bimodal_score": bimodal_score,
    }


def shape_weighted_entropy(
    components: Dict[str, float],
    weights: Tuple[float, float, float],
) -> float:
    wN, wT, wB = weights
    return float(
        wN * components["H_shannon"]
        + wT * components["H_tsallis"]
        + wT * components["tail_score"]
        + wB * components["bimodal_score"]
    )


# ----------------------------------------------------------------------
# Independent structural metrics
# ----------------------------------------------------------------------

def bimodality_coefficient(C: np.ndarray) -> float:
    """Bias-corrected sample bimodality coefficient (Pfister et al., 2013).

    Values above the uniform-distribution benchmark 5/9 are conventionally
    treated as evidence compatible with bimodality.  The coefficient is a
    distributional benchmark, not a spatial statistic.
    """
    vals = C[occupied_mask(C)].astype(float)
    n = int(vals.size)
    if n < 4 or float(np.nanstd(vals)) == 0.0:
        return np.nan
    g = float(skew(vals, bias=False))
    k_excess = float(kurtosis(vals, fisher=True, bias=False))
    correction = 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0))
    denom = k_excess + correction
    if denom <= 0.0:
        return np.nan
    return float((g * g + 1.0) / denom)


def esteban_ray_polarization(C: np.ndarray, alpha: float = 1.6) -> float:
    """Normalized discrete Esteban--Ray polarization index.

    Affiliation positions are rescaled to [0, 1].  The raw identification-
    alienation sum is divided by its value for two equally sized groups at the
    endpoints, so that the canonical maximally separated bipolar distribution
    has value one.
    """
    vals = C[occupied_mask(C)].astype(float)
    if vals.size == 0:
        return np.nan
    support, counts = np.unique(vals, return_counts=True)
    if support.size < 2:
        return 0.0
    p = counts.astype(float) / float(counts.sum())
    span = float(np.max(support) - np.min(support))
    if span <= 0.0:
        return 0.0
    y = (support - np.min(support)) / span
    distances = np.abs(y[:, None] - y[None, :])
    raw = float(np.sum((p[:, None] ** (1.0 + alpha)) * p[None, :] * distances))
    bipolar_reference = 2.0 * (0.5 ** (2.0 + alpha))
    return float(raw / bipolar_reference)

def polarization_index(C: np.ndarray, max_abs_C: float) -> float:
    vals = C[occupied_mask(C)]
    if vals.size == 0 or max_abs_C <= 0:
        return np.nan
    return float(np.mean(np.abs(vals)) / max_abs_C)


def central_depletion_index(C: np.ndarray, center_threshold: int) -> float:
    vals = C[occupied_mask(C)]
    if vals.size == 0:
        return np.nan
    return float(1.0 - np.mean(np.abs(vals) < center_threshold))


def tail_mass_index(C: np.ndarray, tail_threshold: int) -> float:
    vals = C[occupied_mask(C)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(np.abs(vals) >= tail_threshold))


def spatial_segregation_metrics(C: np.ndarray) -> Tuple[float, float, float]:
    """Return same-sign edge share, Moran's I, and sign assortativity.

    All three measures use the same occupied Moore edges and are evaluated in a
    single vectorized pass.  This preserves their definitions while keeping the
    120-run, 101-step validation computationally practical.
    """
    occ = occupied_mask(C)
    vals = C[occ].astype(float)
    n = int(vals.size)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    denom = float(np.sum((vals - mean) ** 2))

    edge_pairs = [
        (C[:, :-1], C[:, 1:]),
        (C[:-1, :], C[1:, :]),
        (C[:-1, :-1], C[1:, 1:]),
        (C[:-1, 1:], C[1:, :-1]),
    ]
    moran_numerator = 0.0
    n_edges = 0
    same = 0
    sign_edges = 0
    mixing = np.zeros((2, 2), dtype=float)
    for left, right in edge_pairs:
        valid = ~np.isnan(left) & ~np.isnan(right)
        if np.any(valid):
            x = left[valid]
            y = right[valid]
            moran_numerator += float(np.sum((x - mean) * (y - mean)))
            n_edges += int(x.size)

        valid_sign = valid & (left != 0) & (right != 0)
        if np.any(valid_sign):
            x_sign = left[valid_sign] > 0
            y_sign = right[valid_sign] > 0
            same += int(np.sum(x_sign == y_sign))
            sign_edges += int(x_sign.size)
            np.add.at(mixing, (x_sign.astype(int), y_sign.astype(int)), 1.0)
            np.add.at(mixing, (y_sign.astype(int), x_sign.astype(int)), 1.0)

    pair_share = float(same / sign_edges) if sign_edges > 0 else np.nan
    moran = (
        float((n / n_edges) * (moran_numerator / denom))
        if n_edges > 0 and denom > 0.0
        else np.nan
    )
    total = float(mixing.sum())
    if total <= 0.0:
        return pair_share, moran, np.nan
    e = mixing / total
    a = e.sum(axis=1)
    b = e.sum(axis=0)
    expected = float(np.sum(a * b))
    assort_denom = 1.0 - expected
    assort = (
        float((np.trace(e) - expected) / assort_denom)
        if assort_denom > 1e-12
        else np.nan
    )
    return pair_share, moran, assort


def segregation_index(C: np.ndarray) -> float:
    """Fraction of occupied non-neutral Moore edges with the same sign."""
    return spatial_segregation_metrics(C)[0]


def morans_i(C: np.ndarray) -> float:
    """Global Moran's I using binary Moore-contiguity weights."""
    return spatial_segregation_metrics(C)[1]


def sign_assortativity(C: np.ndarray) -> float:
    """Newman's nominal assortativity of affiliation sign on Moore edges."""
    return spatial_segregation_metrics(C)[2]


# ----------------------------------------------------------------------
# Per-step and per-history metric computation
# ----------------------------------------------------------------------

def compute_step_metrics(
    C: np.ndarray,
    M: np.ndarray,
    params: SimulationParams,
    alpha_smooth: float = 0.5,
    q_tsallis: float = 1.5,
    beta_tail: float = 1.0,
    beta_bimodal: float = 1.0,
    tail_threshold: Optional[int] = None,
    center_threshold: Optional[int] = None,
    kappa_center: float = 1.0,
    include_structural: bool = True,
) -> Dict[str, float]:
    C_min, C_max = params.C_range
    max_abs_C = max(abs(C_min), abs(C_max))
    if tail_threshold is None:
        tail_threshold = int(round(0.7 * max_abs_C))
    if center_threshold is None:
        center_threshold = int(round(0.4 * max_abs_C))

    grid = np.arange(C_min, C_max + 1)
    m_tail, m_bi = build_masks(grid, tail_threshold, center_threshold, kappa_center=kappa_center)

    st = classify_microstates(C, M, params)
    hists = histogram_C_by_state(C, st, C_min, C_max, alpha=alpha_smooth)

    nB = hists["B"]["n"]
    nA = hists["A"]["n"]
    nT = hists["T"]["n"]
    n_total = max(1, int(nB + nA + nT))
    q_state = {"B": nB / n_total, "A": nA / n_total, "T": nT / n_total}

    row: Dict[str, float] = {
        "qB": q_state["B"],
        "qA": q_state["A"],
        "qT": q_state["T"],
        "tail_threshold": tail_threshold,
        "center_threshold": center_threshold,
        "q_tsallis": q_tsallis,
        "beta_tail": beta_tail,
        "beta_bimodal": beta_bimodal,
    }
    if include_structural:
        pair_share, moran_value, assortativity_value = spatial_segregation_metrics(C)
        row.update({
            "polarization": polarization_index(C, max_abs_C=max_abs_C),
            "central_depletion": central_depletion_index(C, center_threshold=center_threshold),
            "tail_mass": tail_mass_index(C, tail_threshold=tail_threshold),
            "segregation": pair_share,
            "bimodality_coefficient": bimodality_coefficient(C),
            "esteban_ray_polarization": esteban_ray_polarization(C),
            "morans_I": moran_value,
            "sign_assortativity": assortativity_value,
        })

    lattice = {
        "S_full_lattice": 0.0,
        "S_shannon_lattice": 0.0,
        "S_tail_ablation_lattice": 0.0,
        "S_bimodal_ablation_lattice": 0.0,
        "tail_score_lattice": 0.0,
        "bimodal_score_lattice": 0.0,
        "weighted_tail_contribution_lattice": 0.0,
        "weighted_bimodal_contribution_lattice": 0.0,
        "wN_lattice": 0.0,
        "wT_lattice": 0.0,
        "wB_lattice": 0.0,
    }

    for key in ["B", "A", "T"]:
        p = hists[key]["p"]
        n = int(hists[key]["n"])
        qX = q_state[key]
        row[f"n_{key}"] = n

        if n == 0:
            for col in [
                "H_shannon", "H_tsallis", "tail_score", "bimodal_score",
                "S_full", "S_tail_ablation", "S_bimodal_ablation",
                "wN", "wT", "wB",
            ]:
                row[f"{col}_{key}"] = np.nan
            continue

        weights, _meta = fit_reference_weights(p, grid)
        comps = shape_entropy_components(
            p, grid,
            q_tsallis=q_tsallis,
            beta_tail=beta_tail,
            beta_bimodal=beta_bimodal,
            m_tail=m_tail,
            m_bi=m_bi,
        )
        S_full = shape_weighted_entropy(comps, weights)

        # Simple ablations. These intentionally use the same raw scores, but remove
        # the adaptive shape weighting to keep the comparisons transparent.
        S_tail_ablation = comps["H_shannon"] + comps["tail_score"]
        S_bimodal_ablation = comps["H_shannon"] + comps["bimodal_score"]

        row[f"H_shannon_{key}"] = comps["H_shannon"]
        row[f"H_tsallis_{key}"] = comps["H_tsallis"]
        row[f"tail_score_{key}"] = comps["tail_score"]
        row[f"bimodal_score_{key}"] = comps["bimodal_score"]
        row[f"S_full_{key}"] = S_full
        row[f"S_tail_ablation_{key}"] = S_tail_ablation
        row[f"S_bimodal_ablation_{key}"] = S_bimodal_ablation
        row[f"wN_{key}"], row[f"wT_{key}"], row[f"wB_{key}"] = weights

        lattice["S_full_lattice"] += qX * S_full
        lattice["S_shannon_lattice"] += qX * comps["H_shannon"]
        lattice["S_tail_ablation_lattice"] += qX * S_tail_ablation
        lattice["S_bimodal_ablation_lattice"] += qX * S_bimodal_ablation
        lattice["tail_score_lattice"] += qX * comps["tail_score"]
        lattice["bimodal_score_lattice"] += qX * comps["bimodal_score"]
        lattice["weighted_tail_contribution_lattice"] += qX * weights[1] * comps["tail_score"]
        lattice["weighted_bimodal_contribution_lattice"] += qX * weights[2] * comps["bimodal_score"]
        lattice["wN_lattice"] += qX * weights[0]
        lattice["wT_lattice"] += qX * weights[1]
        lattice["wB_lattice"] += qX * weights[2]

    row.update(lattice)
    return row


def compute_metrics_for_history(
    C_history: np.ndarray,
    M_history: np.ndarray,
    params: SimulationParams,
    stats_df: Optional[pd.DataFrame] = None,
    alpha_smooth: float = 0.5,
    q_tsallis: float = 1.5,
    beta_tail: float = 1.0,
    beta_bimodal: float = 1.0,
    tail_threshold: Optional[int] = None,
    center_threshold: Optional[int] = None,
    kappa_center: float = 1.0,
    include_structural: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for t in range(C_history.shape[0]):
        row = compute_step_metrics(
            C_history[t], M_history[t], params=params,
            alpha_smooth=alpha_smooth,
            q_tsallis=q_tsallis,
            beta_tail=beta_tail,
            beta_bimodal=beta_bimodal,
            tail_threshold=tail_threshold,
            center_threshold=center_threshold,
            kappa_center=kappa_center,
            include_structural=include_structural,
        )
        row["step"] = t
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    if stats_df is not None:
        df = df.merge(stats_df, on="step", how="left")
    return df
