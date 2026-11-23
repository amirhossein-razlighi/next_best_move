"""Random layouts and the offline solver that labels them (training targets and the "hard" flag; never used by the models).

Layout: cue ball (white), the red target and 0-4 other object balls of distinct colours (yellow, blue, purple, orange: obstacles,
which a combination may also use), uniformly placed on the cloth: centres >= R + 0.005 from the cushions, >= rc + R + 0.03 from
every pocket point, >= 2R + 0.02 apart.

Solver: every angle on a 0.25 deg grid x 9 powers 0.15 .. 0.95 (12,960 simulated shots). Among the contiguous successful angle
intervals (per power; the 8 widest) the pick is the centre of the WIDEST one (edges refined by bisection to < 0.01 deg) whose
centre also succeeds at power +- 0.05 (if none does: the widest). The label power is min(p_lo + 0.15, (p_lo + p_hi) / 2) on the
contiguous successful power interval [p_lo, p_hi] (0.01 grid) through that angle (a firm but controlled speed; kept at the grid
power if it fails); then the angle is re-centred on the successful angle interval through it AT the label power (scan +- 10 deg
in 0.05 deg steps, edges bisected; `width` = that interval). The label shot always succeeds.

HARD layout: no DIRECT successful shot anywhere on the solver grid (physics.direct_mask: the cue hits the red first without a
cushion, the red drops without a cushion or another contact), so a bank, kick or combination is needed.
"""
from __future__ import annotations

import math

import numpy as np

from nbm.tasks.billiards import physics as P

MAX_OTHERS = 4
ANG_STEP = 0.25
ANG_GRID = np.arange(int(round(360 / ANG_STEP))) * ANG_STEP          # 1440 angles
POW_GRID = np.round(np.arange(0.15, 0.96, 0.10), 2)                  # 0.15 .. 0.95 (9 powers)
POW_FINE = np.round(np.arange(0.05, 1.001, 0.01), 2)                 # power refinement grid
POW_ROBUST = 0.05
POW_MARGIN = 0.15                                                    # label power = min(p_lo + 0.15, middle of the interval)
LOC_STEP, LOC_N = 0.05, 200                                          # final angle re-centring scan (+- 10 deg)
MIN_SEP = 2 * P.R + 0.02                                             # layout: centre distance between balls
POCKET_MARGIN = 0.03                                                 # layout: centre >= rc + R + margin from a pocket point
WALL_MARGIN = 0.005


def random_layout(rng: np.random.Generator, n_others: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(xy (n, 2), colours (n,)): cue (colour 0), red (1), then n_others (default: uniform 0..4) distinct colours of 2..5, sorted."""
    k = int(rng.integers(0, MAX_OTHERS + 1)) if n_others is None else int(n_others)
    cols = [0, 1] + sorted(int(c) for c in rng.choice([2, 3, 4, 5], size=k, replace=False))
    lo, hi = P.R + WALL_MARGIN, np.array([P.L, P.W]) - P.R - WALL_MARGIN
    pts: list[np.ndarray] = []
    while len(pts) < len(cols):
        p = rng.uniform(lo, hi)
        if np.any(np.hypot(*(P.POCKETS - p).T) < P.POCKET_R + P.R + POCKET_MARGIN):
            continue
        if any(math.hypot(*(p - q)) < MIN_SEP for q in pts):
            continue
        pts.append(p)
    return np.array(pts), np.array(cols, dtype=np.int64)


def _runs(b: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a circular bool array -> [(start, length)]."""
    n = len(b)
    if b.all():
        return [(0, n)]
    if not b.any():
        return []
    s0 = int(np.argmin(b))  # a False index: start scanning after it
    out, i = [], 0
    while i < n:
        j = (s0 + 1 + i) % n
        if b[j]:
            k = 0
            while k < n and b[(j + k) % n]:
                k += 1
            out.append((j, k))
            i += k
        else:
            i += 1
    return out


def _refine_edge(xy, alive, inside: float, outside: float, p: float, iters: int = 5) -> float:
    """Bisection between a successful angle `inside` and a failing `outside` (degrees, at most one grid step apart)."""
    for _ in range(iters):
        m = 0.5 * (inside + outside)
        if P.shot_ok(xy, alive, m % 360.0, p):
            inside = m
        else:
            outside = m
    return inside


def solve(xy, alive=None, top_k: int = 8) -> dict | None:
    """The label of a layout (module doc), or None if no shot on the grid succeeds. Keys: angle, power, width (successful angle
    interval at the label power), ang_lo / ang_hi, grid_power, grid_width, power_robust, grid_power_robust, p_lo / p_hi, pocket,
    shot_type, n_succ / frac_succ / n_direct (grid counts), hard, cue_final, n_potted_obj, t_rest."""
    xy = P.as_xy(xy)
    na, npw = len(ANG_GRID), len(POW_GRID)
    angs, pows = np.tile(ANG_GRID, npw), np.repeat(POW_GRID, na)
    pm, info = P.batch(xy, alive, angs, pows)
    succ = P.success_mask(pm).reshape(npw, na)
    direct = P.direct_mask(pm, info).reshape(npw, na)
    n_succ = int(succ.sum())
    out = {"n_succ": n_succ, "frac_succ": n_succ / succ.size, "n_direct": int(direct.sum()), "hard": bool(not direct.any())}
    if n_succ == 0:
        return None
    cands = []
    for k in range(npw):
        for s, ln in _runs(succ[k]):
            cands.append((ln, -k, s, k))
    cands.sort(reverse=True)
    best = None
    for ln, _, s, k in cands[:top_k]:
        p = float(POW_GRID[k])
        if ln == na:
            lo, hi = 0.0, 360.0
        else:
            first, last = ANG_GRID[s], ANG_GRID[s] + (ln - 1) * ANG_STEP
            lo = _refine_edge(xy, alive, first, first - ANG_STEP, p)
            hi = _refine_edge(xy, alive, last, last + ANG_STEP, p)
        c = (0.5 * (lo + hi)) % 360.0
        width = hi - lo
        if not P.shot_ok(xy, alive, c, p):  # a hole inside the run (rare): the grid point nearest the centre
            c = float(ANG_GRID[(s + ln // 2) % na])
        robust = P.shot_ok(xy, alive, c, p - POW_ROBUST) and P.shot_ok(xy, alive, c, min(1.0, p + POW_ROBUST))
        key = (robust, width, -p)
        if best is None or key > best[0]:
            best = (key, c, p, width, lo, hi, robust)
    (_, c, p, width, lo, hi, robust) = best
    # label power: on the successful power interval through c
    pmf, _ = P.batch(xy, alive, np.full(len(POW_FINE), c), POW_FINE)
    okf = P.success_mask(pmf)
    j = int(np.argmin(np.abs(POW_FINE - p)))
    p_final = p
    if okf[j]:
        a_, b_ = j, j
        while a_ > 0 and okf[a_ - 1]:
            a_ -= 1
        while b_ < len(POW_FINE) - 1 and okf[b_ + 1]:
            b_ += 1
        p_lo, p_hi = float(POW_FINE[a_]), float(POW_FINE[b_])
        pm_ = round(min(p_lo + POW_MARGIN, 0.5 * (p_lo + p_hi)), 4)
        if P.shot_ok(xy, alive, c, pm_):
            p_final = pm_
    else:
        p_lo = p_hi = p
    # re-centre the angle on the successful interval through c at the label power (local scan, then bisection)
    loc = c + LOC_STEP * np.arange(-LOC_N, LOC_N + 1)
    pml, _ = P.batch(xy, alive, loc % 360.0, np.full(len(loc), p_final))
    okl = P.success_mask(pml)
    a_, b_ = LOC_N, LOC_N
    if okl[LOC_N]:
        while a_ > 0 and okl[a_ - 1]:
            a_ -= 1
        while b_ < len(loc) - 1 and okl[b_ + 1]:
            b_ += 1
        lo2 = _refine_edge(xy, alive, loc[a_], loc[a_] - LOC_STEP, p_final) if a_ > 0 else loc[0]
        hi2 = _refine_edge(xy, alive, loc[b_], loc[b_] + LOC_STEP, p_final) if b_ < len(loc) - 1 else loc[-1]
        c2 = (0.5 * (lo2 + hi2)) % 360.0
        if P.shot_ok(xy, alive, c2, p_final):
            c = c2
        width_f, lo_f, hi_f = hi2 - lo2, lo2, hi2
    else:
        width_f, lo_f, hi_f = width, lo, hi
    sim = P.simulate(xy, alive, c, p_final)
    if not (sim["pocket"][1] >= 0 and sim["pocket"][0] < 0):
        raise AssertionError("the label shot must succeed")
    out.update(angle=float(c), power=float(p_final), grid_power=p, grid_width=float(width), width=float(width_f),
               ang_lo=float(lo_f % 360.0), ang_hi=float(hi_f % 360.0), grid_power_robust=bool(robust),
               power_robust=bool(P.shot_ok(xy, alive, c, max(0.0, p_final - POW_ROBUST))
                                 and P.shot_ok(xy, alive, c, min(1.0, p_final + POW_ROBUST))),
               p_lo=p_lo, p_hi=p_hi, pocket=int(sim["pocket"][1]), shot_type=P.shot_type(sim["info"]),
               cue_final=[float(v) for v in sim["final_xy"][0]], n_potted_obj=int(sim["info"]["n_obj_potted"]),
               t_rest=float(sim["t_rest"]))
    return out
