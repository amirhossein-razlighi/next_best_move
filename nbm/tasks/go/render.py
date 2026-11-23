"""Pictures of a Go position, the red-disc action mark and its readout.

Picture: the whole 19 x 19 board on wood, intersection (x, y) at the centre of cell [x s, (x + 1) s) x [y s, (y + 1) s) with
s = size / 19, thick edge lines, the nine star points, stones of radius 0.47 s; no coordinates, no move history. Drawn at 4x
the size and downsampled (Lanczos), so the stones are anti-aliased.

Action mark M(x, a): a RED disc of radius 0.38 s (smaller than a stone, so a disc on an empty point is clearly not a stone)
blended with alpha 0.9 over the cell of the move (nbm.marks.blend with the disc's anti-aliased coverage).

Readout (shared decoder D for a disc): per cell, the least-squares strength of the disc in (output - condition) on the inner
pixels (8% of a cell dropped at each edge), divided by alpha and clipped to [0, 1.5] (1 = a disc drawn at full strength).
The threshold-free decision is the strongest cell over all 361 points; whether it is a legal move is checked afterwards
(an illegal point is a wrong answer, it never makes the readout pick another point).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw

from nbm import marks, readout
from nbm.tasks.go.rules import BLACK, N, Point, grid_from

WOOD = (222, 184, 112)
LINE = (30, 25, 15)
STAR_POINTS = (3, 9, 15)
SUPERSAMPLE = 4
DISC_ALPHA = 0.9
DISC_RADIUS = 0.38  # of a cell side (stones: 0.47)
INNER_MARGIN = 0.08  # fraction of a cell ignored at each edge when decoding


def cell_box(pt: Point, size: int) -> marks.Box:
    """Pixel box (x0, y0, x1, y1) of the cell of point pt = (x, y) in a size x size picture."""
    s = size / N
    x, y = pt
    return int(round(x * s)), int(round(y * s)), int(round((x + 1) * s)), int(round((y + 1) * s))


def render_board(row: Mapping, size: int) -> Image.Image:
    """The position of a row (black / white stone index lists) as a size x size RGB picture."""
    g = grid_from(row["black"], row["white"])
    S = size * SUPERSAMPLE
    s = S / N
    img = Image.new("RGB", (S, S), WOOD)
    dr = ImageDraw.Draw(img)

    def c(i: int) -> float:
        return (i + 0.5) * s

    end = c(N - 1)
    thin, thick = max(1, round(0.045 * s)), max(2, round(0.11 * s))
    for i in range(N):
        t = thick if i == 0 or i == N - 1 else thin
        dr.line([(c(i), c(0) - t // 2), (c(i), end)], fill=LINE, width=t)
        dr.line([(c(0) - t // 2, c(i)), (end, c(i))], fill=LINE, width=t)
    for sx in STAR_POINTS:
        for sy in STAR_POINTS:
            r = 0.11 * s
            dr.ellipse([c(sx) - r, c(sy) - r, c(sx) + r, c(sy) + r], fill=LINE)
    r = 0.47 * s
    for y in range(N):
        for x in range(N):
            if g[y, x]:
                fill, edge = ((25, 25, 25), (0, 0, 0)) if g[y, x] == BLACK else ((245, 245, 240), (60, 60, 60))
                dr.ellipse([c(x) - r, c(y) - r, c(x) + r, c(y) + r], fill=fill, outline=edge, width=max(1, round(0.04 * s)))
    return img.resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=None)
def disc_mask(h: int, w: int) -> np.ndarray:
    """(h, w) anti-aliased coverage in [0, 1] of the disc centred in an h x w cell (4 x 4 supersampled)."""
    ss = 4
    y, x = (np.mgrid[0:h * ss, 0:w * ss] + 0.5) / ss
    inside = (x - w / 2) ** 2 + (y - h / 2) ** 2 <= (DISC_RADIUS * min(h, w)) ** 2
    return inside.reshape(h, ss, w, ss).mean(axis=(1, 3)).astype(np.float32)


def draw_move(cond: Image.Image, pt: Point) -> Image.Image:
    """The condition picture with the red disc on point pt (the training target)."""
    size = cond.size[0]
    x0, y0, x1, y1 = cell_box(pt, size)
    coverage = np.zeros((size, size), np.float32)
    coverage[y0:y1, x0:x1] = disc_mask(y1 - y0, x1 - x0)
    return marks.blend(cond, coverage, marks.RED, alpha=DISC_ALPHA)


def red_strengths(out: Image.Image, cond: Image.Image) -> np.ndarray:
    """(19, 19) [y, x] strength of the red disc per point (0 = nothing drawn, 1 = full target strength, clipped at 1.5)."""
    size = cond.size[0]
    o = np.asarray(out.convert("RGB").resize(cond.size, Image.BILINEAR), dtype=np.float32)
    c = np.asarray(cond.convert("RGB"), dtype=np.float32)
    red = np.asarray(marks.RED, dtype=np.float32)
    a = np.zeros((N, N))
    m = int(size / N * INNER_MARGIN)
    for y in range(N):
        for x in range(N):
            x0, y0, x1, y1 = cell_box((x, y), size)
            cc = c[y0 + m:y1 - m, x0 + m:x1 - m].reshape(-1, 3)
            d = (o[y0 + m:y1 - m, x0 + m:x1 - m].reshape(-1, 3) - cc).reshape(-1)
            disc = disc_mask(y1 - y0, x1 - x0)[m:y1 - y0 - m, m:x1 - x0 - m].reshape(-1, 1)
            A = (disc * (red - cc)).reshape(-1)
            a[y, x] = float(A @ d / (A @ A + 1e-6))
    return np.clip(a / DISC_ALPHA, 0, marks.MAX_STRENGTH)


def read_move(out: Image.Image, cond: Image.Image) -> tuple[Point, float]:
    """Threshold-free readout: (the point with the strongest red disc (first in row-major order on ties), its strength)."""
    a = red_strengths(out, cond)
    k = readout.strongest(a.reshape(-1))
    return (k % N, k // N), float(a.reshape(-1)[k])
