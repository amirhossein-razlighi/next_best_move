"""Connect Four pictures (256 px) and the video clip of a line of play.

Picture: light grey background; the board is a dark slate rectangle at x 16..240, y 48..240 with 7 x 6 cells of 32 px (2 x 2
image-model tokens, 2 x 2 video latent cells). Each cell is a hole (radius 12): empty = the background colour, or a Red / Blue
disc. Turn marker: a disc of the side to move's colour centred at (128, 24) in the top margin (none when the game is over).

DECIDE picture (the action mark M(x, a), the image model's target and the clip's DECIDE blocks): the mover's disc drawn in the
landing cell of the chosen column, and that cell's 32 x 32 square tinted GREEN outside the hole (green is not a disc colour).
The turn marker is unchanged. IMAGINE picture = the position after the move (turn marker = the next mover).

Clip (17 blocks, nbm.models.video): block 0 = the position, then a DECIDE and an IMAGINE block per ply along the line (both
sides alternating, up to 8 plies); once the line or the game ends the last picture is held.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from nbm.tasks.connect4 import game as G

SIZE = 256
CELL = 32
X0, Y0 = 16, 48
TURN_XY = (128, 24)
R_DISC = 12.0
R_TINT = 12.5  # the GREEN tint covers the cell square outside this radius
R_READ = 8.0  # the decoder reads a cell's state on its inner disc
R_TINT_READ = 14.0  # the decoder reads the tint on square pixels outside this radius, 1 px in from the cell border
BG = np.array([232, 232, 228], np.uint8)
BOARD = np.array([62, 68, 78], np.uint8)
RED = np.array([222, 34, 40], np.uint8)
BLUE = np.array([30, 96, 232], np.uint8)
GREEN = np.array([10, 225, 70], np.uint8)
PIECE_COLOUR = {".": BG, "x": RED, "o": BLUE}
N_BLOCKS = 17
MAX_PLIES = 8


def cell_box(col: int, row: int) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of a cell (row 0 = bottom)."""
    x0, y0 = X0 + CELL * col, Y0 + CELL * (G.H - 1 - row)
    return x0, y0, x0 + CELL, y0 + CELL


def _centre(col: int, row: int) -> tuple[float, float]:
    return X0 + CELL * col + CELL / 2 - 0.5, Y0 + CELL * (G.H - 1 - row) + CELL / 2 - 0.5


def _segments(lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(all indices, segment id of each, segment sizes) for per-cell means with np.bincount."""
    return (np.concatenate(lists), np.concatenate([np.full(len(x), k) for k, x in enumerate(lists)]),
            np.array([len(x) for x in lists], np.float32))


@lru_cache(maxsize=None)
def masks() -> dict:
    """Flat pixel indices (into a 256 x 256 picture), per cell index i = 6 * col + row: "disc", "tint" (cell square outside
    the hole), "read" (inner disc), "tint_read" (the decoder's tint pixels); the turn marker's "turn_disc" / "turn_read"; and
    the segmented forms "read_seg" / "tint_read_seg" of the decoder regions."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    n = G.W * G.H
    disc, tint, read, tint_read = [None] * n, [None] * n, [None] * n, [None] * n
    for c in range(G.W):
        for r in range(G.H):
            cx, cy = _centre(c, r)
            d = np.hypot(xx - cx, yy - cy)
            x0, y0, x1, y1 = cell_box(c, r)
            square = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            inner = (xx >= x0 + 1) & (xx < x1 - 1) & (yy >= y0 + 1) & (yy < y1 - 1)
            i = G.idx(c, r)
            disc[i] = np.flatnonzero(d <= R_DISC)
            tint[i] = np.flatnonzero(square & (d > R_TINT))
            read[i] = np.flatnonzero(d <= R_READ)
            tint_read[i] = np.flatnonzero(inner & (d >= R_TINT_READ))
    d = np.hypot(xx - (TURN_XY[0] - 0.5), yy - (TURN_XY[1] - 0.5))
    return {"disc": disc, "tint": tint, "read": read, "tint_read": tint_read, "turn_disc": np.flatnonzero(d <= R_DISC),
            "turn_read": np.flatnonzero(d <= R_READ), "read_seg": _segments(read), "tint_read_seg": _segments(tint_read)}


@lru_cache(maxsize=None)
def _empty_board() -> np.ndarray:
    im = np.empty((SIZE * SIZE, 3), np.uint8)
    im[:] = BG
    im.reshape(SIZE, SIZE, 3)[Y0:Y0 + G.H * CELL, X0:X0 + G.W * CELL] = BOARD
    for k in masks()["disc"]:
        im[k] = BG
    im.setflags(write=False)
    return im


def render(b: str, decide: int | None = None, turn: bool = True) -> np.ndarray:
    """uint8 (256, 256, 3) picture of board b. decide = a column: the DECIDE picture of that move (the mover's disc in the
    landing cell + the GREEN cell tint). turn: draw the turn marker (none once the game is over)."""
    m = masks()
    im = _empty_board().copy()
    for i, ch in enumerate(b):
        if ch != ".":
            im[m["disc"][i]] = PIECE_COLOUR[ch]
    if decide is not None:
        r = G.landing(b, decide)
        if r is None:
            raise ValueError(f"column {decide} is full")
        i = G.idx(decide, r)
        im[m["disc"][i]] = PIECE_COLOUR[G.mover(b)]
        im[m["tint"][i]] = GREEN
    if turn and not G.terminal(b):
        im[m["turn_disc"]] = PIECE_COLOUR[G.mover(b)]
    return im.reshape(SIZE, SIZE, 3)


def line_blocks(b: str, line: Sequence[int], n_blocks: int = N_BLOCKS) -> np.ndarray:
    """uint8 (n_blocks, 256, 256, 3) clip of `line` (columns, both sides alternating) from b: block 0 = b, then DECIDE /
    IMAGINE per ply, then the last picture held."""
    frames = [render(b)]
    cur = b
    for c in line:
        if len(frames) >= n_blocks:
            break
        if G.terminal(cur):
            raise ValueError("the line continues after the end of the game")
        frames.append(render(cur, decide=c))
        if len(frames) >= n_blocks:
            break
        cur = G.play(cur, c)
        frames.append(render(cur))
    while len(frames) < n_blocks:
        frames.append(frames[-1])
    return np.stack(frames[:n_blocks])
