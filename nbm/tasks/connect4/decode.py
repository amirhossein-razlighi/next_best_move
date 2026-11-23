"""Reading Connect Four pictures back: board state, turn marker, and the marked move (no learned part).

Cell state = the nearest of (empty, Red, Blue) to the mean colour of the cell's inner disc (radius <= 8). Mark strength of a
cell = the least-squares alpha of GREEN on the cell's square pixels outside the hole (distance from the centre >= 14, 1 px in
from the cell border) against the reference picture (the position the move is made from).

Threshold-free readout (the paper's): the cell with the strongest mark is the move, whatever its strength (it only has to be
positive). The move is valid iff its column is not full AND the marked cell is that column's landing row: a disc drawn in the
wrong row is an impossible move and counts as wrong. The rules are never given to the model; they are only used to score.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from nbm import readout
from nbm.tasks.connect4 import game as G
from nbm.tasks.connect4 import render as R

_PROTOS = np.stack([R.BG, R.RED, R.BLUE]).astype(np.float32)
_STATES = ".xo"
LINE_MARK_THRESHOLD = 0.5  # the imagined continuation stops at the first later DECIDE block whose strongest mark is this faint


def _flat(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.shape != (R.SIZE, R.SIZE, 3):
        raise ValueError(f"expected a {R.SIZE} x {R.SIZE} RGB picture, got {a.shape}")
    return a.reshape(-1, 3)


def read_cells(img: np.ndarray) -> str:
    """The 42-character board shown in a picture (no gravity / count check)."""
    ix, seg, n = R.masks()["read_seg"]
    a = _flat(img)[ix]
    mean = np.stack([np.bincount(seg, weights=a[:, ch], minlength=G.W * G.H) for ch in range(3)], 1) / n[:, None]
    d = np.linalg.norm(mean[:, None, :] - _PROTOS[None], axis=2)
    return "".join(_STATES[k] for k in np.argsort(d, axis=1)[:, 0])


def read_turn(img: np.ndarray) -> str:
    """The turn marker's state ('x', 'o', or '.' = none)."""
    m = _flat(img)[R.masks()["turn_read"]].mean(0)
    return _STATES[int(np.argmin(np.linalg.norm(_PROTOS - m, axis=1)))]


def mark_strengths(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(7, 6) least-squares alpha of GREEN on each cell's tint pixels of img against ref, indexed [col, row]."""
    ix, seg, _ = R.masks()["tint_read_seg"]
    a, f = _flat(img)[ix], _flat(ref)[ix]
    d = R.GREEN.astype(np.float32)[None] - f
    num = np.bincount(seg, weights=((a - f) * d).sum(1), minlength=G.W * G.H)
    den = np.bincount(seg, weights=(d * d).sum(1), minlength=G.W * G.H)
    return (num / np.maximum(den, 1e-6)).astype(np.float32).reshape(G.W, G.H)


def read_move(img: np.ndarray, b: str, min_strength: float = 0.0) -> dict:
    """The move marked in a DECIDE picture made from position b: the cell with the strongest mark (first in (col, row) order
    on ties). Returns col, row, strength, drawn (strength > min_strength; the paper's readout uses 0), legal (column not
    full), landing_ok (row = the column's landing row) and valid (drawn, legal and landing_ok)."""
    al = mark_strengths(img, R.render(b))
    k = readout.strongest(al.ravel())
    col, row = divmod(k, G.H)
    strength = float(al[col, row])
    drawn = strength > min_strength
    legal = drawn and col in G.legal(b)
    landing_ok = drawn and G.landing(b, col) == row
    return {"col": col, "row": row, "strength": strength, "drawn": bool(drawn), "legal": bool(legal),
            "landing_ok": bool(landing_ok), "valid": bool(drawn and legal and landing_ok)}


def read_line(blocks: Sequence[np.ndarray], b: str, first: dict) -> dict:
    """The continuation a clip imagines after its first move `first` (read_move of blocks[1]), checked block by block
    against a bookkept board (the position + the model's own moves): each IMAGINE block must show the board after the move
    (and the next mover's turn marker), each later DECIDE block must mark a legal move in its landing row (strongest mark
    above LINE_MARK_THRESHOLD), and after the end of the game the final board must be held. The first failure ends the line.
    Returns moves (the columns read), plies, stop (why the reading ended), imagine_ok / later_legal (rates over the
    checked blocks, None if none), n_later (later DECIDE blocks checked) and end_hold_ok (only when the game ended)."""
    moves, imagine_ok, later_legal = [], [], []
    out: dict = {}
    stop = "max_plies"
    cur, k = b, 1
    while k < len(blocks):
        if G.terminal(cur):
            out["end_hold_ok"] = all(read_cells(x) == cur for x in blocks[k:])
            stop = "game_over"
            break
        if k == 1:
            mv = first
            if not mv["valid"]:
                stop = "first_illegal" if mv["drawn"] else "first_undrawn"
                break
        else:
            mv = read_move(blocks[k], cur, LINE_MARK_THRESHOLD)
            if not mv["drawn"]:
                stop = "undrawn"
                break
            ok = mv["legal"] and mv["landing_ok"]
            later_legal.append(ok)
            if not ok:
                stop = "illegal"
                break
        nxt = G.play(cur, mv["col"])
        moves.append(mv["col"])
        if k + 1 >= len(blocks):
            break
        turn = "." if G.terminal(nxt) else G.mover(nxt)
        ok = read_cells(blocks[k + 1]) == nxt and read_turn(blocks[k + 1]) == turn
        imagine_ok.append(ok)
        if not ok:
            stop = "imagine_mismatch"
            break
        cur = nxt
        k += 2
    mean = lambda v: float(np.mean(v)) if v else None  # noqa: E731
    return {"moves": moves, "plies": len(moves), "stop": stop, "imagine_ok": mean(imagine_ok), "later_legal": mean(later_legal),
            "n_later": len(later_legal), **out}
