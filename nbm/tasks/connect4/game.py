"""Connect Four rules and the game-theoretic labels derived from the exact solver's move scores.

Board = a 42-character string, column-major, each column bottom -> top: index 6 * col + row (row 0 = bottom). '.' empty,
'x' = Red (moves first), 'o' = Blue (second player). The side to move follows from the counts (Red iff #x == #o). Columns 0..6
run left -> right.

Move scores (solver.py, Pascal Pons' convention, for the side to move; None = full column): > 0 win, 0 draw, < 0 loss; a win
with the mover's own k-th stone (counting all its stones on the board) scores 22 - k, a loss to the opponent's k-th stone
-(22 - k). So a larger score is a faster win or a slower loss.
"""
from __future__ import annotations

from typing import Sequence

W, H = 7, 6
EMPTY = "." * (W * H)
CENTRE_ORDER = (3, 2, 4, 1, 5, 0, 6)  # tie-break among equally good columns: the most central first
NAME = {"x": "Red", "o": "Blue"}

Scores = Sequence[int | None]


def idx(col: int, row: int) -> int:
    return 6 * col + row


def count(b: str) -> tuple[int, int]:
    return b.count("x"), b.count("o")


def mover(b: str) -> str:
    nx, no = count(b)
    return "x" if nx == no else "o"


def other(p: str) -> str:
    return "o" if p == "x" else "x"


def legal(b: str) -> list[int]:
    """Columns that are not full."""
    return [c for c in range(W) if b[6 * c + 5] == "."]


def landing(b: str, col: int) -> int | None:
    """Row a disc dropped into `col` lands on (None if the column is full)."""
    h = len(b[6 * col:6 * col + 6].rstrip("."))
    return h if h < H else None


def play(b: str, col: int, p: str | None = None) -> str:
    """The board after `p` (default: the side to move) drops a disc into `col`."""
    r = landing(b, col)
    if r is None:
        raise ValueError(f"column {col} is full")
    i = idx(col, r)
    return b[:i] + (p or mover(b)) + b[i + 1:]


def _lines() -> list[tuple[int, ...]]:
    out = []
    for c in range(W):
        for r in range(H):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= x < W and 0 <= y < H for x, y in cells):
                    out.append(tuple(idx(x, y) for x, y in cells))
    return out


LINES = _lines()  # the 69 lines of four
LINES_THROUGH = [[ln for ln in LINES if i in ln] for i in range(W * H)]


def has_four(b: str, p: str) -> bool:
    return any(all(b[i] == p for i in ln) for ln in LINES)


def winner(b: str) -> str | None:
    for p in "xo":
        if has_four(b, p):
            return p
    return None


def terminal(b: str) -> bool:
    return winner(b) is not None or "." not in b


def wins_at(b: str, col: int, p: str) -> bool:
    """Does p complete a four by dropping into `col`?"""
    r = landing(b, col)
    if r is None:
        return False
    i = idx(col, r)
    return any(all(j == i or b[j] == p for j in ln) for ln in LINES_THROUGH[i])


def immediate_wins(b: str, p: str | None = None) -> list[int]:
    p = p or mover(b)
    return [c for c in legal(b) if wins_at(b, c, p)]


def valid(b: str) -> bool:
    """Gravity, counts and no four: a non-terminal position some game can reach (reachability itself is not checked)."""
    if len(b) != W * H or set(b) - set(".xo"):
        return False
    for c in range(W):
        col = b[6 * c:6 * c + 6]
        if "." in col and col[col.index("."):].strip(".") != "":
            return False
    nx, no = count(b)
    return nx - no in (0, 1) and winner(b) is None


def mirror(b: str) -> str:
    """The left-right mirror image."""
    return "".join(b[6 * (W - 1 - c):6 * (W - 1 - c) + 6] for c in range(W))


def canon(b: str) -> str:
    """Key of a position up to the mirror symmetry (the data splits share no key)."""
    return min(b, mirror(b))


def plies(b: str) -> int:
    return W * H - b.count(".")


# ---------------------------------------------------------------- labels from the solver's move scores
def sign(s: int) -> int:
    return 0 if s == 0 else (1 if s > 0 else -1)


def value(scores: Scores) -> int:
    """Game-theoretic value of the position for the side to move: +1 win / 0 draw / -1 loss."""
    return sign(max(s for s in scores if s is not None))


def best_set(scores: Scores) -> list[int]:
    """The exact-best columns (maximal score: fastest win / slowest loss)."""
    m = max(s for s in scores if s is not None)
    return [c for c in range(W) if scores[c] == m]


def vp_set(scores: Scores) -> list[int]:
    """Value-preserving columns: the position stays won / drawn / lost after them."""
    v = value(scores)
    return [c for c in range(W) if scores[c] is not None and sign(scores[c]) == v]


def canonical_best(scores: Scores) -> int:
    """The training label: an exact-best column, the most central one on ties."""
    bs = set(best_set(scores))
    return next(c for c in CENTRE_ORDER if c in bs)


def critical(scores: Scores) -> bool:
    """Not every legal column has the same game-theoretic value (the only positions kept in the data)."""
    return len({sign(s) for s in scores if s is not None}) > 1


def kind_of(b: str) -> str:
    """'win' (the mover can win at once), 'block' (no win, the opponent threatens an immediate win), else 'quiet' (the best
    move needs lookahead; critical quiet positions form the hard test set)."""
    p = mover(b)
    if immediate_wins(b, p):
        return "win"
    if immediate_wins(b, other(p)):
        return "block"
    return "quiet"
