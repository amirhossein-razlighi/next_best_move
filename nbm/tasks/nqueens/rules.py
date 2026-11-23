"""N-queens rules: partial placements, the exact solver, completions, symmetry classes, and the scoring of a placement.

A state is a string of N characters (N <= 12): s[r] = "." if row r is empty, otherwise the base-36 digit of the queen's
column ("0".."9", "a", "b"). A state is a partial placement of k pairwise non-attacking queens (0 <= k < N). The action is
one more queen on a cell (r, c). A placement is correct iff the new state still extends to at least one full solution
("extendable"): the correct cells are exactly the cells, outside the state's rows, of the solutions that contain the state.

Solutions are enumerated by bitmask backtracking (N = 12: 14,200 solutions in about a second); the counts are checked
against the known sequence. The symmetry class of a solution is the lexicographically smallest of its 8 images under the
dihedral group of the square.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

DIGITS = "0123456789ab"
KNOWN_COUNTS = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200}
KNOWN_CLASSES = {4: 1, 5: 2, 6: 1, 7: 6, 8: 12, 9: 46, 10: 92, 11: 341, 12: 1787}
CATEGORIES = ("correct", "dead", "attacked", "occupied", "off_board")

Cell = tuple[int, int]


# ------------------------------------------------------------------ states
def empty(n: int) -> str:
    return "." * n


def cols_of(s: str) -> list[int]:
    """Column of the queen of every row, -1 for an empty row."""
    return [-1 if ch == "." else DIGITS.index(ch) for ch in s]


def from_cols(cols) -> str:
    return "".join("." if c < 0 else DIGITS[c] for c in cols)


def queens(s: str) -> list[Cell]:
    return [(r, c) for r, c in enumerate(cols_of(s)) if c >= 0]


def from_queens(n: int, qs) -> str:
    cols = [-1] * n
    for r, c in qs:
        assert cols[r] < 0, (n, qs)
        cols[r] = c
    return from_cols(cols)


def k_of(s: str) -> int:
    """Number of queens on the board."""
    return sum(ch != "." for ch in s)


def place(s: str, r: int, c: int) -> str:
    assert s[r] == ".", (s, r, c)
    return s[:r] + DIGITS[c] + s[r + 1:]


def attacks(r1: int, c1: int, r2: int, c2: int) -> bool:
    return r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)


def is_partial(s: str) -> bool:
    """True iff the queens of s are pairwise non-attacking."""
    return all(not attacks(*a, *b) for a, b in itertools.combinations(queens(s), 2))


def safe_cells(s: str) -> list[Cell]:
    """Empty on-board cells not attacked by any queen of s (row-major order)."""
    n, q = len(s), queens(s)
    used_c = {c for _, c in q}
    d1 = {r - c for r, c in q}
    d2 = {r + c for r, c in q}
    return [(r, c) for r in range(n) if s[r] == "." for c in range(n)
            if c not in used_c and (r - c) not in d1 and (r + c) not in d2]


def top_row(s: str) -> int | None:
    """The topmost empty row (the row the training label fills)."""
    return s.index(".") if "." in s else None


# ------------------------------------------------------------------ solver
def enumerate_solutions(n: int) -> list[str]:
    """All solutions of n as strings (row r -> column), in lexicographic order."""
    out, full = [], (1 << n) - 1
    cols = [0] * n

    def rec(r, cm, d1, d2):
        if r == n:
            out.append(from_cols(cols))
            return
        free = full & ~(cm | d1 | d2)
        while free:
            b = free & -free
            free ^= b
            cols[r] = b.bit_length() - 1
            rec(r + 1, cm | b, ((d1 | b) << 1) & full, (d2 | b) >> 1)
    rec(0, 0, 0, 0)
    return out


def count_completions_bitmask(s: str) -> int:
    """Independent completion counter (used by the checks): the number of full solutions containing the queens of s, by
    backtracking over the rows with the pre-placed queens forced."""
    n, pre = len(s), cols_of(s)
    if not is_partial(s):
        return 0

    def rec(r, cm, d1, d2):
        if r == n:
            return 1
        tot = 0
        for c in ([pre[r]] if pre[r] >= 0 else range(n)):
            if cm >> c & 1 or d1 >> (r - c + n) & 1 or d2 >> (r + c) & 1:
                continue
            tot += rec(r + 1, cm | 1 << c, d1 | 1 << (r - c + n), d2 | 1 << (r + c))
        return tot
    return rec(0, 0, 0, 0)


@lru_cache(maxsize=None)
def solutions(n: int) -> tuple[tuple[str, ...], np.ndarray]:
    """(solution strings, int8 array (M, n) of their columns), counts checked against the known sequence."""
    sols = enumerate_solutions(n)
    assert len(sols) == KNOWN_COUNTS[n], (n, len(sols))
    arr = np.array([cols_of(x) for x in sols], np.int8).reshape(len(sols), n)
    arr.setflags(write=False)
    return tuple(sols), arr


def completion_mask(s: str) -> np.ndarray:
    """bool (M,) over solutions(len(s)): the solutions that contain every queen of s."""
    _, arr = solutions(len(s))
    q = queens(s)
    if not q:
        return np.ones(len(arr), bool)
    rs = np.array([r for r, _ in q])
    cs = np.array([c for _, c in q], np.int8)
    return (arr[:, rs] == cs).all(1)


def n_completions(s: str) -> int:
    return int(completion_mask(s).sum())


def extendable_cells(s: str) -> list[Cell]:
    """Every correct next placement: the empty-row cells of the solutions that contain s (sorted)."""
    n = len(s)
    _, arr = solutions(n)
    comp = arr[completion_mask(s)]
    out = set()
    for r in range(n):
        if s[r] == ".":
            out.update((r, int(c)) for c in np.unique(comp[:, r]))
    return sorted(out)


def is_extendable(s: str, r: int, c: int) -> bool:
    if not (0 <= r < len(s) and 0 <= c < len(s)) or s[r] != ".":
        return False
    return bool(completion_mask(place(s, r, c)).any())


def classify(s: str, cell: Cell) -> str:
    """Category of placing a queen on canvas cell `cell` (any of the 12 x 12 canvas cells) in state s: correct (extendable)
    / dead (legal, no completion) / attacked / occupied / off_board. A second queen in a filled row counts as attacked."""
    n = len(s)
    r, c = cell
    if r >= n or c >= n:
        return "off_board"
    if s[r] != ".":
        return "occupied" if cols_of(s)[r] == c else "attacked"
    if any(attacks(r, c, qr, qc) for qr, qc in queens(s)):
        return "attacked"
    return "correct" if is_extendable(s, r, c) else "dead"


# ------------------------------------------------------------------ symmetry
SYMMETRIES = ("id", "r90", "r180", "r270", "fh", "fv", "tr", "atr")


def sym_cell(name: str, r: int, c: int, n: int) -> Cell:
    m = n - 1
    return {"id": (r, c), "r90": (c, m - r), "r180": (m - r, m - c), "r270": (m - c, r), "fh": (r, m - c), "fv": (m - r, c),
            "tr": (c, r), "atr": (m - c, m - r)}[name]


def sym_state(name: str, s: str) -> str:
    n = len(s)
    return from_queens(n, [sym_cell(name, r, c, n) for r, c in queens(s)])


def canon(sol: str) -> str:
    """Name of the symmetry class of a solution: the smallest of its 8 images."""
    return min(sym_state(t, sol) for t in SYMMETRIES)


@lru_cache(maxsize=None)
def classes(n: int) -> dict[str, list[int]]:
    """Symmetry classes of n: canonical solution -> indices into solutions(n)."""
    sols, _ = solutions(n)
    out: dict[str, list[int]] = {}
    for i, x in enumerate(sols):
        out.setdefault(canon(x), []).append(i)
    if n in KNOWN_CLASSES:
        assert len(out) == KNOWN_CLASSES[n], (n, len(out))
    return out
