"""Go rules on the 19 x 19 board: coordinates, SGF game records, replay with captures / suicide / simple ko, legal moves.

Conventions (shared by the data files, the pictures and the readout):
  * a point (x, y): x = column from the LEFT, y = row from the TOP (as SGF letters, 'a' = 0); its index is y * 19 + x.
  * GTP text: column letter without I, then 19 - y, e.g. (3, 15) -> "D4"; "pass" for a pass.
  * a position row: black / white = sorted index lists of the stones, side = "black" | "white" (to move), ko = index of the
    point the side to move may not play because of the simple-ko rule (-1 if none), correct = [index of the KataGo move].
The picture shows only the stones, so a position's legal moves are computed from the stones and the ko point (no superko).
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

import numpy as np

N = 19
COLS = "ABCDEFGHJKLMNOPQRST"  # GTP columns (no I)
EMPTY, BLACK, WHITE = 0, 1, 2
# game phase of a position = the number of moves already played (the strata of the evaluation)
PHASES = (("p20_59", 20, 60), ("p60_119", 60, 120), ("p120_179", 120, 180), ("p180_", 180, 10**6))

Point = tuple[int, int]


# ---------------------------------------------------------------- coordinates
def gtp(pt: Point | None) -> str:
    if pt is None:
        return "pass"
    x, y = pt
    return f"{COLS[x]}{N - y}"


def from_gtp(s: str) -> Point | None:
    s = s.strip().upper()
    if s in ("PASS", ""):
        return None
    return COLS.index(s[0]), N - int(s[1:])


def idx_of(pt: Point) -> int:
    return pt[1] * N + pt[0]


def pt_of(i: int) -> Point:
    return int(i) % N, int(i) // N


def phase_of(move_number: int) -> str:
    return next(name for name, lo, hi in PHASES if lo <= move_number < hi)


# ---------------------------------------------------------------- SGF
_PROP = re.compile(r"([A-Z]+)((?:\s*\[(?:\\.|[^\]\\])*\])+)", re.S)
_VAL = re.compile(r"\[((?:\\.|[^\]\\])*)\]", re.S)


def parse_sgf(text: str) -> tuple[dict, list[dict]]:
    """SGF game without variations -> (root properties {ID: [values]}, [node properties]). A record with a second game tree or
    a variation raises ValueError (the KGS archive has none)."""
    nodes, cur, j, trees = [], None, 0, 0
    while j < len(text):
        c = text[j]
        if c == "(":
            trees += 1
            if trees > 1:
                raise ValueError("variations / several game trees")
            j += 1
        elif c == ")":
            j += 1
        elif c == ";":
            cur = {}
            nodes.append(cur)
            j += 1
        elif "A" <= c <= "Z":
            m = _PROP.match(text, j)
            if not m or cur is None:
                raise ValueError(f"bad property at {j}")
            cur.setdefault(m.group(1), []).extend(_VAL.findall(m.group(2)))
            j = m.end()
        else:
            j += 1
    if not nodes:
        raise ValueError("empty game")
    return nodes[0], nodes[1:]


def sgf_pt(v: str) -> Point | None:
    """SGF point 'pd' -> (x, y); '' or 'tt' (the 19 x 19 pass) -> None."""
    v = v.strip()
    if v == "" or v == "tt":
        return None
    x, y = ord(v[0]) - 97, ord(v[1]) - 97
    if not (0 <= x < N and 0 <= y < N):
        raise ValueError(f"off-board point {v!r}")
    return x, y


# ---------------------------------------------------------------- board
class IllegalMove(Exception):
    pass


def _nbrs(x: int, y: int):
    if x > 0:
        yield x - 1, y
    if x < N - 1:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y < N - 1:
        yield x, y + 1


class Board:
    """19 x 19 board with captures, the suicide rule and simple ko. g[y, x] in {EMPTY, BLACK, WHITE}."""

    def __init__(self):
        self.g = np.zeros((N, N), dtype=np.int8)
        self.ko: Point | None = None  # the point the side to move may not play (simple ko)

    def copy(self) -> "Board":
        b = Board()
        b.g, b.ko = self.g.copy(), self.ko
        return b

    def group(self, x: int, y: int) -> tuple[set, set]:
        """(stones, liberties) of the chain at (x, y)."""
        c = self.g[y, x]
        stones, libs, stack = {(x, y)}, set(), [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for nx, ny in _nbrs(cx, cy):
                v = self.g[ny, nx]
                if v == EMPTY:
                    libs.add((nx, ny))
                elif v == c and (nx, ny) not in stones:
                    stones.add((nx, ny))
                    stack.append((nx, ny))
        return stones, libs

    def play(self, colour: int, pt: Point | None) -> int:
        """Play pt (None = pass) for colour; raises IllegalMove (occupied, ko, suicide). Returns the number of captured stones."""
        if pt is None:
            self.ko = None
            return 0
        x, y = pt
        if self.g[y, x] != EMPTY:
            raise IllegalMove("occupied")
        if self.ko == (x, y):
            raise IllegalMove("ko")
        opp = WHITE if colour == BLACK else BLACK
        self.g[y, x] = colour
        captured = []
        for nx, ny in _nbrs(x, y):
            if self.g[ny, nx] == opp:
                st, lib = self.group(nx, ny)
                if not lib:
                    captured += st
                    for sx, sy in st:
                        self.g[sy, sx] = EMPTY
        st, lib = self.group(x, y)
        if not lib:
            self.g[y, x] = EMPTY
            raise IllegalMove("suicide")
        self.ko = captured[0] if len(captured) == 1 and len(st) == 1 and len(lib) == 1 else None
        return len(captured)

    def legal(self, colour: int) -> np.ndarray:
        """(19, 19) bool [y, x]: empty, not the ko point, not suicide (taking the last liberty of an opposing chain is legal)."""
        ok = np.zeros((N, N), dtype=bool)
        opp = WHITE if colour == BLACK else BLACK
        cache: dict = {}

        def libs_of(x, y):
            if (x, y) not in cache:
                st, lib = self.group(x, y)
                for s in st:
                    cache[s] = len(lib)
            return cache[(x, y)]

        for y in range(N):
            for x in range(N):
                if self.g[y, x] != EMPTY or self.ko == (x, y):
                    continue
                for nx, ny in _nbrs(x, y):
                    v = self.g[ny, nx]
                    if v == EMPTY or (v == colour and libs_of(nx, ny) >= 2) or (v == opp and libs_of(nx, ny) == 1):
                        ok[y, x] = True
                        break
        return ok


def colour_of(c: str) -> int:
    return BLACK if c.upper() == "B" else WHITE


def side_name(colour: int) -> str:
    return "black" if colour == BLACK else "white"


def replay(moves: Iterable[tuple[str, Point | None]]) -> list[tuple[Board, int]]:
    """moves [(colour "B" / "W", point or None)] -> [(board before move k, colour to move at k)] for k = 0 .. len(moves) (the
    last entry is the final position). Raises IllegalMove on a bad record."""
    b = Board()
    out = []
    to_move = BLACK
    for c, pt in moves:
        col = colour_of(c)
        out.append((b.copy(), col))
        b.play(col, pt)
        to_move = WHITE if col == BLACK else BLACK
    out.append((b.copy(), to_move))
    return out


def stones(g: np.ndarray) -> tuple[list[int], list[int]]:
    """(black index list, white index list) of a grid."""
    flat = g.reshape(-1)
    return np.flatnonzero(flat == BLACK).tolist(), np.flatnonzero(flat == WHITE).tolist()


def grid_from(black: Iterable[int], white: Iterable[int]) -> np.ndarray:
    """(19, 19) int8 grid [y, x] of a position's stones."""
    g = np.zeros(N * N, dtype=np.int8)
    g[np.asarray(list(black), dtype=int)] = BLACK
    g[np.asarray(list(white), dtype=int)] = WHITE
    return g.reshape(N, N)


def board_of_row(row: Mapping) -> Board:
    b = Board()
    b.g = grid_from(row["black"], row["white"])
    k = int(row["ko"])
    b.ko = pt_of(k) if k >= 0 else None
    return b


def legal_mask(row: Mapping) -> np.ndarray:
    """(19, 19) bool [y, x] of the legal points of the side to move of a position row (simple ko included)."""
    return board_of_row(row).legal(BLACK if row["side"] == "black" else WHITE)
