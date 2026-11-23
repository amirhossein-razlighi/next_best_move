"""Exact move scores from Pascal Pons' Connect Four solver (labels for the data builder; evaluation reads them from the data).

The solver (https://github.com/PascalPons/connect4, AGPL-3.0) and its 7 x 6 opening book are not part of this repository:
`bash nbm/tasks/connect4/pons/build.sh` clones the solver at the pinned commit, downloads the book and compiles the small batch
driver pons/c4board.cpp into $NBM_CONNECT4_SOLVER/bin/c4board (default ./tools/connect4). The driver reads one board per line
and prints the 7 move scores (Pons' convention, game.py; -1000 = full column).

One driver process is kept per Python process (`solver()`): a worker forked from a process that already started the driver
must not share its pipe, so solver() starts a fresh driver whenever the process id changed.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from nbm.tasks.connect4 import game as G

FULL = -1000  # the driver's (and the data files') score of a full column


def solver_dir() -> Path:
    return Path(os.environ.get("NBM_CONNECT4_SOLVER", "tools/connect4")).expanduser()


class Solver:
    """A persistent c4board process. scores(b) -> 7 scores (None = full column) for the side to move; results are cached."""

    def __init__(self, root: str | os.PathLike | None = None):
        root = Path(root) if root is not None else solver_dir()
        binary, book = root / "bin" / "c4board", root / "book" / "7x6.book"
        for f in (binary, book):
            if not f.exists():
                raise FileNotFoundError(f"{f} missing: run `bash nbm/tasks/connect4/pons/build.sh` (or set "
                                        "NBM_CONNECT4_SOLVER)")
        self.proc = subprocess.Popen([str(binary), str(book)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.pid = os.getpid()
        self.cache: dict[str, tuple[int | None, ...]] = {}

    def _ask(self, lines: list[str]) -> list[str]:
        if os.getpid() != self.pid:
            raise RuntimeError("Solver used from a forked process: call connect4.solver.solver() in each process")
        out = []
        with self.lock:
            for k in range(0, len(lines), 256):
                chunk = lines[k:k + 256]
                self.proc.stdin.write("".join(x + "\n" for x in chunk))
                self.proc.stdin.flush()
                out += [self.proc.stdout.readline().strip() for _ in chunk]
        return out

    def scores_many(self, boards: list[str]) -> list[tuple[int | None, ...]]:
        todo = [b for b in dict.fromkeys(boards) if b not in self.cache]
        for b in todo:
            if not G.valid(b) or "." not in b:
                raise ValueError(f"invalid or terminal board {b}")
        for b, line in zip(todo, self._ask([f"B {b}" for b in todo])):
            if line == "ERR":
                raise RuntimeError(f"solver error on {b}")
            self.cache[b] = tuple(None if int(s) == FULL else int(s) for s in line.split())
        if len(self.cache) > 2_000_000:
            self.cache.clear()
        return [self.cache[b] for b in boards]

    def scores(self, b: str) -> tuple[int | None, ...]:
        return self.scores_many([b])[0]

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


_SOLVER: Solver | None = None


def solver() -> Solver:
    """The solver of this process (started on first use, restarted in a forked child)."""
    global _SOLVER
    if _SOLVER is None or _SOLVER.pid != os.getpid():
        _SOLVER = Solver()
    return _SOLVER


def best_line(b: str, max_plies: int, slv: Solver | None = None) -> list[int]:
    """The ground-truth continuation of the video model: both sides play the canonical best column (game.canonical_best)
    until the game ends or max_plies."""
    slv = slv or solver()
    out, cur = [], b
    while len(out) < max_plies and not G.terminal(cur):
        c = G.canonical_best(slv.scores(cur))
        out.append(c)
        cur = G.play(cur, c)
    return out
