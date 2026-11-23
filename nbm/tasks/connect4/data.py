"""Connect Four data: self-play games labelled by the exact solver, positions deduplicated up to mirror symmetry, four splits.

    bash nbm/tasks/connect4/pons/build.sh                                   # the solver (once)
    python -m nbm.tasks.connect4.data --out $NBM_DATA/connect4 --workers 32   # deterministic; ~200k labelled positions

Games: game seed s -> random.Random(s). Each player draws its own noise level eps from EPS_LEVELS (uniform, independent). Each
move: with probability eps a uniformly random legal column (a realistic mistake); otherwise with probability 1/2 a uniformly
random exact-best column and else a uniformly random value-preserving column (diversity among good moves). Positions = every
non-terminal position of a game with at least 2 legal columns.

Splits, built in this order, never sharing a position or its mirror image (key = game.canon):
  train      the first N distinct CRITICAL positions (not every legal column has the same game value) of the games with seeds
             1,000,000, 1,000,001, ... in order; non-critical positions are skipped (their keys still count as taken).
  val, test  one critical position per game (seeds 2,000,000+ / 3,000,000+), drawn with random.Random(7 * seed + 1) among
             the game's critical positions whose key is not taken yet; games without one are skipped.
  test_hard  the same from seeds 4,000,000+, restricted to HARD positions: critical, no immediate win for the mover and no
             immediate winning threat of the opponent (the best move needs lookahead).
Row columns: id, split, game_seed, game_ply, eps_x, eps_o, board, ply, mover, scores (7 ints, -1000 = full column), value
(+1 / 0 / -1 for the mover), best (exact-best columns), choice (training label: game.canonical_best), vp (value-preserving
columns), critical, kind (win / block / quiet), hard, n_legal, line (the video's ground-truth continuation:
solver.best_line, up to 8 plies), line_len, line_end (win_x / win_o / draw / open).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from nbm.progress import Progress
from nbm.tasks.connect4 import game as G
from nbm.tasks.connect4 import solver as S
from nbm.tasks.connect4.render import MAX_PLIES

EPS_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
SIZES = {"train": 200_000, "val": 1_000, "test": 2_000, "test_hard": 1_000}
SEED0 = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000, "test_hard": 4_000_000}
SPLITS = ("train", "val", "test", "test_hard")


def play_game(seed: int) -> tuple[list[tuple[str, tuple]], dict]:
    """Every non-terminal (board, scores) of the self-play game of `seed`, and the players' noise levels."""
    rng = random.Random(seed)
    eps = {"x": rng.choice(EPS_LEVELS), "o": rng.choice(EPS_LEVELS)}
    slv = S.solver()
    b, out = G.EMPTY, []
    while not G.terminal(b):
        sc = slv.scores(b)
        out.append((b, sc))
        lg = G.legal(b)
        if rng.random() < eps[G.mover(b)]:
            c = rng.choice(lg)
        elif rng.random() < 0.5:
            c = rng.choice(G.best_set(sc))
        else:
            c = rng.choice(G.vp_set(sc))
        b = G.play(b, c)
    return out, eps


def game_positions(seed: int) -> tuple[int, dict, list[tuple[str, tuple]]]:
    """(seed, eps, positions with at least 2 legal columns): one worker task."""
    pos, eps = play_game(seed)
    return seed, eps, [(b, sc) for b, sc in pos if len(G.legal(b)) >= 2]


def eval_candidates(split: str, positions: list[tuple[str, tuple]], taken: set) -> list[tuple[int, str, tuple]]:
    """(game ply, board, scores) of a game's positions eligible for an evaluation split."""
    out = []
    for k, (b, sc) in enumerate(positions):
        if G.canon(b) in taken or not G.critical(sc):
            continue
        if split == "test_hard" and G.kind_of(b) != "quiet":
            continue
        out.append((k, b, sc))
    return out


def pick_eval_position(split: str, seed: int, positions: list[tuple[str, tuple]], taken: set):
    """The position an evaluation split takes from game `seed` ((ply, board, scores), or None)."""
    cand = eval_candidates(split, positions, taken)
    return random.Random(seed * 7 + 1).choice(cand) if cand else None


def label(b: str, sc: tuple, slv: S.Solver | None = None) -> dict:
    """The solver labels of one position (row columns from board to line_end)."""
    line = S.best_line(b, MAX_PLIES, slv)
    end = b
    for c in line:
        end = G.play(end, c)
    w = G.winner(end)
    kind = G.kind_of(b)
    crit = G.critical(sc)
    return {"board": b, "ply": G.plies(b), "mover": G.mover(b), "scores": [S.FULL if s is None else s for s in sc],
            "value": G.value(sc), "best": G.best_set(sc), "choice": G.canonical_best(sc), "vp": G.vp_set(sc), "critical": crit,
            "kind": kind, "hard": bool(crit and kind == "quiet"), "n_legal": len(G.legal(b)), "line": line,
            "line_len": len(line), "line_end": f"win_{w}" if w else "draw" if "." not in end else "open"}


def _label_job(item: tuple[dict, str, list[int]]) -> dict:
    meta, b, sc = item
    return {**meta, **label(b, tuple(None if s == S.FULL else s for s in sc))}


def select(sizes: dict[str, int], pool: Pool, chunk: int = 2000) -> dict[str, list]:
    """Play games split by split and pick the positions: {split: [(meta, board, scores)]}."""
    taken: set = set()
    picked: dict[str, list] = {s: [] for s in SPLITS}
    for split in SPLITS:
        need = sizes[split]
        prog = Progress(total=need, desc=f"connect4 data {split}", unit="row")
        seed = SEED0[split]
        while len(picked[split]) < need:
            left = need - len(picked[split])  # chunks sized to the rows still needed (any chunk size gives the same rows)
            n = min(chunk, 64 + (left // 5 if split == "train" else int(left * 1.15)))
            seeds = list(range(seed, seed + n))
            seed += n
            for gs, eps, pos in pool.imap(game_positions, seeds, chunksize=8):
                if len(picked[split]) >= need:
                    break
                meta = {"game_seed": gs, "eps_x": eps["x"], "eps_o": eps["o"]}
                if split == "train":
                    for k, (b, sc) in enumerate(pos):
                        if len(picked[split]) >= need:
                            break
                        key = G.canon(b)
                        if key in taken:
                            continue
                        taken.add(key)
                        if G.critical(sc):
                            picked[split].append(({**meta, "game_ply": k}, b, sc))
                else:
                    pick = pick_eval_position(split, gs, pos, taken)
                    if pick is None:
                        continue
                    k, b, sc = pick
                    taken.add(G.canon(b))
                    picked[split].append(({**meta, "game_ply": k}, b, sc))
                prog.update(len(picked[split]) - prog.n)
        prog.close()
    return picked


COLUMNS = ("id", "split", "game_seed", "game_ply", "eps_x", "eps_o", "board", "ply", "mover", "scores", "value", "best",
           "choice", "vp", "critical", "kind", "hard", "n_legal", "line", "line_len", "line_end")


def build(out: str | Path, sizes: dict[str, int] | None = None, workers: int = 32) -> dict[str, str]:
    """Write <out>/<split>.parquet for every split and digest.json; returns the digests."""
    sizes = {**SIZES, **(sizes or {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    with Pool(workers) as pool:
        picked = select(sizes, pool)
        for split in SPLITS:
            items = [(m, b, [S.FULL if s is None else s for s in sc]) for m, b, sc in picked[split]]
            prog = Progress(total=len(items), desc=f"connect4 labels {split}", unit="row")
            rows = []
            for r in pool.imap(_label_job, items, chunksize=64):
                rows.append(r)
                prog.update(1)
            prog.close()
            df = pd.DataFrame(rows)
            df.insert(0, "id", [f"c4_{split}_{i:06d}" for i in range(len(df))])
            df.insert(1, "split", split)
            df = df[list(COLUMNS)]
            df.to_parquet(out / f"{split}.parquet", index=False)
            digests[split] = table_digest(df)
            print(f"{split}: {len(df)} rows -> {out / f'{split}.parquet'}", flush=True)
    (out / "digest.json").write_text(json.dumps(digests, indent=2))
    return digests


def table_digest(df: pd.DataFrame) -> str:
    """sha256 over (board, choice, line) of every row: compare a rebuild with the released digest.json."""
    h = hashlib.sha256()
    for b, ch, ln in zip(df.board, df.choice, df.line):
        h.update(f"{b}|{ch}|{','.join(str(int(x)) for x in ln)};".encode())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build the Connect Four splits (nbm/tasks/connect4/data.py docstring).")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    for s in SPLITS:
        p.add_argument(f"--n_{s}", type=int, default=SIZES[s], help=f"rows of {s} (default {SIZES[s]})")
    a = p.parse_args(argv)
    digests = build(a.out, {s: getattr(a, f"n_{s}") for s in SPLITS}, a.workers)
    print(json.dumps(digests, indent=2))


if __name__ == "__main__":
    main()
