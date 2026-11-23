"""Build the chess puzzle splits from the Lichess puzzle database (rows only; pictures are rendered on the fly).

    python -m nbm.tasks.chess.data --puzzles lichess_db_puzzle.csv.zst --out $NBM_DATA/chess [--workers 16]

The database (https://database.lichess.org/#puzzles, CC0) lists for each puzzle the FEN *before* the opponent's last move and
the moves: Moves[0] is that opponent move, Moves[1] the first move the solver must find, then the opponent's replies and the
solver's moves alternate. One row = one puzzle:
    fen    the position after Moves[0] (the side to move must find the next best move)
    side   "white" / "black", the side to move
    line   Moves[1:] cut at 8 plies (the clip of the video model); line[0] is the answer (the image model's target)
    rating, rating_bucket, themes, game_id, pos_key (board + side + castling + en passant), puzzle_id

Leakage control: every puzzle of one source game goes to one split (a hash of the game id: 3% of the games test, 2% val), and
positions are deduplicated on pos_key, a position seen in an evaluation split being dropped from the training split. Val (2,000)
and test (10,000) take equal numbers of puzzles from 8 rating buckets among puzzles with a well-established rating (rating
deviation <= 90, >= 200 plays); train keeps every other puzzle of the training games, shuffled (the models train on its first
3,000,000 rows). Everything is seeded (seed 0), so the same database file gives the same splits; the paper used the database
snapshot of 9 October 2026 (docs/tasks/chess.md lists its checksum).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import pandas as pd

RATING_BINS = [0, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 4000]
RATING_LABELS = ["<1000", "1000-1250", "1250-1500", "1500-1750", "1750-2000", "2000-2250", "2250-2500", "2500+"]
MAX_PLIES = 8
COLUMNS = ["puzzle_id", "fen", "side", "line", "rating", "rating_bucket", "themes", "game_id", "pos_key"]


def split_of(game_id: str, val_frac: float, test_frac: float) -> str:
    """Split of a source game: the first 8 hex digits of sha1(game id) as a fraction of 0xFFFFFFFF."""
    h = int(hashlib.sha1(game_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def to_puzzle(args: tuple[str, str]) -> tuple[str, str, str, list[str]]:
    """(database FEN, Moves) -> (fen to solve, pos_key, side to move, solution moves); raises if a move is illegal."""
    fen, moves = args
    board = chess.Board(fen)
    moves = moves.split()
    board.push_uci(moves[0])
    fen2 = board.fen()
    side = "white" if board.turn else "black"
    for u in moves[1:]:
        board.push_uci(u)
    return fen2, " ".join(fen2.split()[:4]), side, moves[1:]


def stratified(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """n // 8 puzzles of every rating bucket, shuffled."""
    per = n // len(RATING_LABELS)
    parts = []
    for lab in RATING_LABELS:
        g = df[df.rating_bucket == lab]
        parts.append(g.sample(n=min(per, len(g)), random_state=rng))
    return pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)


def build(puzzles: str, out: str | os.PathLike, val_frac: float = 0.02, test_frac: float = 0.03, n_val: int = 2000,
          n_test: int = 10000, max_rd: int = 90, min_plays: int = 200, seed: int = 0, workers: int = 4,
          nrows: int | None = None) -> dict:
    """Write <out>/{train,val,test}.parquet and stats.json; returns the stats. nrows: read only the first rows of the
    database (for tests)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(puzzles, compression="zstd" if str(puzzles).endswith(".zst") else None, nrows=nrows,
                     usecols=["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "NbPlays", "Themes", "GameUrl"])
    df["game_id"] = df.GameUrl.str.extract(r"lichess\.org/([A-Za-z0-9]{8})")[0].fillna(df.PuzzleId)
    with Pool(workers) as pool:
        ex = pool.map(to_puzzle, zip(df.FEN, df.Moves), chunksize=5000)
    df["fen"], df["pos_key"], df["side"], df["solution"] = map(list, zip(*ex))
    df = df.rename(columns={"PuzzleId": "puzzle_id", "Rating": "rating", "RatingDeviation": "rating_dev", "NbPlays": "n_plays",
                            "Themes": "themes"})
    df["rating_bucket"] = pd.cut(df.rating, RATING_BINS, labels=RATING_LABELS, right=False).astype(str)
    df["split"] = [split_of(g, val_frac, test_frac) for g in df.game_id]

    # one row per position; a position that occurs in several splits stays in the evaluation split
    df["prio"] = df.split.map({"test": 0, "val": 1, "train": 2})
    df = df.sort_values(["prio", "n_plays"], ascending=[True, False]).drop_duplicates("pos_key", keep="first")

    rng = np.random.RandomState(seed)
    reliable = (df.rating_dev <= max_rd) & (df.n_plays >= min_plays)
    val = stratified(df[(df.split == "val") & reliable], n_val, rng)
    test = stratified(df[(df.split == "test") & reliable], n_test, rng)
    train = df[df.split == "train"].sample(frac=1, random_state=rng).reset_index(drop=True)
    assert not (set(train.pos_key) & (set(val.pos_key) | set(test.pos_key)))
    assert not (set(train.game_id) & (set(val.game_id) | set(test.game_id)))
    assert not (set(val.game_id) & set(test.game_id))

    stats = {"source": str(puzzles), "seed": seed, "splits": {}}
    for name, d in (("train", train), ("val", val), ("test", test)):
        d = d.assign(line=[s[:MAX_PLIES] for s in d.solution])[COLUMNS].reset_index(drop=True)
        d.to_parquet(out / f"{name}.parquet", index=False)
        stats["splits"][name] = {"rows": len(d), "white_to_move": int((d.side == "white").sum()),
                                 "rating_buckets": d.rating_bucket.value_counts().reindex(RATING_LABELS).fillna(0).astype(int).to_dict()}
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--puzzles", required=True, help="lichess_db_puzzle.csv.zst")
    p.add_argument("--out", required=True, help="output directory, e.g. $NBM_DATA/chess")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(build(a.puzzles, a.out, seed=a.seed, workers=a.workers), indent=2))


if __name__ == "__main__":
    main()
