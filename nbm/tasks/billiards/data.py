"""Billiards data: random layouts labelled by the offline solver, split by seed range.

    python -m nbm.tasks.billiards.data --out $NBM_DATA/billiards [--workers 32] [--splits train val test test_hard test_mirror]
                                       [--limit N]

Everything is deterministic: the layout of seed s comes from numpy.random.default_rng(s) (solver.random_layout), and its label
from the solver, which runs the bit-reproducible C simulator. A layout is kept iff some shot on the solver grid succeeds (98% of
them). Splits:
  train        the first 200,000 kept layouts of the seeds 1,000,000, 1,000,001, ...
  val          the first   1,000 kept layouts of the seeds 2,000,000, ...
  test         the first   2,000 kept layouts of the seeds 3,000,000, ...
  test_hard    the first   1,000 HARD layouts (no direct pot on the solver grid, see solver.py) of the seeds 4,000,000, ...
  test_mirror  every test layout mirrored by one exact table symmetry (left-right, top-bottom or both, drawn per row with
               default_rng([row, 99])) and solved again; orig_row = its test row, variant = the symmetry
Positions are continuous, so no layout is shared between splits, also up to the table's symmetries. The solver simulates ~13,000
shots per layout, about 0.15 CPU-seconds: everything takes ~9 CPU-hours (~20 minutes on 32 cores). --limit N builds only the
first N rows of each split (a quick sample: they are exactly the first N rows of the full split).

Columns: id ("<split>_<seed>", test_mirror: "<test id>_<symmetry>"), seed, split, k_others (0-4 other object balls), xy (2n floats:
cue, red, the others; table widths), colors (n ints, render.BALL_COLORS), then the solver label (solver.solve: angle, power,
width, ...; pocket, shot_type, hard). Files: <out>/<split>.parquet and <out>/digest.json (a hash of every row's positions and
label shot per split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nbm.progress import Progress
from nbm.tasks.billiards import physics as P
from nbm.tasks.billiards import solver as S

SIZES = {"train": 200_000, "val": 1_000, "test": 2_000, "test_hard": 1_000}
FIRST_SEED = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000, "test_hard": 4_000_000}
SPLITS = ("train", "val", "test", "test_hard", "test_mirror")
MIRRORS = ("lr", "tb", "both")
CHUNK = 4000  # at most this many seeds labelled per round


def label_seed(item: tuple[int, str]) -> dict | None:
    """(seed, split) -> the row of that layout, or None if it is not kept (unsolvable, or not hard for test_hard)."""
    seed, split = item
    xy, cols = S.random_layout(np.random.default_rng(seed))
    lab = S.solve(xy)
    if lab is None or (split == "test_hard" and not lab["hard"]):
        return None
    row = {"id": f"{split}_{seed}", "seed": int(seed), "split": split, "k_others": len(cols) - 2,
           "xy": [float(v) for v in xy.reshape(-1)], "colors": [int(c) for c in cols]}
    row.update(lab)
    return row


def mirror_row(item: tuple[int, dict]) -> dict:
    """(test row index, test row) -> its mirrored, re-solved row (test_mirror)."""
    i, r = item
    how = MIRRORS[int(np.random.default_rng([i, 99]).integers(3))]
    xy = P.mirror_xy(np.asarray(r["xy"]).reshape(-1, 2), how)
    lab = S.solve(xy)
    if lab is None:
        raise AssertionError("a mirrored layout must stay solvable")
    row = {"id": f"{r['id']}_{how}", "split": "test_mirror", "seed": int(r["seed"]), "k_others": int(r["k_others"]),
           "xy": [float(v) for v in xy.reshape(-1)], "colors": [int(c) for c in r["colors"]], "orig_row": i, "variant": how}
    row.update(lab)
    return row


def build_split(split: str, pool: Pool, n: int) -> pd.DataFrame:
    """The first n kept rows of a seeded split."""
    rows: list[dict] = []
    nxt = FIRST_SEED[split]
    prog = Progress(total=n, desc=f"billiards {split}: label layouts", unit="row")
    while len(rows) < n:
        size = min(CHUNK, max(64, 2 * (n - len(rows))))  # rows stay in seed order, so the round size does not change them
        for r in pool.imap(label_seed, [(s, split) for s in range(nxt, nxt + size)], chunksize=16):
            if r is not None and len(rows) < n:
                rows.append(r)
                prog.update(1)
        nxt += size
    prog.close()
    return pd.DataFrame(rows)


def table_digest(df: pd.DataFrame) -> str:
    """sha256 over every row's positions and label shot (12 significant digits)."""
    h = hashlib.sha256()
    for xy, a, p in zip(df.xy, df.angle, df.power):
        h.update((",".join(f"{v:.12g}" for v in xy) + f"|{a:.12g}|{p:.12g};").encode())
    return h.hexdigest()


def load(data_dir: str | os.PathLike, split: str) -> pd.DataFrame:
    """A split as a DataFrame with xy as float lists and colors as int lists."""
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    for k in ("xy", "cue_final"):
        if k in df:
            df[k] = df[k].apply(lambda v: [float(x) for x in v])
    df["colors"] = df["colors"].apply(lambda v: [int(x) for x in v])
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the billiards splits (module docstring).")
    ap.add_argument("--out", required=True, help="output directory, e.g. $NBM_DATA/billiards")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows of every split (a quick sample)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.build()  # compile the simulator once, before the workers load it
    digest = json.load(open(out / "digest.json")) if (out / "digest.json").exists() else {}
    with Pool(a.workers) as pool:
        for split in a.splits:
            if split == "test_mirror":
                test = load(out, "test")
                items = list(enumerate(test.head(a.limit).to_dict("records") if a.limit else test.to_dict("records")))
                prog = Progress(total=len(items), desc="billiards test_mirror: mirror and re-solve the test layouts", unit="row")
                rows = []
                for r in pool.imap(mirror_row, items, chunksize=8):
                    rows.append(r)
                    prog.update(1)
                prog.close()
                df = pd.DataFrame(rows)
            else:
                df = build_split(split, pool, min(SIZES[split], a.limit or SIZES[split]))
            df.to_parquet(out / f"{split}.parquet", index=False)
            digest[split] = table_digest(load(out, split))
            print(f"billiards {split}: {len(df)} rows -> {out / f'{split}.parquet'}", flush=True)
    json.dump(digest, open(out / "digest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
