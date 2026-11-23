"""Build the Go data: KGS game records -> positions -> KataGo labels -> test / val / train splits (by game).

    python -m nbm.tasks.go.build games     --kgs <KGS dir> --work <work>
    python -m nbm.tasks.go.build positions --work <work> --round eval
    python -m nbm.tasks.go.build value     --work <work> --round eval --net <net> [--shard 0] [--part j --parts m]   # GPU
    python -m nbm.tasks.go.build select    --work <work> --round eval
    python -m nbm.tasks.go.build label     --work <work> --round eval --net <net> [--shard 0] [--part j --parts m]   # GPU
    python -m nbm.tasks.go.build split     --work <work> --round eval  --out <data dir>      # -> test.parquet, val.parquet
    python -m nbm.tasks.go.build positions --work <work> --round train --out <data dir>     # needs val / test (their games are excluded)
    ... value / select / label for --round train (shards 0..31), then
    python -m nbm.tasks.go.build split     --work <work> --round train --out <data dir>     # -> train.parquet

games      every SGF of the archive (u-go.net KGS: one player 7d+ or both 6d), kept if: 19 x 19, year <= 2015 (before strong bots),
           no handicap / setup stones, >= 80 moves, no player name of a known bot, every move legal in our replay (captures,
           suicide, simple ko). The valid games get a seeded random order (permutation seed 0): game_rank.
positions  two rounds. eval: the first 80,000 games (game order) with >= 5 eligible plies, 5 plies each, drawn without
           replacement from [20, min(first pass, n_moves - 5)) (per-game RNG seeded (0, crc32(game))); val / test are cut from
           this round. train: every valid game except the val / test games: its eval-round plies plus new plies (per-game RNG
           seeded (1, crc32(game))) from the same range, up to 30 per game. A ply t = the position after t moves.
value      KataGo's raw network value (1 visit, with the game history) of every position (KataGo, one GPU).
select     positions whose value for the side to move is in [0.05, 0.95] (not yet decided), the first 125,000 (eval) /
           1,100,000 (train) in game order (game_rank, ply).
label      400-visit KataGo search of every selected position: label = the move with the most root visits (katago.py); the train
           round reuses the eval round's labels of the positions it shares with it.
split      keeps a labelled position iff KataGo answered, the label is not a pass and is legal under our rules (simple ko), and the
           search's root winrate is in [0.05, 0.95]. eval: in game order, test = the first games until 2,000 positions, val = the
           next until 1,000 (the last game of a split is truncated); a position whose stones + side to move occur in more than one
           game (a common opening) never goes to val / test. train: the kept positions in game order up to 1,000,000, without any
           board (stones + side) that occurs in val / test.

Outputs: <data dir>/{test,val,train}.parquet (columns: docs/tasks/go.md) and <step>.stats.json files in <work>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from nbm.progress import Progress
from nbm.tasks.go import katago as K
from nbm.tasks.go import rules as R

BOT_RE = re.compile(r"bot|^zen|zen19|petgo|^elf|leela|gnugo|pachi|mogo|crazy|fuego|^aya|golois|dolbaram|katago|manyfaces|abakus|"
                    r"darkforest|gotrend|golaxy|steenvreter|orego|nexus|^cgos|^lz", re.I)
MAX_YEAR, MIN_MOVES = 2015, 80
MIN_PLY, TAIL = 20, 5  # plies drawn from [MIN_PLY, min(first pass, n_moves - TAIL))
EVAL_GAMES, EVAL_PER_GAME, TRAIN_PER_GAME = 80_000, 5, 30
ROUND_SEED = {"eval": 0, "train": 1}
ROUND_SHARDS = {"eval": 1, "train": 32}
SELECT_N = {"eval": 125_000, "train": 1_100_000}
VALUE_LO, VALUE_HI = 0.05, 0.95
N_TEST, N_VAL, N_TRAIN = 2000, 1000, 1_000_000
META = ("year", "black_rank", "white_rank", "komi_game", "result", "ruleset")
TRAIN_COLUMNS = ["id", "game", "game_rank", "ply", "phase", "side", "black", "white", "ko", "correct", "best", "best_winrate",
                 "root_winrate", "legal_n", "from_eval_round"]


# ---------------------------------------------------------------- games
def year_of(path: Path, root: dict) -> int:
    m = re.match(r"(\d{4})", (root.get("DT") or [""])[0])
    if m:
        return int(m.group(1))
    m = re.search(r"(20\d\d)", path.parent.name)
    return int(m.group(1)) if m else -1


def load_game(job: tuple[str, str]) -> dict:
    """(path, name relative to the archive root) -> {game, keep, reason, metadata, moves ["B D4", ...] if kept}."""
    path, rel = job
    out = {"game": rel}
    try:
        root, nodes = R.parse_sgf(open(path, encoding="utf-8", errors="replace").read())
    except Exception as e:  # noqa: BLE001
        return {**out, "keep": False, "reason": f"parse:{type(e).__name__}"}

    def g1(k):
        return (root.get(k) or [""])[0]

    out.update(year=year_of(Path(path), root), player_black=g1("PB"), player_white=g1("PW"), black_rank=g1("BR"),
               white_rank=g1("WR"), komi_game=g1("KM"), result=g1("RE"), ruleset=g1("RU"))
    if g1("SZ") not in ("19", ""):
        return {**out, "keep": False, "reason": "size"}
    if out["year"] > MAX_YEAR:
        return {**out, "keep": False, "reason": "year"}
    if g1("HA") not in ("", "0", "1") or "AB" in root or "AW" in root or any("AB" in n or "AW" in n or "AE" in n for n in nodes):
        return {**out, "keep": False, "reason": "handicap_or_setup"}
    if BOT_RE.search(out["player_black"]) or BOT_RE.search(out["player_white"]):
        return {**out, "keep": False, "reason": "bot"}
    moves = []
    try:
        for n in nodes:
            for c in ("B", "W"):
                if c in n:
                    moves.append((c, R.sgf_pt(n[c][0])))
    except ValueError:
        return {**out, "keep": False, "reason": "bad_point"}
    if len(moves) < MIN_MOVES:
        return {**out, "keep": False, "reason": "short"}
    try:
        R.replay(moves)
    except R.IllegalMove as e:
        return {**out, "keep": False, "reason": f"illegal:{e}"}
    return {**out, "keep": True, "reason": "ok", "n_moves": len(moves), "moves": [f"{c} {R.gtp(p)}" for c, p in moves]}


def order_games(games: list[dict]) -> pd.DataFrame:
    """The valid games (games in sorted file order) in the seeded random game order; game_rank = position in that order."""
    kept = [g for g in games if g["keep"]]
    order = np.random.default_rng(0).permutation(len(kept))
    df = pd.DataFrame([kept[i] for i in order]).drop(columns=["keep", "reason"])
    df.insert(1, "game_rank", np.arange(len(df)))
    return df


def cmd_games(a: argparse.Namespace) -> None:
    kgs = Path(a.kgs)
    files = sorted(kgs.rglob("*.sgf"))
    jobs = [(str(f), f.relative_to(kgs).as_posix()) for f in files]
    with Pool(a.workers) as pool:
        games = list(pool.imap(load_game, jobs, chunksize=64))
    df = order_games(games)
    df.to_parquet(work(a) / "games.parquet")
    stats = {"files": len(files), "reasons": dict(Counter(g["reason"] for g in games)), "valid_games": len(df)}
    dump(stats, work(a) / "games.stats.json")


# ---------------------------------------------------------------- positions
def ply_range(moves: list[str]) -> tuple[int, int]:
    """[lo, hi) of the plies a position may be drawn from: after MIN_PLY moves, before the first pass and the last TAIL moves."""
    first_pass = next((k for k, m in enumerate(moves) if m.split()[1] == "pass"), len(moves))
    return MIN_PLY, min(first_pass, len(moves) - TAIL)


def eval_plies(game: str, moves: list[str]) -> list[int]:
    """The eval round's EVAL_PER_GAME plies of a game ([] if its range is shorter)."""
    lo, hi = ply_range(moves)
    if hi - lo < EVAL_PER_GAME:
        return []
    rng = np.random.default_rng([ROUND_SEED["eval"], zlib.crc32(game.encode())])
    return sorted(rng.choice(np.arange(lo, hi), size=EVAL_PER_GAME, replace=False).tolist())


def train_plies(game: str, moves: list[str], old: list[int]) -> list[int]:
    """The train round's plies of a game: its eval-round plies `old` plus new ones up to TRAIN_PER_GAME."""
    lo, hi = ply_range(moves)
    old = sorted(t for t in old if lo <= t < hi)
    pool = np.setdiff1d(np.arange(lo, hi), old)
    need = max(0, min(TRAIN_PER_GAME, hi - lo) - len(old))
    rng = np.random.default_rng([ROUND_SEED["train"], zlib.crc32(game.encode())])
    new = rng.choice(pool, size=need, replace=False).tolist() if need else []
    return sorted(set(old) | set(int(t) for t in new))


def position_rows(game: Mapping, plies: list[int]) -> list[dict]:
    """One row per ply: the stones, side to move, ko point, the move history (for KataGo) and the game's metadata."""
    moves = [(m.split()[0], R.from_gtp(m.split()[1])) for m in game["moves"]]
    states = R.replay(moves)
    rows = []
    for t in plies:
        b, col = states[t]
        black, white = R.stones(b.g)
        rows.append({"id": f"{game['game']}#{t}", "game": game["game"], "game_rank": int(game["game_rank"]), "ply": int(t),
                     "side": R.side_name(col), "black": black, "white": white, "ko": R.idx_of(b.ko) if b.ko else -1,
                     "moves": list(game["moves"][:t]), "legal_n": int(b.legal(col).sum()), "phase": R.phase_of(t),
                     "game_move": game["moves"][t], **{k: game[k] for k in META}})
    return rows


def _eval_worker(game: dict) -> list[dict]:
    return position_rows(game, eval_plies(game["game"], game["moves"]))


def _train_worker(job: tuple[dict, list[int]]) -> list[dict]:
    game, old = job
    rows = position_rows(game, train_plies(game["game"], game["moves"], old))
    for r in rows:
        r["from_eval_round"] = r["ply"] in old
    return rows


def eval_positions(games: pd.DataFrame, pool: Pool) -> pd.DataFrame:
    """The eval round: games in game order, the first EVAL_GAMES of them that have EVAL_PER_GAME eligible plies (drawn from the
    first EVAL_GAMES * 1.05 + 100 games: nearly every game has enough plies)."""
    cand = games.head(int(EVAL_GAMES * 1.05) + 100).to_dict("records")
    rows, used = [], 0
    for r in pool.map(_eval_worker, cand, chunksize=16):
        if r:
            rows += r
            used += 1
            if used == EVAL_GAMES:
                break
    if used < min(EVAL_GAMES, len(games)) and len(cand) < len(games):
        raise RuntimeError(f"only {used} of the first {len(cand)} games have {EVAL_PER_GAME} eligible plies")
    return pd.DataFrame(rows)


def cmd_positions(a: argparse.Namespace) -> None:
    games = pd.read_parquet(work(a) / "games.parquet")
    d = round_dir(a)
    shards = a.shards or ROUND_SHARDS[a.round]
    with Pool(a.workers) as pool:
        if a.round == "eval":
            df = eval_positions(games, pool)
            parts = {k: df[df.game_rank % shards == k] for k in range(shards)}
            stats = {"games": int(df.game.nunique()), "positions": len(df)}
        else:
            held = held_games(Path(a.out))
            old = pd.concat([pd.read_parquet(f, columns=["game", "ply"]) for f in sorted((work(a) / "eval").glob("positions_*.parquet"))])
            old_plies = old.groupby("game").ply.apply(list).to_dict()
            games = games[~games.game.isin(held)]
            parts, stats = {}, {"games": len(games), "heldout_games_excluded": len(held)}
            for k in range(shards):
                mine = games[games.game_rank % shards == k].to_dict("records")
                res = pool.imap(_train_worker, [(g, old_plies.get(g["game"], [])) for g in mine], chunksize=16)
                parts[k] = pd.DataFrame([x for r in res for x in r])
            stats["positions"] = int(sum(len(p) for p in parts.values()))
    for k, p in parts.items():
        p.reset_index(drop=True).to_parquet(d / f"positions_{k:02d}.parquet")
    dump(stats, d / "positions.stats.json")


# ---------------------------------------------------------------- KataGo: value, select, label
def read_jsonl(paths) -> list[dict]:
    out = []
    for f in paths:
        with open(f) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a partial last line after a kill
                    continue
    return out


def _katago_rows(a: argparse.Namespace, kind: str) -> tuple[list[dict], Path, set]:
    """Rows of positions / selected shard a.shard (part a.part of a.parts), the output file and the ids already done."""
    src = round_dir(a) / f"{'positions' if kind == 'values' else 'selected'}_{a.shard:02d}.parquet"
    df = pd.read_parquet(src, columns=["id", "moves"])
    df = df.iloc[[i for i in range(len(df)) if i % a.parts == a.part]]
    out = round_dir(a) / kind
    out.mkdir(exist_ok=True)
    path = out / f"{a.shard:02d}_{a.part:02d}.jsonl"
    done = {r["id"] for r in read_jsonl([path])} if path.exists() else set()
    if kind == "labels" and a.round == "train":  # labels made in the eval round are reused
        done |= {r["id"] for r in read_jsonl(sorted((work(a) / "eval" / "labels").glob("*.jsonl")))}
    rows = [{"id": i, "moves": list(m)} for i, m in zip(df.id, df.moves) if i not in done]
    return rows, path, done


def cmd_value(a: argparse.Namespace) -> None:
    rows, path, _ = _katago_rows(a, "values")
    if not rows:
        return
    eng = K.Engine(a.net, visits=1, threads=64, search_threads=1, batch=64)
    prog = Progress(total=len(rows), desc=f"KataGo value {a.round} shard {a.shard} part {a.part}", unit="pos")
    with open(path, "a") as fh:
        def on(resp):
            fh.write(json.dumps({"id": rows[int(resp["id"])]["id"], "value": K.parse_value(resp), "error": resp.get("error")}) + "\n")
            prog.update(1)
        eng.run((K.value_query(str(i), r) for i, r in enumerate(rows)), on)
    prog.close()
    eng.close()


def select(positions: pd.DataFrame, values: Mapping[str, float], n: int) -> pd.DataFrame:
    """Positions with value in [VALUE_LO, VALUE_HI], the first n in game order (game_rank, ply)."""
    v = positions.id.map(values)
    keep = positions[(v >= VALUE_LO) & (v <= VALUE_HI)].sort_values(["game_rank", "ply"], kind="stable")
    return keep.head(n)


def cmd_select(a: argparse.Namespace) -> None:
    d = round_dir(a)
    values = {r["id"]: r["value"] for r in read_jsonl(sorted((d / "values").glob("*.jsonl"))) if r.get("value") is not None}
    files = sorted(d.glob("positions_*.parquet"))
    idx = pd.concat([pd.read_parquet(f, columns=["id", "game_rank", "ply"]) for f in files])
    missing = int((~idx.id.isin(values.keys())).sum())
    if missing > 0.001 * len(idx):
        raise SystemExit(f"{missing} of {len(idx)} positions have no value: run the value step first")
    chosen = set(select(idx, values, a.n or SELECT_N[a.round]).id)
    for f in files:
        df = pd.read_parquet(f)
        df[df.id.isin(chosen)].reset_index(drop=True).to_parquet(d / f.name.replace("positions_", "selected_"))
    dump({"positions": len(idx), "selected": len(chosen)}, d / "select.stats.json")


def cmd_label(a: argparse.Namespace) -> None:
    rows, path, done = _katago_rows(a, "labels")
    if not rows:
        return
    eng = K.Engine(a.net, visits=K.LABEL_VISITS, threads=32, search_threads=2, batch=64)
    prog = Progress(total=len(rows), desc=f"KataGo label {a.round} shard {a.shard} part {a.part}", unit="pos")
    with open(path, "a") as fh:
        def on(resp):
            fh.write(json.dumps({"id": rows[int(resp["id"])]["id"], **K.parse_label(resp)}) + "\n")
            prog.update(1)
        eng.run((K.label_query(str(i), r) for i, r in enumerate(rows)), on)
    prog.close()
    eng.close()
    dump({"net": os.path.basename(a.net), "katago": K.version(), "settings": eng.settings, "rules": K.RULES, "komi": K.KOMI,
          "labelled": len(rows), "reused": len(done), "finished": time.ctime()}, path.with_suffix(".meta.json"))


# ---------------------------------------------------------------- splits
def labelled_row(r: Mapping, lab: Mapping | None) -> tuple[dict | None, str]:
    """(position row, its label or None) -> (split row or None, reason): kept iff answered, not a pass, legal under our rules
    (simple ko) and the search's root winrate in [VALUE_LO, VALUE_HI]."""
    if lab is None:
        return None, "unlabelled"
    if "error" in lab or not lab.get("best"):
        return None, "katago_error"
    if lab["best"].lower() == "pass":
        return None, "best_pass"
    r = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    bx, by = R.from_gtp(lab["best"])
    if not R.legal_mask(r)[by, bx]:
        return None, "best_illegal_ours"
    wr = lab["root_winrate"]
    if not (VALUE_LO <= wr <= VALUE_HI):
        return None, "decided"
    best_wr = next(m["winrate"] for m in lab["cands"] if m["move"] == lab["best"])
    return {**r, "correct": [R.idx_of((bx, by))], "best": lab["best"], "best_order0": lab["best_order0"], "best_winrate": best_wr,
            "root_winrate": wr, "root_score": lab["root_score"], "root_visits": lab["root_visits"],
            "cands": json.dumps(lab["cands"])}, "kept"


def board_key(df: pd.DataFrame) -> pd.Series:
    """side + stones: equal keys = the same picture and the same question."""
    return df.side + ":" + df.black.map(lambda v: ",".join(map(str, v))) + ":" + df.white.map(lambda v: ",".join(map(str, v)))


def kept_rows(positions: pd.DataFrame, labels: Mapping[str, dict], reasons: Counter) -> list[dict]:
    rows = []
    prog = Progress(total=len(positions), desc="filter positions", unit="pos")
    for r in positions.to_dict("records"):
        prog.update(1)
        row, why = labelled_row(r, labels.get(r["id"]))
        reasons[why] += 1
        if row is not None:
            rows.append(row)
    prog.close()
    return rows


def eval_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Kept eval-round positions -> {test, val}: whole games in game order, boards seen in another game kept out."""
    df = df.sort_values(["game_rank", "ply"], kind="stable").reset_index(drop=True)
    key = board_key(df)
    df["shared"] = key.map(df.assign(key=key).groupby("key").game.nunique()).values > 1
    split = np.array([""] * len(df), dtype=object)
    goal, counts, order, cur = {"test": N_TEST, "val": N_VAL}, {"test": 0, "val": 0}, ["test", "val"], 0
    for _, gdf in df.groupby("game_rank", sort=True):
        while cur < len(order) and counts[order[cur]] >= goal[order[cur]]:
            cur += 1
        if cur == len(order):
            break
        s = order[cur]
        for i in gdf.index:
            if counts[s] >= goal[s]:
                break
            if df.at[i, "shared"]:
                continue
            split[i] = s
            counts[s] += 1
    return {s: df[split == s].reset_index(drop=True) for s in order}


def train_split(df: pd.DataFrame, held: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Kept train-round positions -> the first N_TRAIN in game order whose board is in neither val nor test."""
    df = df.sort_values(["game_rank", "ply"], kind="stable").reset_index(drop=True)
    held_keys = set(board_key(held["val"])) | set(board_key(held["test"]))
    dup = board_key(df).isin(held_keys).values
    return df[~dup].head(N_TRAIN).reset_index(drop=True), int(dup.sum())


def held_games(data: Path) -> set:
    return set(pd.read_parquet(data / "val.parquet", columns=["game"]).game) | set(pd.read_parquet(data / "test.parquet", columns=["game"]).game)


def cmd_split(a: argparse.Namespace) -> None:
    d, out = round_dir(a), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    label_dirs = [work(a) / "eval" / "labels"] + ([d / "labels"] if a.round == "train" else [])
    labels = {r["id"]: r for r in read_jsonl(sorted(p for ld in label_dirs for p in ld.glob("*.jsonl")))}
    reasons: Counter = Counter()
    rows = []
    for f in sorted(d.glob("selected_*.parquet")):
        rows += kept_rows(pd.read_parquet(f), labels, reasons)
    df = pd.DataFrame(rows)
    stats = {"reasons": dict(reasons), "kept": len(df)}
    if a.round == "eval":
        for s, part in eval_splits(df).items():
            part.to_parquet(out / f"{s}.parquet")
            stats[s] = {"n": len(part), "games": int(part.game.nunique())}
    else:
        held = {s: pd.read_parquet(out / f"{s}.parquet") for s in ("val", "test")}
        if set(df.game) & held_games(out):
            raise SystemExit("a val / test game is among the train-round positions")
        train, n_dup = train_split(df, held)
        train[TRAIN_COLUMNS].to_parquet(out / "train.parquet")
        stats["train"] = {"n": len(train), "games": int(train.game.nunique()), "board_in_val_test": n_dup}
    dump(stats, d / "split.stats.json")


# ---------------------------------------------------------------- CLI
def work(a: argparse.Namespace) -> Path:
    p = Path(a.work)
    p.mkdir(parents=True, exist_ok=True)
    return p


def round_dir(a: argparse.Namespace) -> Path:
    p = work(a) / a.round
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump(obj: dict, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(json.dumps(obj), flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build the Go data (nbm/tasks/go/build.py docstring).")
    p.add_argument("step", choices=["games", "positions", "value", "select", "label", "split"])
    p.add_argument("--work", required=True, help="directory of the intermediate files")
    p.add_argument("--round", choices=["eval", "train"], default="eval")
    p.add_argument("--kgs", help="games: the unpacked KGS archive (directory of .sgf files)")
    p.add_argument("--out", help="split (and train-round positions): the data directory with test / val / train.parquet")
    p.add_argument("--net", help="value / label: the KataGo network (.bin.gz)")
    p.add_argument("--shard", type=int, default=0, help="value / label: positions shard")
    p.add_argument("--part", type=int, default=0, help="value / label: rows i % parts == part of the shard")
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--shards", type=int, default=None, help="positions: shard count (default: eval 1, train 32)")
    p.add_argument("--n", type=int, default=None, help="select: positions to label (default: eval 125,000, train 1,100,000)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    {"games": cmd_games, "positions": cmd_positions, "value": cmd_value, "select": cmd_select, "label": cmd_label,
     "split": cmd_split}[a.step](a)


if __name__ == "__main__":
    main()
