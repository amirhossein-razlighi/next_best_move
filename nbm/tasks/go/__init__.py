"""Go (19 x 19): the next best move of a KGS game position, labelled by KataGo; image model only.

Data (nbm/tasks/go/build.py, docs/tasks/go.md): positions from strong human games (KGS archive), the label = KataGo's move after a
400-visit search (Chinese rules, komi 7.5). Splits by game: test 2,000 / val 1,000 positions, train 1,000,000.

Image model: condition = the whole board (nbm/tasks/go/render.py), prompt "<Black|White> to play. Mark the best move with a red
dot.", target = the board with a red disc on the KataGo move; the disc's cell is the highlight region of the loss. Training batches
hold one side to move (group 0 = white, 1 = black: the two prompts have different token layouts).

Evaluation (one-shot; the paper's Table 2 numbers come from the 2,000 test positions):
  readout   threshold-free: the point with the strongest red disc over all 361 points (render.read_move). legal = it is a legal
            move (empty, not suicide, not the ko point); best = it is the KataGo move (an illegal point is wrong).
  batches   the paper's evaluation batches: all white-to-move rows first, then all black-to-move rows, in row order, eval.batch_size
            per batch; global batch k goes to process k % world. Outputs depend on the batch composition at bfloat16 rounding
            level, so these batches are needed to reproduce the paper's per-row records.
  noise     row i of sample j: CPU generator seeded (seed + j) * 1,000,003 + i (sample j of a majority vote = the single-sample
            evaluation with seed + j).
  votes     --votes n: the most frequent decision over n samples (ties: the earliest sample).
Metrics: legal_move, best_move (= score), best_move/<phase> (moves already played: 20-59, 60-119, 120-179, 180+), chance (mean
1 / number of legal moves).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nbm import readout
from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, row_seed
from nbm.tasks.go import render
from nbm.tasks.go.rules import PHASES, from_gtp, gtp, legal_mask, pt_of

SIDES = ("white", "black")  # batch group 0 / 1
TRAIN_COLUMNS = ["id", "side", "black", "white", "ko", "correct"]
SAVE_PICTURES = 8  # board + output of the rows < 8 that the main process samples, written to <out>/images


def prompt_for(side: str) -> str:
    return f"{side.capitalize()} to play. Mark the best move with a red dot."


def phase_stratified(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Rows reordered (seeded, random within a phase) so that every prefix is proportionally phase-stratified: the order of the
    validation rows of the in-training evaluation (its first train.eval_n rows)."""
    rng = np.random.default_rng(seed)
    key = np.empty(len(df))
    for ph in sorted(df.phase.unique()):
        idx = np.flatnonzero(df.phase.values == ph)
        key[rng.permutation(idx)] = (np.arange(len(idx)) + 0.5) / len(idx)
    return df.iloc[np.argsort(key, kind="stable")].reset_index(drop=True)


def side_batches(sides: np.ndarray, bs: int) -> list[list[int]]:
    """The evaluation batches: white-to-move rows in groups of bs, then black-to-move rows."""
    out = []
    for s in SIDES:
        idx = np.flatnonzero(sides == s).tolist()
        out += [idx[j:j + bs] for j in range(0, len(idx), bs)]
    return out


class Go(Task):
    families = ("image",)

    def load_split(self, split: str) -> pd.DataFrame:
        """test / val / train parquet of data.dir; train reads only the columns training needs; val in the phase-stratified
        order (test keeps the file order)."""
        path = Path(self.data_dir) / f"{split}.parquet"
        if split == "train":
            return pd.read_parquet(path, columns=TRAIN_COLUMNS)
        df = pd.read_parquet(path)
        if "phase" not in df:  # files of the paper's release name the phase column "level"
            df["phase"] = df["level"]
        return phase_stratified(df) if split == "val" else df

    def batch_groups(self, df: pd.DataFrame) -> np.ndarray:
        return (df.side.values == "black").astype(np.int64)

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        move = pt_of(row["correct"][0])
        cond = render.render_board(row, self.resolution)
        return ImageExample(cond, render.draw_move(cond, move), prompt_for(row["side"]), [render.cell_box(move, self.resolution)])

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        rows = df.to_dict("records")
        size = self.resolution
        batches = [(k, b) for k, b in enumerate(side_batches(df.side.values, ctx.batch_size)) if k % ctx.world == ctx.rank]
        prog = ctx.progress(len(batches) * ctx.votes, f"go {ctx.split} ({len(rows)} rows x {ctx.votes} samples, {ctx.steps} steps)")
        mine = []
        for j in range(ctx.votes):
            for _, idx in batches:
                conds = [render.render_board(rows[i], size) for i in idx]
                outs = ctx.generate(conds, [prompt_for(rows[i]["side"]) for i in idx], [row_seed(ctx.seed + j, i) for i in idx])
                for i, cond, out in zip(idx, conds, outs):
                    pt, strength = render.read_move(out, cond)
                    mine.append((i, j, gtp(pt), strength))
                    if ctx.out_dir is not None and j == 0 and i < SAVE_PICTURES:
                        (ctx.out_dir / "images").mkdir(parents=True, exist_ok=True)
                        out.save(ctx.out_dir / "images" / f"{i:05d}_output.png")
                        cond.save(ctx.out_dir / "images" / f"{i:05d}_board.png")
                prog.update(1)
        prog.close()
        samples: dict[int, dict[int, tuple[str, float]]] = {}
        for i, j, move, strength in ctx.gather(mine):
            samples.setdefault(i, {})[j] = (move, strength)
        records = []
        for i, row in enumerate(rows):
            moves = [samples[i][j][0] for j in range(ctx.votes)]
            move = readout.majority_vote(moves)
            legal = legal_mask(row)
            label = gtp(pt_of(row["correct"][0]))
            mx, my = from_gtp(move)
            records.append({"id": row["id"], "side": row["side"], "phase": row["phase"], "label": label, "move": move,
                            "legal": bool(legal[my, mx]), "best": move == label, "samples": moves,
                            "strength": samples[i][0][1], "n_legal": int(legal.sum())})
        return EvalResult(records, summarize(records))


def summarize(records: list[dict]) -> dict:
    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")

    out = {"n": len(records), "legal_move": mean([r["legal"] for r in records]), "best_move": mean([r["best"] for r in records]),
           "chance": mean([1 / r["n_legal"] for r in records])}
    for name, _, _ in PHASES:
        v = [r["best"] for r in records if r["phase"] == name]
        if v:
            out[f"best_move/{name}"], out[f"n/{name}"] = mean(v), len(v)
    out["score"] = out["best_move"]
    return out


TASK = Go
