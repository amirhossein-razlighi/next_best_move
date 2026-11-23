"""Chess evaluation: one picture (image model) or clip (video model) per puzzle and vote sample, read with the threshold-free
readout, scored on the first move.

Noise and batches follow the paper's evaluations exactly, so that a record can be reproduced row by row (outputs depend on the
composition of a batch at bfloat16 rounding level):
  image  The rows are cut into consecutive chunks of eval.chunk_rows (2,500: the 10,000 test puzzles were run as 4 chunks;
         eval.chunk_rows=10000 is the single-piece run of Tab. 1's 20-step row). Inside a chunk the white-to-move rows come
         first, then the black ones, in batches of ctx.batch_size (8) of one side each (one prompt layout per batch); the k-th
         batch of a chunk starts from nbm.models.noise.batch_noise(seed + j, k), j = the vote sample.
  video  The rows are cut into chunks of eval.chunk_rows (1,000); the rows of a chunk are dealt round-robin into eval.lanes (4)
         lanes (the paper's 4-GPU layout) and each lane is cut into batches of ctx.batch_size (4). Every row has its own noise,
         nbm.tasks.base.row_seed(seed, row, j).
Majority vote (--votes n): sample j of every row, j = 0 .. n - 1, read out separately; the most frequent decision wins, NO_MOVE
counting as one more decision, ties going to the earliest sample (nbm.readout.majority_vote).
"""
from __future__ import annotations

import chess
import numpy as np
import pandas as pd
from PIL import Image

from nbm import readout
from nbm.models.noise import batch_noise
from nbm.models.video import video_to_blocks
from nbm.tasks.base import EvalContext, EvalResult, row_seed
from nbm.tasks.chess import board as B
from nbm.tasks.chess import decode as D

SUBSETS = ("sacrifice", "far_from_training")  # boolean columns of the split (load_split), scored when present
EXAMPLES = 8  # default of eval.examples: rows (of the first ones) whose pictures the main process writes to <out>/examples/


def image_batches(sides: np.ndarray, chunk_rows: int, bs: int) -> list[tuple[int, list[int]]]:
    """[(k, rows)]: per chunk of chunk_rows rows, the white rows then the black rows in batches of bs; k counts the batches of
    a chunk from 0."""
    out = []
    for c0 in range(0, len(sides), chunk_rows):
        rows = np.arange(c0, min(len(sides), c0 + chunk_rows))
        k = 0
        for side in ("white", "black"):
            idx = rows[sides[rows] == side]
            for i in range(0, len(idx), bs):
                out.append((k, idx[i:i + bs].tolist()))
                k += 1
    return out


def video_batches(n: int, chunk_rows: int, lanes: int, bs: int) -> list[list[int]]:
    """Per chunk of chunk_rows rows: lane r = rows r, r + lanes, ... of the chunk, cut into batches of bs."""
    out = []
    for c0 in range(0, n, chunk_rows):
        rows = list(range(c0, min(n, c0 + chunk_rows)))
        for r in range(lanes):
            lane = rows[r::lanes]
            out += [lane[i:i + bs] for i in range(0, len(lane), bs)]
    return out


def save_example(ctx: EvalContext, row: int, pid: str, pictures: list[Image.Image]) -> None:
    """Main process: the given pictures side by side as <out>/examples/<row>_<puzzle id>.png, for rows < eval.examples."""
    if ctx.out_dir is None or row >= int(ctx.cfg.eval.get("examples", EXAMPLES)):
        return
    w, h = pictures[0].size
    strip = Image.new("RGB", (w * len(pictures), h), "white")
    for i, p in enumerate(pictures):
        strip.paste(p.convert("RGB").resize((w, h)), (i * w, 0))
    (ctx.out_dir / "examples").mkdir(parents=True, exist_ok=True)
    strip.save(ctx.out_dir / "examples" / f"{row:05d}_{pid}.png")


def evaluate_image(ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
    """Image model: every (vote sample, batch) item on the process it falls to (item p -> process p % world)."""
    size = int(ctx.cfg.model.resolution)
    chunk = int(ctx.cfg.eval.get("chunk_rows") or len(df) or 1)
    batches = image_batches(df.side.values, chunk, ctx.batch_size)
    items = [(j, k, rows) for j in range(ctx.votes) for k, rows in batches]
    mine = [it for p, it in enumerate(items) if p % ctx.world == ctx.rank]
    shape = ctx.model.latent_shape(size, size)
    out = []
    prog = ctx.progress(len(mine), f"chess image {ctx.split}: {len(df)} puzzles x {ctx.votes} samples, {ctx.steps} steps")
    for j, k, rows in mine:
        conds = [B.render_board(chess.Board(df.fen[i]), size) for i in rows]
        pics = ctx.model.generate(conds, [B.image_prompt(df.side[i]) for i in rows], ctx.steps,
                                  batch_noise(ctx.seed + j, k, len(rows), shape, ctx.model.device))
        for i, cond, pic in zip(rows, conds, pics):
            out.append((i, j, D.read_image_move(pic, df.fen[i], cond)))
            if j == 0:
                save_example(ctx, i, df.puzzle_id[i], [cond, B.image_pair(df.fen[i], df.line[i][0], size)[1], pic])
        prog.update(1)
    prog.close()
    return summarize(ctx, df, ctx.gather(out))


def evaluate_video(ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
    """Video model: one clip per row and vote sample; the decision is read from the clip's first DECIDE block."""
    size = int(ctx.cfg.model.resolution)
    chunk = int(ctx.cfg.eval.get("chunk_rows") or len(df) or 1)
    lanes = int(ctx.cfg.eval.get("lanes", 1))
    batches = video_batches(len(df), chunk, lanes, ctx.batch_size)
    items = [(j, rows) for j in range(ctx.votes) for rows in batches]
    mine = [it for p, it in enumerate(items) if p % ctx.world == ctx.rank]
    out = []
    prog = ctx.progress(len(mine), f"chess video {ctx.split}: {len(df)} puzzles x {ctx.votes} samples, {ctx.steps} steps")
    for j, rows in mine:
        conds = [B.render_board(chess.Board(df.fen[i]), size) for i in rows]
        clips = ctx.generate(conds, [B.video_prompt(df.side[i]) for i in rows], [row_seed(ctx.seed, i, j) for i in rows])
        for i, cond, clip in zip(rows, conds, clips):
            blocks = video_to_blocks(clip)
            out.append((i, j, D.read_clip_move(blocks[1], df.fen[i], cond)))
            if j == 0:
                save_example(ctx, i, df.puzzle_id[i], blocks)
        prog.update(1)
    prog.close()
    return summarize(ctx, df, ctx.gather(out))


def summarize(ctx: EvalContext, df: pd.DataFrame, decisions: list[tuple[int, int, dict]]) -> EvalResult:
    """Records (one per puzzle) and metrics from the (row, sample, decision) triples of every process."""
    by_row: dict[int, dict[int, dict]] = {}
    for i, j, d in decisions:
        by_row.setdefault(i, {})[j] = d
    records = []
    for i in range(len(df)):
        samples = [by_row[i][j]["move"] for j in range(ctx.votes)]
        answer = df.line[i][0]
        pred = readout.majority_vote(samples)
        rec = {"row": i, "puzzle_id": df.puzzle_id[i], "fen": df.fen[i], "answer": answer, "rating": int(df.rating[i]),
               "rating_bucket": df.rating_bucket[i], "pred": pred, "correct": D.same_move(pred, answer), "samples": samples,
               "blue_peak": by_row[i][0]["blue_peak"], "red_peak": by_row[i][0]["red_peak"]}
        rec.update({s: bool(df[s][i]) for s in SUBSETS if s in df})
        records.append(rec)
    acc = lambda rs: float(np.mean([r["correct"] for r in rs])) if rs else float("nan")  # noqa: E731
    metrics = {"n": len(records), "acc": acc(records), "score": acc(records),
               "legal": float(np.mean([r["samples"][0] != D.NO_MOVE for r in records])) if records else float("nan")}
    for s in SUBSETS:
        if s in df:
            sub = [r for r in records if r[s]]
            metrics[f"acc_{s}"], metrics[f"n_{s}"] = acc(sub), len(sub)
    if ctx.votes > 1:
        metrics["acc_sample0"] = float(np.mean([D.same_move(r["samples"][0], r["answer"]) for r in records]))
        metrics["acc_per_sample"] = float(np.mean([[D.same_move(m, r["answer"]) for m in r["samples"]] for r in records]))
        metrics["acc_any_sample"] = float(np.mean([any(D.same_move(m, r["answer"]) for m in r["samples"]) for r in records]))
    for b in sorted({r["rating_bucket"] for r in records}):
        metrics[f"acc_rating_{b}"] = acc([r for r in records if r["rating_bucket"] == b])
    return EvalResult(records, metrics)
