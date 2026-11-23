"""Toy task "dark cell" (for the tests and as a template; not listed by nbm.tasks.available_tasks).

A 256 px board of 4 x 4 cells, one of them dark. The action is that cell: the image model marks it RED; the video model's clip
is DECIDE (the RED mark), IMAGINE (the cell cleared), then that board held. Readout: the strongest red cell. The rows are
generated from a seed, so the task needs no data files. Options (cfg.options): n_train / n_val / n_test rows (defaults below).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from nbm import marks, readout
from nbm.models.video import N_BLOCKS, video_to_blocks
from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, VideoExample, row_seed

SIZE = 256
GRID = 4
CELL = SIZE // GRID
BG = (235, 232, 225)
DARK = (60, 60, 60)
PROMPT = "Mark the dark cell in red."
VIDEO_PROMPT = "Continue."
SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}
DEFAULT_ROWS = {"train": 1000, "val": 64, "test": 64}


def cell_box(cell: int) -> marks.Box:
    r, c = divmod(cell, GRID)
    return (c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL)


def render(cell: int | None, shade: int = 0) -> Image.Image:
    """The board; cell None = no dark cell (the IMAGINE frame). shade shifts the light cells (training jitter)."""
    arr = np.empty((SIZE, SIZE, 3), np.uint8)
    arr[:] = np.clip(np.array(BG) + shade, 0, 255)
    if cell is not None:
        x0, y0, x1, y1 = cell_box(cell)
        arr[y0:y1, x0:x1] = DARK
    arr[::CELL, :] = arr[:, ::CELL] = 160  # grid lines
    return Image.fromarray(arr)


def decode(out: Image.Image, cond: Image.Image) -> dict:
    """Red strength per cell (shared decoder D) and the threshold-free decision (the strongest red cell)."""
    coef, _ = marks.fit_strengths(marks.as_array(out, cond.size), marks.as_array(cond), [cell_box(k) for k in range(GRID * GRID)],
                                  [marks.RED, marks.BLUE], margin=4)
    red = marks.to_strength(coef[:, 0])
    return {"cell": readout.strongest(red), "red_peak": float(red.max())}


class DarkCell(Task):
    families = ("image", "video")

    def load_split(self, split: str) -> pd.DataFrame:
        n = int(self.options.get(f"n_{split}", DEFAULT_ROWS[split]))
        rng = np.random.default_rng(SPLIT_SEEDS[split])
        return pd.DataFrame({"id": [f"{split}{i}" for i in range(n)], "cell": rng.integers(GRID * GRID, size=n)})

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        shade = 0 if rng is None else int(20 * rng.uniform()) - 10
        cond = render(row["cell"], shade)
        box = cell_box(row["cell"])
        return ImageExample(cond, marks.tint(cond, [(box, marks.RED)]), PROMPT, [box])

    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample:
        cond = render(row["cell"])
        decide = marks.tint(cond, [(cell_box(row["cell"]), marks.RED)])
        blocks = [cond, decide] + [render(None)] * (N_BLOCKS - 2)
        return VideoExample(np.stack([np.asarray(b) for b in blocks]), VIDEO_PROMPT)

    def prompts(self) -> list[str]:
        return [VIDEO_PROMPT]

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        """One-shot: one output per row and vote sample (noise seed row_seed(seed, row, sample)); majority vote over the samples."""
        conds = [render(int(c)) for c in df.cell]
        prompt = PROMPT if ctx.family == "image" else VIDEO_PROMPT
        mine = []
        batches = ctx.my_batches(len(df))
        prog = ctx.progress(len(batches) * ctx.votes, f"dark cell {ctx.split} ({len(df)} rows x {ctx.votes})")
        for k in range(ctx.votes):
            for _, idx in batches:
                outs = ctx.generate([conds[i] for i in idx], [prompt] * len(idx), [row_seed(ctx.seed, i, k) for i in idx])
                for i, out in zip(idx, outs):
                    pic = out if ctx.family == "image" else video_to_blocks(out)[1]
                    mine.append((i, k, decode(pic, conds[i])))
                prog.update(1)
        prog.close()
        by_row: dict[int, dict] = {}
        for i, k, d in ctx.gather(mine):
            by_row.setdefault(i, {})[k] = d
        records = []
        for i, row in enumerate(df.itertuples()):
            samples = [by_row[i][k]["cell"] for k in range(ctx.votes)]
            pred = readout.majority_vote(samples)
            records.append({"id": row.id, "cell": int(row.cell), "pred": pred, "samples": samples, "correct": pred == row.cell,
                            "red_peak": by_row[i][0]["red_peak"]})
        acc = float(np.mean([r["correct"] for r in records])) if records else float("nan")
        return EvalResult(records, {"n": len(records), "acc": acc, "score": acc})


TASK = DarkCell
