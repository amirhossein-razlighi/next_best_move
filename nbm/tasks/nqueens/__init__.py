"""N-queens as a next-best-move task (paper, Appendix A.6): place one more queen so that the board can still be completed.

Input: a partial N-queens board (N = 5..12, k non-attacking queens) drawn in the top-left corner of a fixed 12 x 12 canvas
(render.py). Prompt: "Continue." (the rules are never stated). Target: the same picture with the next queen drawn red; the
label is the topmost empty row and the source solution's column there, but a decision counts as correct whenever the new
board still extends to a full solution (any extendable cell, any row). Image model only (Qwen-Image-2.1).

Evaluation (one decision per board): one generation per board and vote sample from render(board), read by the threshold-free
readout (render.decode: the canvas cell with the strongest red glyph, any of the 144 cells), scored by rules.classify as
correct / dead (legal, no completion) / attacked / occupied / off_board. Noise of board i, sample j:
nbm.tasks.base.row_seed(seed, i, j) = seed * 1,000,003 + i + 10,000,019 j, so the result does not depend on the batch layout
or the number of processes beyond bfloat16 rounding (the paper's records used batches of 8 consecutive boards). With
--votes n the decision is the majority cell of the n samples (ties: the earliest sample).

Splits (data.py): train, seen, val, test_a (held-out solution families), test_b (never-trained sizes 11, 12), test_hard.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PIL import Image

from nbm import readout
from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, row_seed
from nbm.tasks.nqueens import render as V
from nbm.tasks.nqueens import rules as R

PROMPT = "Continue."
N_STRIPS = 8  # boards whose (input | output) pictures the main process saves


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """95% Wilson interval of k successes in n trials."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def score_decision(s: str, out: Image.Image | np.ndarray) -> dict:
    """Readout + scoring of one generated picture for board s."""
    d = V.decode(out, s)
    cat = R.classify(s, d["cell"])
    r, c = d["cell"]
    return {"cell_r": int(r), "cell_c": int(c), "cat": cat, "correct": cat == "correct", "legal": cat in ("correct", "dead"),
            "alpha": round(d["alpha"], 4), "second": round(d["second"], 4), "copy_err": round(d["copy_err"], 3),
            "faint": bool(d["alpha"] < 0.5)}


def summarize(records: list[dict]) -> dict:
    """acc (= score) = share of completable placements; legal = on-board, empty and not attacked; cat/<category> shares;
    acc/N<n> per board size; random_safe = expected accuracy of a uniformly random non-attacked empty cell on the same
    boards (the chance level the paper compares with); faint = share of decisions whose mark strength is below 0.5."""
    n = len(records)

    def mean(key, rs=records):
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")
    k = int(sum(r["correct"] for r in records))
    out = {"n": n, "n_correct": k, "acc": mean("correct"), "acc_ci95": wilson(k, n), "legal": mean("legal"),
           "random_safe": float(np.mean([r["n_ext"] / max(1, r["n_safe"]) for r in records])) if records else float("nan"),
           "faint": mean("faint")}
    for c in R.CATEGORIES:
        out[f"cat/{c}"] = float(np.mean([r["cat"] == c for r in records])) if records else float("nan")
    for size in sorted({r["N"] for r in records}):
        out[f"acc/N{size}"] = mean("correct", [r for r in records if r["N"] == size])
    out["score"] = out["acc"]
    return out


class NQueens(Task):
    families = ("image",)

    def load_split(self, split: str) -> pd.DataFrame:
        df = pd.read_parquet(self.data_dir / f"{split}.parquet")
        df["state"] = df["state"].astype(str)
        return df

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        s, cell = str(row["state"]), (int(row["label_r"]), int(row["label_c"]))
        return ImageExample(V.render_pil(s), V.render_pil(s, cell), PROMPT, [V.cell_box(*cell)])

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        states = [str(s) for s in df.state]
        batches = ctx.my_batches(len(df))
        mine = []
        prog = ctx.progress(len(batches) * ctx.votes, f"nqueens {ctx.split}: {len(df)} boards x {ctx.votes} samples, {ctx.steps} steps")
        for j in range(ctx.votes):
            for _, idx in batches:
                outs = ctx.generate([V.render_pil(states[i]) for i in idx], [PROMPT] * len(idx), [row_seed(ctx.seed, i, j) for i in idx])
                for i, out in zip(idx, outs):
                    mine.append((i, j, score_decision(states[i], out)))
                    if j == 0 and i < N_STRIPS and ctx.out_dir is not None:
                        (ctx.out_dir / "strips").mkdir(parents=True, exist_ok=True)
                        strip = np.concatenate([V.render(states[i]), V.to_array(out)], 1)
                        Image.fromarray(strip).save(ctx.out_dir / "strips" / f"{i:05d}_N{len(states[i])}_{mine[-1][2]['cat']}.png")
                prog.update(1)
        prog.close()
        by_row: dict[int, dict[int, dict]] = {}
        for i, j, d in ctx.gather(mine):
            by_row.setdefault(i, {})[j] = d
        records = []
        for i, row in enumerate(df.itertuples(index=False)):
            samples = [by_row[i][j] for j in range(ctx.votes)]
            cell = readout.majority_vote([(d["cell_r"], d["cell_c"]) for d in samples])
            first = next(d for d in samples if (d["cell_r"], d["cell_c"]) == cell)
            rec = {"row": i, "id": row.id, "N": int(row.N), "k": int(row.k), "state": states[i], "n_ext": int(row.n_ext),
                   "n_safe": int(row.n_safe), "subtype": str(row.subtype), **first}
            if ctx.votes > 1:
                rec["samples"] = [[d["cell_r"], d["cell_c"]] for d in samples]
            records.append(rec)
        return EvalResult(records, summarize(records))


TASK = NQueens
