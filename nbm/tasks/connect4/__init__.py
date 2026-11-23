"""Connect Four (paper Table 2, Appendix A.3): the model chooses a column for the side to move by drawing the disc.

One fixed neutral prompt, "Continue." (no rule text, no legend); 256 px pictures (render.py).
  image model  input = the position; target = the DECIDE picture of the training label (row `choice`, the canonical
               exact-best column): the mover's disc in the column's landing cell, tinted GREEN.
  video model  the clip of the solver's best line from the position (both sides, up to 8 plies, render.line_blocks): block 0 =
               the position, then DECIDE / IMAGINE blocks; the decision is the clip's first DECIDE block.
Evaluation (one decision per position): the threshold-free readout of the output picture / first DECIDE block (decode.py);
the move is valid iff it is legal and the disc is drawn in the correct landing row. Metrics per split (any exact-best column
counts): legal_landing (valid), best (an exact-best column), keeps_value (the move keeps the game-theoretic value: a won
position stays won, a drawn one drawn; "keeps the win" in the paper, whose hard test set is 94% won positions). The video
record also reads the imagined continuation (decode.read_line). Labels come from the data, so evaluation needs no solver.

Noise: row i, vote sample k -> seed nbm.tasks.base.row_seed(seed, i, k) (= seed * 1,000,003 + i for k = 0, the paper's
evaluation); batches = consecutive rows (image 8, video 2), so a row's output matches the paper's records at equal batches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nbm import readout
from nbm.models.video import video_to_blocks
from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, VideoExample, row_seed
from nbm.tasks.connect4 import decode as DEC
from nbm.tasks.connect4 import game as G
from nbm.tasks.connect4 import render as R

PROMPT = "Continue."
N_STRIPS = 8  # the first rows' input / output pictures saved under <out>/strips
LIST_COLUMNS = ("scores", "best", "vp", "line")


def read_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    """<data_dir>/<split>.parquet (data.py) with its list columns as lists of Python ints."""
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    for k in LIST_COLUMNS:
        df[k] = df[k].apply(lambda v: [int(x) for x in v])
    return df


def move_scores(row: dict) -> list[int | None]:
    return [None if int(s) == -1000 else int(s) for s in row["scores"]]


def score_move(col: int | None, scores: list[int | None]) -> dict:
    """best / keeps_value of a decided column (None = no valid move)."""
    ok = col is not None and scores[col] is not None
    return {"col": col, "best": bool(ok and scores[col] == max(s for s in scores if s is not None)),
            "keeps_value": bool(ok and G.sign(scores[col]) == G.value(scores)), "move_score": scores[col] if ok else None}


def _mean(records: list[dict], key: str) -> float:
    v = [r[key] for r in records if r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


class ConnectFour(Task):
    families = ("image", "video")

    def load_split(self, split: str) -> pd.DataFrame:
        return read_split(self.data_dir, split)

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        b, c = row["board"], int(row["choice"])
        cond, target = Image.fromarray(R.render(b)), Image.fromarray(R.render(b, decide=c))
        return ImageExample(cond, target, PROMPT, [R.cell_box(c, G.landing(b, c))])

    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample:
        return VideoExample(R.line_blocks(row["board"], [int(x) for x in row["line"]]), PROMPT)

    def prompts(self) -> list[str]:
        return [PROMPT]

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        rows = df.to_dict("records")
        conds = [Image.fromarray(R.render(r["board"])) for r in rows]
        batches = ctx.my_batches(len(rows))
        video = ctx.family == "video"
        prog = ctx.progress(len(batches) * ctx.votes, f"connect4 {ctx.family} {ctx.split} ({len(rows)} rows x {ctx.votes})")
        mine, acc = [], []
        for k in range(ctx.votes):
            for _, idx in batches:
                outs = ctx.generate([conds[i] for i in idx], [PROMPT] * len(idx), [row_seed(ctx.seed, i, k) for i in idx])
                for i, out in zip(idx, outs):
                    b = rows[i]["board"]
                    blocks = [np.asarray(x) for x in video_to_blocks(out)] if video else None
                    pic = blocks[1] if video else np.asarray(out.convert("RGB"))
                    mv = DEC.read_move(pic, b)
                    res = {"move": mv}
                    if video and k == 0:
                        res["line"] = DEC.read_line(blocks, b, mv)
                    if k == 0 and i < N_STRIPS and ctx.out_dir is not None:
                        strip = np.concatenate(blocks if video else [np.asarray(conds[i]), pic], 1)
                        (ctx.out_dir / "strips").mkdir(exist_ok=True)
                        Image.fromarray(strip).save(ctx.out_dir / "strips" / f"{i:05d}_{rows[i]['id']}.png")
                    mine.append((i, k, res))
                    acc.append(score_move(mv["col"] if mv["valid"] else None, move_scores(rows[i]))["keeps_value"])
                prog.update(1, keeps_value=f"{np.mean(acc):.3f}")
        prog.close()
        by_row: dict[int, dict] = {}
        for i, k, res in ctx.gather(mine):
            by_row.setdefault(i, {})[k] = res
        records = [self._record(i, r, by_row[i], ctx.votes) for i, r in enumerate(rows)]
        return EvalResult(records, self._metrics(records, video))

    @staticmethod
    def _record(i: int, row: dict, samples: dict[int, dict], votes: int) -> dict:
        """One record per position: the decision (majority over the valid sample moves when votes > 1) and its scores, the
        first sample's mark (mark_*, legal, landing_ok) and, for the video model, the imagined line of the first sample."""
        first = samples[0]["move"]
        cols = [s["move"]["col"] if s["move"]["valid"] else None for s in (samples[k] for k in range(votes))]
        decision = readout.majority_vote(cols)
        rec = {"row": i, "id": row["id"], "ply": int(row["ply"]), "kind": row["kind"], "hard": bool(row["hard"]),
               "value": int(row["value"]), "choice": int(row["choice"]), "mark_col": first["col"], "mark_row": first["row"],
               "mark_strength": round(first["strength"], 4), "legal": first["legal"], "landing_ok": first["landing_ok"],
               "legal_landing": decision is not None, **score_move(decision, move_scores(row))}
        if votes > 1:
            rec["samples"] = cols
        if "line" in samples[0]:
            rec["imagined"] = samples[0]["line"]
        return rec

    @staticmethod
    def _metrics(records: list[dict], video: bool) -> dict:
        """legal_landing, best, keeps_value (all rows and hard rows); video: imagined-line rates. score = best (video: plus
        0.001 x the later-move legality of the imagined lines)."""
        m: dict = {"n": len(records)}
        for k in ("legal_landing", "best", "keeps_value"):
            m[k] = _mean(records, k)
        hard = [r for r in records if r["hard"]]
        m["n_hard"] = len(hard)
        for k in ("best", "keeps_value"):
            m[f"{k}_hard"] = _mean(hard, k)
        m["score"] = m["best"]
        if video:
            lines = [r["imagined"] for r in records]
            for k in ("plies", "imagine_ok", "later_legal"):
                m[f"imagined_{k}"] = _mean(lines, k)
            stops: dict[str, int] = {}
            for ln in lines:
                stops[ln["stop"]] = stops.get(ln["stop"], 0) + 1
            m["imagined_stops"] = stops
            later = m["imagined_later_legal"]
            m["score"] = m["best"] + 0.001 * (0.0 if np.isnan(later) else later)
        return m


TASK = ConnectFour
