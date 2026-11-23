"""Billiards ("pot the red"): a continuous action (cue angle and power) on a simulated pool table (paper Table 2, Appendix A.5).

A layout holds the cue ball, the red target ball and 0-4 other balls. The model draws ONE shot, decoded by the threshold-free
readout (decode.py) and executed in the simulator (physics.py): the shot succeeds ("ball potted") iff the red ball is potted and
the cue ball is not. Rules and physics are never given to the model; it learns them from solver-labelled shots (solver.py).

  image model  input = the layout picture (512 x 288), prompt "Continue."; target = the same picture + the label shot's mark
               (render.py: cyan aim line to the cushion under the balls, magenta cue stick, cyan power bar).
  video model  the 17-block clip of the label shot: layout, DECIDE (= the image target), the simulated balls at t = 0.2 .. 2.8 s,
               the resting state. Only the DECIDE block is read: its shot is executed (the imagined motion is not used).

Evaluation (Task.evaluate): one decision per layout. Noise of row i, vote sample k: a CPU generator seeded with
(seed + k) * 1,000,003 + i (nbm.models.noise.row_noise), so sample k of a vote is exactly the single-sample evaluation with seed
seed + k. Paper settings: image 20 Euler steps, batch 8; video 30 UniPC steps, batch 2; --steps 1 = the one-step rows. With
--votes n the n decoded shots are combined into one: the circular median of their angles (differences wrapped around the first
sample's angle) and the median power (the paper's billiards vote); outputs without any shot mark are left out.
Splits: train, val, test (Table 2), test_hard and test_mirror (Appendix A.5); see data.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PIL import Image

from nbm.models.video import video_to_blocks
from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, VideoExample
from nbm.tasks.billiards import data as D
from nbm.tasks.billiards import physics as P
from nbm.tasks.billiards import render as RD
from nbm.tasks.billiards.decode import read_shot

PROMPT = "Continue."
SEED_STRIDE = 1_000_003  # noise seed of row i, sample k: (seed + k) * SEED_STRIDE + i
N_STRIPS = 8             # rows whose pictures the main process saves under <out>/strips


def layout(row: dict) -> tuple[np.ndarray, list[int]]:
    """(xy (n, 2), colours) of a row."""
    return np.asarray(row["xy"], np.float64).reshape(-1, 2), [int(c) for c in row["colors"]]


def cue_box(xy: np.ndarray) -> tuple[int, int, int, int]:
    """Pixel box around the cue ball (the highlight region; its loss weight is 1 in the paper runs, so it only splits the logged
    loss into 'hl' / 'rest')."""
    u, v = RD.to_px(*xy[0])
    r = P.R * RD.SCALE + 2
    return (int(u - r), int(v - r), int(u + r + 1), int(v + r + 1))


def play(row: dict, angle: float, power: float) -> dict:
    """Execute a shot from the row's layout and compare it with the label shot."""
    xy, _ = layout(row)
    power = float(np.clip(power, 0.0, 1.0))
    s = P.simulate(xy, None, float(angle) % 360.0, power)
    red, cue = s["pocket"][1] >= 0, s["pocket"][0] >= 0
    return {"success": bool(red and not cue), "red_potted": bool(red), "scratch": bool(cue), "pocket": int(s["pocket"][1]),
            "angle_err": abs(P.ang_diff(float(angle), float(row["angle"]))), "power_err": abs(power - float(row["power"]))}


MISS = {"success": False, "red_potted": False, "scratch": False, "pocket": -1, "angle_err": None, "power_err": None}


def vote(shots: list[tuple[float, float]]) -> tuple[float, float]:
    """Circular median of the angles (wrapped differences to the first one) and median power."""
    a = np.array([s[0] for s in shots])
    wrapped = (a - a[0] + 180.0) % 360.0 - 180.0
    return float((a[0] + np.median(wrapped)) % 360.0), float(np.median([s[1] for s in shots]))


def _strip(pics: list[np.ndarray], cols: int) -> Image.Image:
    """Pictures in a grid of `cols` columns at half size."""
    rows = -(-len(pics) // cols)
    w, h = RD.CW // 2, RD.CH // 2
    im = Image.new("RGB", (cols * w, rows * h), "white")
    for k, p in enumerate(pics):
        im.paste(Image.fromarray(p).resize((w, h), Image.BILINEAR), ((k % cols) * w, (k // cols) * h))
    return im


class Billiards(Task):
    families = ("image", "video")

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.resolution != RD.CH:
            raise ValueError(f"billiards pictures are {RD.CW} x {RD.CH}: model.resolution must be {RD.CH}")

    def load_split(self, split: str) -> pd.DataFrame:
        return D.load(self.data_dir, split)

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        xy, cols = layout(row)
        shot = (float(row["angle"]), float(row["power"]))
        return ImageExample(RD.render_pil(xy, cols), RD.render_pil(xy, cols, None, shot), PROMPT, [cue_box(xy)])

    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample:
        xy, cols = layout(row)
        return VideoExample(RD.shot_blocks(xy, cols, float(row["angle"]), float(row["power"])), PROMPT)

    def prompts(self) -> list[str]:
        return [PROMPT]

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        rows = df.to_dict("records")
        layouts = [layout(r) for r in rows]
        refs = [RD.render(xy, cols) for xy, cols in layouts]
        batches = ctx.my_batches(len(rows))
        prog = ctx.progress(len(batches) * ctx.votes, f"billiards {ctx.family} {ctx.split} ({len(rows)} layouts x {ctx.votes})")
        mine = []
        for k in range(ctx.votes):
            for _, idx in batches:
                outs = ctx.generate([Image.fromarray(refs[i]) for i in idx], [PROMPT] * len(idx),
                                    [(ctx.seed + k) * SEED_STRIDE + i for i in idx])
                for j, i in enumerate(idx):
                    if ctx.family == "image":
                        pic = np.asarray(outs[j].convert("RGB").resize((RD.CW, RD.CH), Image.BILINEAR))
                        shown = [refs[i], pic]
                    else:
                        blocks = [np.asarray(b) for b in video_to_blocks(outs[j])]
                        pic, shown = blocks[1], blocks
                    mine.append((i, k, read_shot(pic, refs[i], layouts[i][0][0])))
                    if ctx.out_dir is not None and k == 0 and i < N_STRIPS:
                        self._save_strip(ctx, i, rows[i], shown)
                prog.update(1, sample=k)
        prog.close()
        by_row: dict[int, dict[int, dict]] = {}
        for i, k, d in ctx.gather(mine):
            by_row.setdefault(i, {})[k] = d
        records = [self._record(i, r, [by_row[i][k] for k in range(ctx.votes)]) for i, r in enumerate(rows)]
        return EvalResult(records, self._summary(records))

    def _save_strip(self, ctx: EvalContext, i: int, row: dict, shown: list[np.ndarray]) -> None:
        """Image: layout / label shot / output stacked; video: the label shot, then the 17 generated blocks."""
        xy, cols = layout(row)
        label = RD.render(xy, cols, None, (float(row["angle"]), float(row["power"])))
        d = ctx.out_dir / "strips"
        d.mkdir(parents=True, exist_ok=True)
        pics = [shown[0], label, shown[1]] if ctx.family == "image" else [label] + shown
        _strip(pics, 1 if ctx.family == "image" else 6).save(d / f"{i:05d}_{row['id']}.png")

    @staticmethod
    def _record(i: int, row: dict, samples: list[dict]) -> dict:
        drawn = [s for s in samples if s["drawn"]]
        rec = {"row": i, "id": row["id"], "k_others": int(row["k_others"]), "hard": bool(row["hard"]), "shot_type": row["shot_type"],
               "label_angle": float(row["angle"]), "label_power": float(row["power"]), "label_pocket": int(row["pocket"]),
               "drawn": bool(drawn)}
        if len(samples) == 1:
            s = samples[0]
            rec.update(angle=s["angle"], power=s["power"], line_score=s["line_score"], bar_score=s["bar_score"])
        else:
            rec.update(angle=None, power=None, samples=[[s["angle"], s["power"], s["drawn"]] for s in samples])
            if drawn:
                rec["angle"], rec["power"] = vote([(s["angle"], s["power"]) for s in drawn])
        rec.update(play(row, rec["angle"], rec["power"]) if drawn else MISS)
        return rec

    @staticmethod
    def _summary(recs: list[dict]) -> dict:
        def mean(key: str, sel: list[dict]) -> float:
            return float(np.mean([float(r[key]) for r in sel])) if sel else float("nan")

        def median(key: str) -> float:
            v = [r[key] for r in recs if r[key] is not None]
            return float(np.median(v)) if v else float("nan")

        hard = [r for r in recs if r["hard"]]
        easy = [r for r in recs if not r["hard"]]
        m = {"n": len(recs), "success": mean("success", recs), "red_potted": mean("red_potted", recs),
             "scratch": mean("scratch", recs), "drawn": mean("drawn", recs), "angle_err_median": median("angle_err"),
             "power_err_median": median("power_err"), "n_hard": len(hard), "success_hard": mean("success", hard),
             "success_easy": mean("success", easy)}
        m["score"] = m["success"] if not math.isnan(m["success"]) else 0.0
        return m


TASK = Billiards
