"""The two pathfinding tasks as nbm Tasks: data rows -> training pictures / clips, and the evaluation (shared code).

Data: <data.dir>/<split>.parquet built by nbm/tasks/pathfinding/data.py (one row = one map with its target route).
Image model (Qwen-Image-2.1): condition = a state along the target route, target = the same + the red disc on the next cell.
In training the state is drawn per access: k ~ U{0 .. len - 1} from random.Random(f"{row seed}:{rng.randint(2**31 - 1)}"), i.e.
every state a closed-loop solve visits; the highlight box is the next cell.
Video model (Wan2.2-TI2V-5B): one clip per training row, the row's seg_di segment of the target route (render.segment_blocks).
One fixed, neutral prompt for both models and every picture: "Continue." (the rules are never told).
"""
from __future__ import annotations

import random

import pandas as pd
from PIL import Image

from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, VideoExample
from nbm.tasks.pathfinding import evaluation as E
from nbm.tasks.pathfinding import render as R

PROMPT = "Continue."


class Pathfinding(Task):
    """Shared base; subclasses set terrain, the split names and the split aliases."""

    families = ("image", "video")
    terrain: bool = False
    split_names: tuple[str, ...] = ()
    aliases: dict[str, str] = {}

    def load_split(self, split: str) -> pd.DataFrame:
        name = self.aliases.get(split, split)
        if name not in self.split_names:
            raise ValueError(f"unknown split {split!r}; this task has {list(self.split_names)} (aliases {self.aliases})")
        df = pd.read_parquet(self.data_dir / f"{name}.parquet")
        if len(df) and bool(df.terrain.iloc[0]) != self.terrain:
            raise ValueError(f"{self.data_dir}/{name}.parquet holds {'terrain' if df.terrain.iloc[0] else 'uniform'} maps; "
                             f"task {self.cfg.task} expects {'terrain' if self.terrain else 'uniform'} maps (check data.dir)")
        return df

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        """rng None: the start state (k = 0); else a random state along the target route (module doc)."""
        m, path = E.target(row)
        k = 0 if rng is None else random.Random(f"{row['seed']}:{rng.randint(2 ** 31 - 1)}").randrange(int(row["len"]))
        cond, tgt = R.state_picture(m, path, k), R.decide_picture(m, path, k)
        return ImageExample(Image.fromarray(cond), Image.fromarray(tgt), PROMPT, [R.cell_box(*path[k + 1], m["n"])])

    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample:
        m, path = E.target(row)
        return VideoExample(R.segment_blocks(m, path, int(row["seg_di"])), PROMPT)

    def prompts(self) -> list[str]:
        return [PROMPT]

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        """Full solves (closed loop / clip execution); in training the quicker validation (nbm.tasks.pathfinding.evaluation)."""
        if getattr(ctx, "in_training", False):
            return E.first_moves(ctx, df, PROMPT) if ctx.family == "image" else E.teacher_forced_clips(ctx, df, PROMPT)
        return E.closed_loop(ctx, df, PROMPT)
