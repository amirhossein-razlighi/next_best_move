"""Chess: find the next best move of a Lichess puzzle (paper Table 1).

Image model (Qwen-Image-2.1): board -> the same board with the answer's origin square tinted blue and its destination red.
Video model (Wan2.2-TI2V-5B): board -> a 17-block clip that decides (blue / green dots / red) and imagines (the board after)
every ply of the puzzle line, opponent replies included (nbm/tasks/chess/board.py). Both are scored on the first move only,
with the threshold-free readout (nbm/tasks/chess/decode.py): next-best-move accuracy, plus the "sacrifice" puzzles (Lichess
theme) and, if the data directory holds far_from_training.txt, the "far from training" puzzles.

Data: $NBM_DATA/<data.dir>/{train,val,test}.parquet from nbm/tasks/chess/data.py. Evaluation: nbm/tasks/chess/evaluate.py.
"""
from __future__ import annotations

import chess
import numpy as np
import pandas as pd

from nbm.tasks.base import AccessRNG, EvalContext, EvalResult, ImageExample, Task, VideoExample
from nbm.tasks.chess import board as B
from nbm.tasks.chess.evaluate import evaluate_image, evaluate_video

FAR_FILE = "far_from_training.txt"  # optional: puzzle ids of the "far from training" test subset, one per line


class Chess(Task):
    families = ("image", "video")

    def load_split(self, split: str) -> pd.DataFrame:
        """Rows of <data_dir>/<split>.parquet in file order, plus the subset flags `sacrifice` and (when FAR_FILE lists
        puzzles of the split) `far_from_training`. The training split is read with the columns training needs only."""
        path = self.data_dir / f"{split}.parquet"
        if split == "train":
            return pd.read_parquet(path, columns=["puzzle_id", "fen", "side", "line"])
        df = pd.read_parquet(path)
        df["line"] = [list(x) for x in df.line]
        df["sacrifice"] = df.themes.str.split().apply(lambda t: "sacrifice" in t)
        far = self.data_dir / FAR_FILE
        if far.exists():
            flag = df.puzzle_id.isin(set(far.read_text().split()))
            if flag.any():  # the listed puzzles are test puzzles: other splits get no column
                df["far_from_training"] = flag
        return df

    def batch_groups(self, df: pd.DataFrame) -> np.ndarray:
        """0 = white to move, 1 = black (the image prompt's token layout depends on the side)."""
        return (df.side.values == "black").astype(np.int64)

    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        """The board and the board with the puzzle answer marked; highlight region = the move's two squares."""
        move = chess.Move.from_uci(row["line"][0])
        cond, target = B.image_pair(row["fen"], row["line"][0], self.resolution)
        boxes = [B.square_box(move.from_square, self.resolution), B.square_box(move.to_square, self.resolution)]
        return ImageExample(cond, target, B.image_prompt(row["side"]), boxes)

    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample:
        """The decide / imagine clip of the puzzle line (up to 8 plies)."""
        blocks = B.clip_blocks(row["fen"], list(row["line"]), self.resolution)
        return VideoExample(np.stack([np.asarray(b) for b in blocks]), B.video_prompt(row["side"]))

    def prompts(self) -> list[str]:
        return [B.video_prompt("white"), B.video_prompt("black")]

    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        return evaluate_image(ctx, df) if ctx.family == "image" else evaluate_video(ctx, df)


TASK = Chess
