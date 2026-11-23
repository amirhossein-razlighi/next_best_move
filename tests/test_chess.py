"""Fast CPU tests of the chess task: rendering -> marks -> threshold-free decoding round trips, the clip layout, the data builder
on a small synthetic puzzle file, the evaluation batching and the evaluation loop with perfect fake models."""
import hashlib
import random

import chess
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from nbm import marks
from nbm.config import load_config
from nbm.models.video import blocks_to_frames
from nbm.tasks import make_task
from nbm.tasks.base import AccessRNG, EvalContext
from nbm.tasks.chess import board as B
from nbm.tasks.chess import data as DB
from nbm.tasks.chess import decode as D
from nbm.tasks.chess.evaluate import image_batches, video_batches

SIZE = 256
# (fen, line): a white and a black puzzle (line[0] = the answer, then the replies), and a promotion
PUZZLES = [
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", ["h5f7"]),
    ("6k1/5ppp/8/8/8/8/5PPP/3r2K1 b - - 0 1", ["d1e1"]),
    ("8/P6k/8/8/8/8/6K1/8 w - - 0 1", ["a7a8q", "h7g6", "a8b8", "g6f5"]),
]


def random_puzzles(n: int, seed: int = 0) -> pd.DataFrame:
    """Lichess-format rows from seeded random games: FEN before the opponent's move, Moves = that move + 3 more."""
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        b = chess.Board()
        for _ in range(rng.randrange(4, 30)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        fen, moves = b.fen(), []
        for _ in range(4):
            if b.is_game_over():
                break
            m = rng.choice(list(b.legal_moves))
            moves.append(m.uci())
            b.push(m)
        if len(moves) < 2:
            continue
        i = len(rows)
        rows.append({"PuzzleId": f"p{i:04d}", "FEN": fen, "Moves": " ".join(moves), "Rating": rng.randrange(600, 2900),
                     "RatingDeviation": 75, "Popularity": 90, "NbPlays": 500, "Themes": rng.choice(["mate", "sacrifice fork"]),
                     "GameUrl": f"https://lichess.org/g{i % 97:07d}#12"})
    return pd.DataFrame(rows)


def cfg(family, tmp_path=None):
    c = load_config(f"configs/chess/{family}.yaml")
    if tmp_path is not None:
        c.data.dir = str(tmp_path)
    return c


# ------------------------------------------------------------------ pictures and decoding
@pytest.mark.parametrize("fen,line", PUZZLES)
def test_image_round_trip(fen, line):
    cond, target = B.image_pair(fen, line[0], SIZE)
    assert cond.size == target.size == (SIZE, SIZE)
    d = D.read_image_move(target, fen, cond)
    assert D.same_move(d["move"], line[0]) and abs(d["blue_peak"] - 1) < 0.02 and abs(d["red_peak"] - 1) < 0.02
    # a faint mark is still the strongest one (threshold-free), small noise does not change it
    faint = np.asarray(cond, np.float32) + 0.2 * (np.asarray(target, np.float32) - np.asarray(cond, np.float32))
    faint += np.random.default_rng(0).normal(0, 3, faint.shape)
    d = D.read_image_move(Image.fromarray(faint.round().clip(0, 255).astype(np.uint8)), fen, cond)
    assert D.same_move(d["move"], line[0]) and d["red_peak"] < 0.5


def test_illegal_or_missing_mark_is_no_move():
    fen = PUZZLES[0][0]
    cond = B.render_board(chess.Board(fen), SIZE)
    bad = marks.tint(cond, [(B.square_box(chess.E2, SIZE), marks.BLUE), (B.square_box(chess.E5, SIZE), marks.RED)])
    assert D.read_image_move(bad, fen, cond)["move"] == D.NO_MOVE  # e2 -> e5 is not a legal move
    assert D.read_clip_move(cond, fen, cond)["move"] == D.NO_MOVE  # nothing drawn at all
    assert not D.same_move(D.NO_MOVE, "e2e4") and D.same_move("a7a8", "a7a8q")


@pytest.mark.parametrize("fen,line", PUZZLES)
def test_clip_layout_and_decide_block(fen, line):
    blocks = B.clip_blocks(fen, line, SIZE)
    assert len(blocks) == 17
    board = chess.Board(fen)
    assert np.array_equal(np.asarray(blocks[0]), np.asarray(B.render_board(board, SIZE)))
    d = D.read_clip_move(blocks[1], fen, blocks[0])
    assert D.same_move(d["move"], line[0])
    for k, u in enumerate(line, start=1):  # every DECIDE block reads back its ply, every IMAGINE block is the board after it
        assert D.same_move(D.read_clip_move(blocks[2 * k - 1], board.fen())["move"], u)
        board.push_uci(u)
        assert np.array_equal(np.asarray(blocks[2 * k]), np.asarray(B.render_board(board, SIZE)))
    for b in blocks[2 * len(line) + 1:]:
        assert np.array_equal(np.asarray(b), np.asarray(blocks[2 * len(line)]))
    # the dots: one on every legal destination of the moved piece
    mv = chess.Move.from_uci(line[0])
    dests = B.legal_destinations(chess.Board(fen), mv.from_square)
    dot = B.draw_dots(blocks[0], dests)
    assert (np.asarray(dot) != np.asarray(blocks[0])).any(-1).sum() > 0
    assert B.video_prompt("black").startswith("Black to move. Show the best continuation")


# ------------------------------------------------------------------ data
def test_data_builder(tmp_path):
    src = tmp_path / "puzzles.csv"
    random_puzzles(400).to_csv(src, index=False)
    stats = DB.build(src, tmp_path / "chess", val_frac=0.25, test_frac=0.25, n_val=16, n_test=16, workers=2)
    sp = {s: pd.read_parquet(tmp_path / "chess" / f"{s}.parquet") for s in ("train", "val", "test")}
    assert list(sp["train"].columns) == DB.COLUMNS and stats["splits"]["test"]["rows"] == len(sp["test"]) > 0
    assert not set(sp["train"].game_id) & (set(sp["val"].game_id) | set(sp["test"].game_id))
    assert not set(sp["train"].pos_key) & set(sp["test"].pos_key)
    for r in sp["test"].itertuples():
        b = chess.Board(r.fen)
        assert ("white" if b.turn else "black") == r.side and 1 <= len(r.line) <= 8
        for u in r.line:
            b.push_uci(u)
    DB.build(src, tmp_path / "again", val_frac=0.25, test_frac=0.25, n_val=16, n_test=16, workers=1)
    for s in sp:  # deterministic
        h = lambda p: hashlib.sha256(pd.read_parquet(p).to_csv().encode()).hexdigest()  # noqa: E731
        assert h(tmp_path / "chess" / f"{s}.parquet") == h(tmp_path / "again" / f"{s}.parquet")


def test_task_examples_and_splits(tmp_path):
    random_puzzles(400).to_csv(tmp_path / "p.csv", index=False)
    DB.build(tmp_path / "p.csv", tmp_path, val_frac=0.25, test_frac=0.25, n_val=16, n_test=16, workers=1)
    (tmp_path / "far_from_training.txt").write_text(pd.read_parquet(tmp_path / "test.parquet").puzzle_id[0] + "\n")
    task = make_task(cfg("image", tmp_path))
    test = task.load_split("test")
    assert test.sacrifice.dtype == bool and test.far_from_training.sum() == 1
    train = task.load_split("train")
    assert list(train.columns) == ["puzzle_id", "fen", "side", "line"]
    assert task.batch_groups(train).tolist() == [int(s == "black") for s in train.side]
    row = train.iloc[0].to_dict()
    ex = task.image_example(row, AccessRNG(torch.Generator().manual_seed(0)))
    mv = chess.Move.from_uci(row["line"][0])
    assert ex.prompt.startswith(row["side"].capitalize() + " to move. Highlight the best move")
    assert list(ex.boxes) == [B.square_box(mv.from_square, SIZE), B.square_box(mv.to_square, SIZE)]
    vtask = make_task(cfg("video", tmp_path))
    vex = vtask.video_example(row, None)
    assert vex.blocks.shape == (17, SIZE, SIZE, 3) and vex.blocks.dtype == np.uint8 and vex.prompt in vtask.prompts()


# ------------------------------------------------------------------ evaluation
def test_batch_layouts():
    sides = np.array(["white", "black", "white", "white", "black", "black", "white"])
    b = image_batches(sides, chunk_rows=4, bs=2)
    assert b == [(0, [0, 2]), (1, [3]), (2, [1]), (0, [6]), (1, [4, 5])]  # white first per chunk, k restarts per chunk
    v = video_batches(10, chunk_rows=6, lanes=4, bs=2)
    assert v == [[0, 4], [1, 5], [2], [3], [6], [7], [8], [9]]  # lane r = rows r, r + 4, ... of the chunk


class FakeImageModel:
    """A perfect image model: draws the answer of the board it is given."""
    device = "cpu"

    def __init__(self, df):
        self.targets = {B.render_board(chess.Board(f), SIZE).tobytes(): B.image_pair(f, l[0], SIZE)[1] for f, l in zip(df.fen, df.line)}

    def latent_shape(self, w, h):
        return (4, 1, h // 8, w // 8)

    def generate(self, conds, prompts, steps, noise):
        assert noise.shape[0] == len(conds)
        return [self.targets[c.tobytes()] for c in conds]


class FakeVideoModel:
    def __init__(self, df):
        self.clips = {B.render_board(chess.Board(f), SIZE).tobytes(): B.clip_blocks(f, l, SIZE) for f, l in zip(df.fen, df.line)}

    def generate_seeded(self, firsts, prompts, steps, shift, seeds):
        clips = [blocks_to_frames(self.clips[f.tobytes()]) for f in firsts]
        return torch.from_numpy(np.stack([np.stack([np.asarray(x) for x in c]) for c in clips]))


@pytest.mark.parametrize("family", ["image", "video"])
def test_evaluate_with_fake_model(family, tmp_path):
    random_puzzles(400).to_csv(tmp_path / "p.csv", index=False)
    DB.build(tmp_path / "p.csv", tmp_path, val_frac=0.25, test_frac=0.25, n_val=16, n_test=16, workers=1)
    c = cfg(family, tmp_path)
    task = make_task(c)
    df = task.load_split("test").head(10).reset_index(drop=True)
    model = FakeImageModel(df) if family == "image" else FakeVideoModel(df)
    ctx = EvalContext(c, model, "test", steps=1, votes=2, seed=0, batch_size=3, out_dir=tmp_path / "out")
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] == 1.0 and res.metrics["n"] == 10 and res.metrics["legal"] == 1.0
    assert all(r["samples"] == [r["answer"][:4]] * 2 and r["correct"] for r in res.records)
    assert res.metrics["acc_any_sample"] == 1.0 and "acc_sacrifice" in res.metrics
    assert len(list((tmp_path / "out" / "examples").glob("*.png"))) == 8
