# Chess puzzles

Paper: Table 1 and Appendix A.1. Image model (Qwen-Image-2.1 + LoRA) and video model (Wan2.2-TI2V-5B + LoRA). Code:
`nbm/tasks/chess/`, configs: `configs/chess/image.yaml`, `configs/chess/video.yaml`.

## Task

A Lichess puzzle: find the next best move of the side to move. One decision per puzzle, scored on the puzzle's first move.

* **Picture** (`board.py`): the board rendered by python-chess (SVG, cairosvg) at 256 x 256 px, from White's side, no
  coordinates; one square = 32 px.
* **Move** (the mark M(x, a)): the origin square tinted blue and the destination square tinted red (alpha 0.6). Image
  model: input = the board, target = the board with the answer marked; prompt `White to move. Highlight the best move: the
  origin square in blue and the destination square in red.` (or `Black ...`). Video model: a 17-block clip of the puzzle line,
  solver moves and opponent replies, up to 8 plies: block 0 = the board, then per ply a DECIDE block (the board with green
  dots on every legal destination of the moved piece and the move's blue / red tints on top) and an IMAGINE block (the board
  after the move); after the line the final board is held. Prompt `White to move. Show the best continuation: at each step
  highlight the move (origin blue, legal squares green, destination red), then show the board after it.` The decision is
  the first DECIDE block.
* **Readout** (`decode.py`, threshold-free): per square, the least-squares strengths of the blue and red tints relative to
  the input board (inner 76% of the square; for a DECIDE block the green dots are fitted as a third colour and ignored). The
  strongest blue square is the origin, the strongest red square the destination, however faint. If that pair is not a legal
  move the decision is `none` (wrong); for the video model a block with no mark at all (strongest strength 0) is also `none`.
  Moves compare on origin and destination (promotion pieces cannot be drawn).

Metrics: `acc` (the decision equals the puzzle's first move = next-best-move accuracy), `acc_sacrifice` (the 1,001 test
puzzles with the Lichess theme `sacrifice`), `acc_far_from_training` (the 1,757 test puzzles whose nearest training position,
along the first 3,000,000 training puzzles' lines, differs on 13 to 16 of the 64 squares; read from
`far_from_training.txt` in the data directory, shipped with the data download), `legal` (the decision of sample 0 is a legal
move), `acc_rating_<bucket>`; with `--votes n` also `acc_sample0`, `acc_per_sample`, `acc_any_sample`.

## Data

Source: the Lichess puzzle database (<https://database.lichess.org/#puzzles>, CC0, `lichess_db_puzzle.csv.zst`). The paper's
splits come from the snapshot of 9 October 2026 (sha256 `76335bfa7d7c4a7f93c1366d81549e53951ebb79dd43d34904cab8f22d962f8d`);
the database grows every month, so a newer file gives other splits: use `scripts/download_data.sh chess` for the paper's
files, or rebuild them from that snapshot:

```bash
python -m nbm.tasks.chess.data --puzzles lichess_db_puzzle.csv.zst --out $NBM_DATA/chess --workers 16   # ~4 min, ~13 GB RAM
```

The builder (deterministic, seed 0) splits by source game (sha1 of the game id: 3% of the games test, 2% val), removes
duplicate positions (a position in an evaluation split is dropped from train), and draws val (2,000) and test (10,000) with
equal numbers per rating bucket (<1000, 1000-1250, ..., 2500+) among puzzles with rating deviation <= 90 and >= 200 plays.
Train keeps every other puzzle of the training games, shuffled (5,808,734 rows); both models train on its first 3,000,000.
From the 9 October 2026 snapshot it reproduces the paper's three splits row for row.

**Columns**: `puzzle_id`, `fen` (the position to solve, after the opponent's setup move), `side` (`white` / `black`), `line`
(the solution, up to 8 UCI moves: `line[0]` is the answer, then the opponent's reply, ...), `rating`, `rating_bucket`,
`themes`, `game_id`, `pos_key` (board, side, castling, en passant).

## Train

```bash
torchrun --standalone --nproc_per_node 8 -m nbm.train --config configs/chess/image.yaml    # 8 processes x 2 GPUs (2 nodes of 8)
torchrun --nnodes 2 --nproc_per_node 8 ... -m nbm.train --config configs/chess/video.yaml  # 16 processes x 1 GPU
```

Image: rank-64 LoRA, learning rate 1e-4 (cosine, 200 warm-up steps), batch 1 per process (global 8), 375,000 steps = one
pass over 3,000,000 puzzles, batches of one side to move. Video: rank 64, learning rate 1e-4, global batch 16, 187,500 steps
= one pass, flow shift 3. Each took about 4 days on 16 A5000 GPUs. The released chess checkpoints: `chess_image` and
`chess_video` (`scripts/download_checkpoints.sh chess_image chess_video`).

## Evaluate

```bash
I="torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/chess/image.yaml --checkpoint checkpoints/chess_image --split test"
V="torchrun --standalone --nproc_per_node 4 -m nbm.evaluate --config configs/chess/video.yaml --checkpoint checkpoints/chess_video --split test"
$I eval.chunk_rows=10000     # image model, 20 Euler steps (Table 1 "Image model")
$I --votes 8                 # + majority vote over 8 samples
$I --steps 1                 # + one step
$I --steps 1 --votes 8       # + one step + majority vote
$V                           # video model, 30 UniPC steps (and --votes 8, --steps 1, --steps 1 --votes 8 as above)
```

Noise and batches are the paper's, so the records can be compared row by row (outputs depend on the batch composition at
bfloat16 rounding level only; results do not depend on the number of processes):

* image: the rows are cut into chunks of `eval.chunk_rows` (2,500, how the paper ran the vote samples and the one-step
  rows); inside a chunk the white-to-move boards come first, then the black ones, in batches of 8 of one side; batch k of a
  chunk starts from one generator seeded `(seed + j) * 1,000,003 + k` on the DiT GPU, j = the vote sample. Table 1's 20-step
  single-sample row ran the 10,000 boards as one chunk (`eval.chunk_rows=10000`).
* video: chunks of 1,000 rows dealt round-robin into 4 lanes (the paper's 4-GPU layout), batches of 4; row i, sample j has its
  own noise seed `seed * 1,000,003 + i + 10,000,019 j`.

`predictions.jsonl` has one record per puzzle: `samples` (the decision of every vote sample, `none` = illegal / no mark),
`pred` (the majority, ties to the earliest sample), `correct`, `answer`, the subset flags and the strongest blue / red
strengths of sample 0. The first `eval.examples` (8) rows' pictures go to `examples/` (image: board | target | output;
video: the 17 generated blocks). Cost on A5000s: image about 0.45 + 0.135 x steps GPU-seconds per board and sample (10,000
boards at 20 steps: ~9 GPU-hours), video about 0.24 x (9 + steps) (30 steps: ~26 GPU-hours).

## Expected numbers (released checkpoints, 10,000 test puzzles, seed 0, threshold-free readout)

| model | readout | All | Sacrifices | Far from training |
|---|---|---|---|---|
| image | 20 steps | 56.0 | 46.3 | 50.4 |
| image | + majority vote (8) | 63.8 | 54.9 | 58.7 |
| image | + one step | 62.5 | 51.7 | 56.6 |
| image | + one step + majority vote | 63.2 | 53.0 | 57.9 |
| video | 30 steps | 56.4 | 47.9 | 51.0 |
| video | + majority vote (8) | 65.4 | 57.3 | 61.5 |
| video | + one step | 63.9 | 55.6 | 59.1 |
| video | + one step + majority vote | 64.7 | 56.2 | 60.0 |
