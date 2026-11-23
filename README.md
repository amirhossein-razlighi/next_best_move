# Learning the Next Best Move with Visual Diffusion Models

**Amir Razlighi, Ali Mahdavi Amiri, Yiorgos Chrysanthou, Andrey Voynov, Daniel Cohen-Or**

[Project page](https://amirhossein-razlighi.github.io/next_best_move/) · [Paper (arXiv, soon)](#) · [Weights and data (Hugging Face, soon)](#)

![Our model solves a chess puzzle one move at a time, including a rook sacrifice, and finds the cheapest route across terrain whose costs are never shown.](docs/assets/teaser.png)

A pretrained visual diffusion model, fine-tuned on pictures of states and their next best moves, learns to choose moves whose
value depends on consequences several steps ahead, with no search at inference. The state is an image, the action is an edit
of that image (blue marks where a piece leaves, red where it lands), and the move is read back from the strongest marks. The
image model (Qwen-Image-2.1) predicts one move per call; the video model (Wan2.2-TI2V-5B) draws several moves and the states
they lead to in one clip. Repeating the prediction solves a puzzle, a route or a game one decision at a time.

This repository has everything needed to build the data, train the models and evaluate them on the paper's tasks: chess
puzzles, shortest paths, terrain with hidden costs, Connect Four, Go, billiards and N-queens.

## Install

Python 3.11 and CUDA GPUs (the paper used 24 GB A5000s).

```bash
git clone https://github.com/amirhossein-razlighi/next_best_move.git && cd next_best_move
uv sync && source .venv/bin/activate      # pinned versions, incl. torch (cu126) and diffusers at the tested revision
```

Without `uv`: install `torch==2.14.1` from the cu126 index, then `pip install -e .`.

Three environment variables say where things live (paths in the configs are relative to them):

| variable | default | content |
|---|---|---|
| `NBM_MODELS` | `./models` | the two base models |
| `NBM_DATA` | `./data` | one folder per task |
| `NBM_OUTPUTS` | `./outputs` | training runs and evaluations |

## Quick start: evaluate a released model

```bash
scripts/download_base_models.sh image                 # Qwen-Image-2.1 (use `video` for Wan2.2-TI2V-5B, or `all`)
scripts/download_checkpoints.sh chess_image           # -> checkpoints/chess_image
scripts/download_data.sh chess                        # -> data/chess  (or build it: docs/tasks/chess.md)

torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/chess/image.yaml \
    --checkpoint checkpoints/chess_image --split test --steps 1 --limit 200
```

This prints the next-best-move accuracy and writes `metrics.json`, `predictions.jsonl` (one record per puzzle) and a few
example pictures under `$NBM_OUTPUTS/eval/`. Leave out `--limit` for the full test set.

Every evaluation uses the paper's readout: the strongest blue and red marks define the move, and a move that cannot be played
counts as a failure. The options select how the model is sampled:

| option | readout |
|---|---|
| (default) | full sampling: 20 Euler steps (image model), 30 UniPC steps (video model) |
| `--steps 1` | the first denoising step only |
| `--votes n` | majority vote over n samples |

An image-model process uses two GPUs (text encoder and VAE on one, the transformer on the other), a video-model process one.
Results do not depend on the number of processes.

## Results

Released checkpoints, seed 0. The commands for every row are in the task pages.

| task | metric (test set) | image | image, one step | video | video, one step | details |
|---|---|---|---|---|---|---|
| chess | next best move (10,000 puzzles) | 56.0 | 62.5 | 56.4 | 63.9 | [chess](docs/tasks/chess.md) |
| shortest path | optimal route, held-out / longer / larger maps | 88.8 / 47.0 / 79.0 | | 94.6 / 66.6 / 79.8 | | [pathfinding](docs/tasks/pathfinding.md) |
| terrain (costs never shown) | cheapest route | 87.5 | | 82.8 | | [pathfinding](docs/tasks/pathfinding.md) |
| Connect Four | keeps the win (hard positions) | 67.5 | 76.4 | 64.5 | 72.0 | [connect4](docs/tasks/connect4.md) |
| Go (19 x 19) | KataGo's move | 20.9 | 31.1 | | | [go](docs/tasks/go.md) |
| billiards | ball potted | 18.6 | 28.0 | 34.6 | 47.9 | [billiards](docs/tasks/billiards.md) |
| N-queens | next queen keeps the board solvable | 78.8 | 85.2 | | | [nqueens](docs/tasks/nqueens.md) |

With a majority vote over 8 samples, chess reaches 63.8 (image) and 65.4 (video).

## Train

```bash
torchrun --standalone --nproc_per_node <P> -m nbm.train --config configs/<task>/<image|video>.yaml [key.path=value ...]
```

Each config holds every hyper-parameter of the paper's model; `key.path=value` overrides any of them. The global batch is
`train.batch_size x train.grad_accum x` the number of processes, and the task pages give the launch that matches the paper.
A run writes `checkpoint-<step>/` (the last two are kept), a validation log, TensorBoard logs and a progress line to
`$NBM_OUTPUTS/<train.output_dir>`; `train.resume=latest` continues it. Slurm templates are in `examples/slurm/`.

## Tasks and data

| task | config | data | external tools for building the data |
|---|---|---|---|
| chess puzzles | `configs/chess/` | Lichess puzzle database | none |
| shortest path, terrain | `configs/shortest_path/`, `configs/terrain/` | generated | none |
| Connect Four | `configs/connect4/` | generated, solver labels | Pascal Pons' solver (fetched by a build script) |
| Go | `configs/go/` | KGS games, KataGo labels | KataGo |
| billiards | `configs/billiards/` | generated, simulator labels | none (the simulator is included) |
| N-queens | `configs/nqueens/` | generated | none |

Each task page in `docs/tasks/` describes the task, the data format, how to download or build the data, and the training and
evaluation commands with the expected numbers.

## Repository

```
nbm/models/        the two backbones: loading, LoRA, flow-matching loss, samplers
nbm/train.py       training               nbm/evaluate.py   evaluation
nbm/marks.py       drawing marks and measuring their strength      nbm/readout.py   strongest-mark readout, majority vote
nbm/tasks/         one package per task; base.py is the Task interface, README.md explains how to add a task
configs/           one YAML file per model           docs/tasks/   one page per task
scripts/           downloads                         tests/        CPU tests
```

Tests run on CPU without model weights: `python -m pytest -q`.

To add your own task, implement the Task interface in `nbm/tasks/<name>/` (see `nbm/tasks/README.md` and the toy task in
`nbm/tasks/_example/`) and write a config; training and evaluation then work unchanged.

## Citation

```bibtex
@article{razlighi2026nextbestmove,
  title   = {Learning the Next Best Move with Visual Diffusion Models},
  author  = {Razlighi, Amir and Mahdavi Amiri, Ali and Chrysanthou, Yiorgos and Voynov, Andrey and Cohen-Or, Daniel},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License and acknowledgements

The code is released under the [MIT License](LICENSE). The base models keep their own licenses (Qwen-Image-2.1:
qwen-research; Wan2.2: Apache-2.0). The chess puzzles come from the [Lichess puzzle database](https://database.lichess.org/#puzzles) (CC0), Go games from the KGS archive
and Go labels from [KataGo](https://github.com/lightvector/KataGo). Connect Four labels use
[Pascal Pons' solver](https://github.com/PascalPons/connect4) (AGPL-3.0), which the build script downloads; it is not part of
this repository, and the batch driver compiled against it is covered by the solver's license.
