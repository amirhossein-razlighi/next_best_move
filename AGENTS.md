# AGENTS.md

Guide for coding agents working in this repository (Claude Code, Codex, Cursor, ...). Humans: start with `README.md`.

## What this is

Code for *Learning the Next Best Move with Visual Diffusion Models*. A LoRA-fine-tuned image model (Qwen-Image-2.1) or video
model (Wan2.2-TI2V-5B) draws the next move into a picture of the state; the move is read back from the strongest marks.
Tasks: chess, shortest path, terrain, Connect Four, Go, billiards, N-queens. One package, `nbm`.

## Commands

```bash
uv sync && source .venv/bin/activate                     # environment (Python 3.11, CUDA)
python -m pytest -q                                      # CPU tests, no weights needed; run after every change
torchrun --standalone --nproc_per_node <P> -m nbm.evaluate --config configs/<task>/<image|video>.yaml \
    --checkpoint <dir> --split <split> [--steps 1] [--votes n] [--limit N]
torchrun --standalone --nproc_per_node <P> -m nbm.train --config configs/<task>/<image|video>.yaml [key.path=value ...]
python -m nbm.tasks.<task>.data --help                   # each task's data builder
```

Task names: `chess`, `pathfinding.shortest_path`, `pathfinding.terrain`, `connect4`, `go`, `billiards`, `nqueens`
(`nbm.tasks.available_tasks()` lists them). An image-model process needs 2 GPUs (`model.split_devices: true`), a video-model
process 1. Paths come from `NBM_MODELS`, `NBM_DATA`, `NBM_OUTPUTS` (defaults `./models`, `./data`, `./outputs`).

## Layout

- `nbm/models/` backbones (loading, LoRA, loss, samplers) · `nbm/train.py` · `nbm/evaluate.py` · `nbm/marks.py` (drawing
  marks, measuring their strength) · `nbm/readout.py` (strongest-mark readout, voting) · `nbm/data.py` (samplers, datasets).
- `nbm/tasks/<task>/`: everything task-specific (rules, rendering, data builder, decoding, evaluation loop, metrics).
  `nbm/tasks/base.py` is the interface; `nbm/tasks/README.md` explains it; `nbm/tasks/_example/` is a minimal toy task.
- `configs/<task>/<image|video>.yaml`: one per model. `model` and `train` keys are read by the core and all required; task
  settings go under `data`, `eval` or `options`. YAML floats need a dot (`1.0e-4`).
- `docs/tasks/<task>.md`: task description, data, commands, expected numbers. `tests/test_<task>.py`: CPU tests.

## Adding a task

Create `nbm/tasks/<name>/__init__.py` with a `Task` subclass assigned to `TASK`, a config in `configs/<name>/`, a page in
`docs/tasks/` and a test file. The registry finds the package by its directory, so no shared file changes. Follow the toy
task in `nbm/tasks/_example/`.

## Rules

- **Numerics are fixed.** The released checkpoints and the documented numbers depend on the exact operations, dtypes, noise
  seeds and samplers in `nbm/models/` and on each task's rendering and decoding. Do not refactor them for style; any change
  there must leave the outputs of the released checkpoints unchanged.
- Keep modules short, with docstrings that say what and why, and type hints on public functions. No dead options.
- Tests must stay CPU-only and fast (under 2 minutes per file).
