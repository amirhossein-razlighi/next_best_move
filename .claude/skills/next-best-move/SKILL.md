---
name: next-best-move
description: Evaluate, train or extend the next-best-move diffusion models of this repository (chess, pathfinding, Connect Four, Go, billiards, N-queens). Use when asked to run an evaluation, reproduce a paper number, train a model, build a dataset or add a new task.
---

Read `AGENTS.md` first; it has the commands, the layout and the rules.

1. **Reproduce a number:** find the row in `docs/tasks/<task>.md` ("Expected numbers"), download what it needs
   (`scripts/download_base_models.sh`, `scripts/download_checkpoints.sh <task>_<image|video>`, `scripts/download_data.sh <task>`),
   run the listed `nbm.evaluate` command, and compare against `metrics.json` in the printed output folder. Try `--limit 32` first.
2. **Train:** use the config unchanged unless asked; match the global batch given in the task page
   (`train.batch_size x train.grad_accum x processes`); resume with `train.resume=latest`.
3. **New task:** copy the structure of `nbm/tasks/_example/`, implement the interface in `nbm/tasks/base.py`
   (guide: `nbm/tasks/README.md`), add `configs/<name>/`, `docs/tasks/<name>.md` and `tests/test_<name>.py`, then run
   `python -m pytest -q`.
4. Never change `nbm/models/` numerics or an existing task's rendering or decoding for style.
