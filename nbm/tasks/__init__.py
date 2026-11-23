"""Task registry: a task is a package nbm/tasks/<name>/ whose __init__.py sets TASK = <its nbm.tasks.base.Task subclass>.

Discovery is by directory, so adding a task never edits a shared file. Packages whose name starts with "_" (e.g. the toy
example used by the tests) can be loaded by name but are not listed. A package may also hold several tasks as modules that
each set TASK (e.g. "pathfinding.shortest_path" and "pathfinding.terrain"). Any other dotted name ("mypkg.mytask") loads a task
module from outside this package.
"""
from __future__ import annotations

import importlib
from pathlib import Path

_HERE = Path(__file__).parent


def available_tasks() -> list[str]:
    """Names of the tasks in this directory (without the "_"-prefixed ones): packages that set TASK in __init__.py, and the
    modules of a package that set TASK themselves."""
    names = []
    for p in sorted(_HERE.iterdir()):
        if not p.is_dir() or p.name.startswith("_") or not (p / "__init__.py").exists():
            continue
        if "\nTASK = " in (p / "__init__.py").read_text():
            names.append(p.name)
        names += [f"{p.name}.{m.stem}" for m in sorted(p.glob("*.py"))
                  if not m.stem.startswith("_") and "\nTASK = " in m.read_text()]
    return names


def get_task(name: str) -> type:
    """The Task subclass of a task name: nbm.tasks.<name> (a package or a module inside one), else a dotted module path."""
    candidates = [f"nbm.tasks.{name}"] + ([name] if "." in name else [])
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name is not None and module_name.startswith(e.name):  # this candidate (or its package) does not exist
                continue
            raise
        if not hasattr(module, "TASK"):
            raise ValueError(f"{module_name} does not define TASK (the Task subclass)")
        return module.TASK
    raise ValueError(f"unknown task {name!r}; available: {available_tasks()}")


def make_task(cfg):
    """The task object of a config (cfg.task names it)."""
    return get_task(cfg.task)(cfg)
