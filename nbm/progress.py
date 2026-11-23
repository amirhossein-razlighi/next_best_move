"""One-line progress reports for long loops in log files.

At most one line every NBM_PROGRESS_S seconds (default 60), plus the first update and the end; only the main process of a
distributed job prints (env RANK / LOCAL_RANK). No carriage returns, so the lines read well in Slurm logs:

    [closed loop] 120/2400 5.0% | 2.10 map/s | 0:00:57 elapsed | ETA 15:04 (+0:18:06) | optimal 0.912
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def _every_s() -> float:
    return float(os.environ.get("NBM_PROGRESS_S", 60))


def _hms(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def eta_text(done: float, total: float, elapsed: float) -> str:
    """'ETA 15:04 (+0:18:06)' from the progress so far ('' if unknown)."""
    if not total or done <= 0 or elapsed <= 0:
        return ""
    left = elapsed / done * (total - done)
    t = _dt.datetime.now().astimezone() + _dt.timedelta(seconds=left)
    return f"ETA {t.strftime('%H:%M %Z').strip()} (+{_hms(left)})"


def is_main_process() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) in ("0", "")


class Progress:
    """p = Progress(total=n, desc="eval", unit="row"); p.update(k, acc="0.51"); p.close()"""

    def __init__(self, total: int | None = None, desc: str = "", unit: str = "it", enabled: bool | None = None,
                 every_s: float | None = None):
        self.total, self.desc, self.unit = total, desc, unit
        self.enabled = is_main_process() if enabled is None else enabled
        self.every_s = _every_s() if every_s is None else every_s
        self.n = 0
        self.t0 = self.last = time.time()
        self.metrics: dict[str, object] = {}
        self.closed = False

    def update(self, n: int = 1, **metrics: object) -> None:
        first = self.n == 0
        self.n += n
        self.metrics.update(metrics)
        now = time.time()
        if self.enabled and (first or now - self.last >= self.every_s):
            self._emit(now)

    def _emit(self, now: float, final: bool = False) -> None:
        self.last = now
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        parts = [f"[{self.desc}] {self.n}" + (f"/{self.total} {100 * self.n / self.total:.1f}%" if self.total else ""),
                 f"{rate:.3g} {self.unit}/s" if rate >= 0.01 or not rate else f"{1 / rate:.3g} s/{self.unit}",
                 f"{_hms(el)} elapsed"]
        if final:
            parts.append("done")
        elif self.total:
            e = eta_text(self.n, self.total, el)
            if e:
                parts.append(e)
        parts += [f"{k} {v}" for k, v in self.metrics.items()]
        print(" | ".join(parts), file=sys.stdout, flush=True)

    def close(self) -> None:
        if self.enabled and not self.closed:
            self._emit(time.time(), final=True)
        self.closed = True

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def track(items: Iterable[T], total: int | None = None, desc: str = "", unit: str = "it",
          enabled: bool | None = None) -> Iterator[T]:
    """for x in track(batches, desc="generate", unit="batch"): ..."""
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    p = Progress(total=total, desc=desc, unit=unit, enabled=enabled)
    try:
        for x in items:
            yield x
            p.update(1)
    finally:
        p.close()
