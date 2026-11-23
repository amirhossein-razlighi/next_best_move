"""KataGo analysis-engine driver: the offline labels of the Go data (never used by the models).

KataGo (https://github.com/lightvector/KataGo, release v1.18.2) runs as `katago analysis` with JSON queries on stdin; a writer
thread keeps at most `inflight` queries outstanding and the responses are matched by id. The binary is $KATAGO (default
`katago` on PATH), the network is given explicitly (the paper's labels: kata1-tf3-b11c768-s11003M-d5973M-7gres from
https://katagotraining.org/networks/).

Every query uses Chinese rules (area scoring, positional superko) and komi 7.5, whatever the game record says (the picture shows
neither), the board 19 x 19 and the full move history of the game (so ko and superko are exact). Winrates and score leads are
from the side to move (analysis.cfg: reportAnalysisWinratesAs = SIDETOMOVE).
  value_query   1 visit: the raw network value of the position (a cheap filter of decided positions before the search)
  label_query   `visits` visits (the paper: 400): label = the root move with the most visits (ties: KataGo's order); every
                root candidate (move, visits, winrate, scoreLead, prior, order, lcb) and the root value are kept
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Callable, Iterable, Mapping

from nbm.tasks.go.rules import N

KATAGO = os.environ.get("KATAGO", "katago")
CONFIG = str(Path(__file__).with_name("analysis.cfg"))
RULES, KOMI = "chinese", 7.5
LABEL_VISITS = 400


def _query(qid: str, row: Mapping) -> dict:
    return {"id": qid, "rules": RULES, "komi": KOMI, "boardXSize": N, "boardYSize": N,
            "moves": [m.split() for m in list(row["moves"])], "analysisPVLen": 1}


def value_query(qid: str, row: Mapping) -> dict:
    return {**_query(qid, row), "maxVisits": 1}


def label_query(qid: str, row: Mapping, visits: int = LABEL_VISITS) -> dict:
    return {**_query(qid, row), "maxVisits": int(visits)}


def parse_label(resp: dict) -> dict:
    """Response of a label query -> {best, best_order0, root_winrate, root_score, root_visits, cands} (or {error})."""
    if "error" in resp:
        return {"error": resp["error"]}
    cands = [{"move": m["move"], "visits": int(m["visits"]), "winrate": float(m["winrate"]), "scoreLead": float(m["scoreLead"]),
              "prior": float(m["prior"]), "order": int(m["order"]), "lcb": float(m.get("lcb", m["winrate"]))} for m in resp["moveInfos"]]
    best = sorted(cands, key=lambda m: (-m["visits"], m["order"]))[0] if cands else None
    root = resp["rootInfo"]
    return {"best": best["move"] if best else None, "best_order0": next((m["move"] for m in cands if m["order"] == 0), None),
            "root_winrate": float(root["winrate"]), "root_score": float(root["scoreLead"]), "root_visits": int(root["visits"]),
            "cands": cands}


def parse_value(resp: dict) -> float | None:
    return float(resp["rootInfo"]["winrate"]) if "rootInfo" in resp else None


class Engine:
    """One `katago analysis` process. settings: maxVisits, numAnalysisThreads, numSearchThreadsPerAnalysisThread,
    nnMaxBatchSize (passed with -override-config; the paper's labels: 400 visits, 32 x 2 threads, batch 64)."""

    def __init__(self, net: str, visits: int, threads: int = 32, search_threads: int = 2, batch: int = 64, log=print):
        self.settings = {"maxVisits": visits, "numAnalysisThreads": threads, "numSearchThreadsPerAnalysisThread": search_threads,
                         "nnMaxBatchSize": batch}
        ov = ",".join(f"{k}={v}" for k, v in self.settings.items())
        self.cmd = [KATAGO, "analysis", "-config", CONFIG, "-model", net, "-override-config", ov]
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.sem = threading.Semaphore(4 * threads)
        self.results: queue.Queue = queue.Queue()
        self.stderr_tail: list[str] = []
        self._log = log
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                self._log(f"katago: non-JSON line {line[:200]}")
                continue
            if r.get("isDuringSearch"):
                continue
            if "id" not in r:  # global warnings / errors
                self._log(f"katago: {line[:300]}")
                continue
            if "warning" in r and "moveInfos" not in r and "error" not in r:  # a per-query warning precedes the response
                self._log(f"katago warning: {line[:300]}")
                continue
            self.sem.release()
            self.results.put(r)
        self.results.put(None)

    def _read_err(self) -> None:
        for line in self.proc.stderr:
            self.stderr_tail = (self.stderr_tail + [line.rstrip()])[-50:]

    def run(self, queries: Iterable[dict], on_result: Callable[[dict], None]) -> int:
        """Send every query, call on_result(response) for each; returns the number of responses."""
        sent: list = [0]

        def writer():
            for q in queries:
                self.sem.acquire()
                self.proc.stdin.write(json.dumps(q) + "\n")
                self.proc.stdin.flush()
                sent[0] += 1
            sent.append(True)

        threading.Thread(target=writer, daemon=True).start()
        got = 0
        while not (len(sent) == 2 and got == sent[0]):
            try:
                r = self.results.get(timeout=600)
            except queue.Empty:
                raise RuntimeError("katago: no response for 600 s; stderr tail:\n" + "\n".join(self.stderr_tail)) from None
            if r is None:
                raise RuntimeError("katago exited; stderr tail:\n" + "\n".join(self.stderr_tail))
            on_result(r)
            got += 1
        return got

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def version() -> str:
    try:
        return subprocess.run([KATAGO, "version"], capture_output=True, text=True, timeout=300).stdout.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"
