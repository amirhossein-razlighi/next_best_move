"""The paper's threshold-free readout and majority vote.

Threshold-free readout: the strongest mark of each colour defines the action (the strongest blue region is where the action
starts, the strongest red region where it ends), whatever its strength; there is no "nothing drawn" outcome. A task's validity
rules are applied afterwards to that decision (e.g. an illegal origin / destination pair is a wrong answer), never used to
pick a different mark.

Majority vote: n samples of the same input (different noise), each read out as above; the most frequent decision wins, ties
go to the decision seen first (the earliest sample).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Hashable, Iterable, Sequence, TypeVar

import numpy as np

D = TypeVar("D")


def ranked(scores: Sequence[float] | np.ndarray, candidates: Iterable[int] | None = None) -> list[int]:
    """Candidate indices sorted by decreasing score; ties keep the candidates' order (stable)."""
    s = np.asarray(scores)
    cand = list(range(len(s))) if candidates is None else list(candidates)
    order = np.argsort(-s[cand], kind="stable")
    return [cand[i] for i in order]


def strongest(scores: Sequence[float] | np.ndarray, candidates: Iterable[int] | None = None) -> int:
    """Index of the strongest mark (the first one on ties), optionally among `candidates` only."""
    return ranked(scores, candidates)[0]


def majority_vote(decisions: Iterable[D | None], key: Callable[[D], Hashable] | None = None) -> D | None:
    """Most frequent decision (None entries ignored; ties -> the earliest). key: compare decisions by key(d), e.g. a move's
    (origin, destination) without the promotion piece; the returned decision is the first sample with the winning key."""
    first: dict[Hashable, D] = {}
    counts: Counter = Counter()
    for d in decisions:
        if d is None:
            continue
        k = d if key is None else key(d)
        first.setdefault(k, d)
        counts[k] += 1
    if not counts:
        return None
    return first[counts.most_common(1)[0][0]]
