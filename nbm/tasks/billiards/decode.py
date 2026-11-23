"""Reading a shot back from a generated picture: the paper's threshold-free billiards readout (rule-free, no simulator).

Colour model: per pixel, the least-squares alpha of CYAN (aim line, power bar) and of MAGENTA (cue stick) over the input picture,
out = (1 - a) x + a colour (`mark_alpha`), clipped to [-1, 1.5] (no positive bias on unmarked pixels). There is no pixel
threshold, no minimum line length and no bar threshold: the shot whose rendered mark best matches the output wins.

  angle  = argmax over angles of the normalised correlation sum(w a) / sqrt(sum(w^2)) between the alpha maps and the mark of that
           angle as render.render_f draws it (w = the antialiased 4-px line profile, sampled every 1 px along and 0.5 px across):
           the aim line from 2 px outside the cue ball to the cushion on the CYAN alpha (pixels of balls in the input skipped: the
           line is drawn under the balls), plus the stick behind the cue ball on the MAGENTA alpha; rows over the power track
           excluded as in render_f. Search: centre-line score on a 0.25 deg grid -> the 5 best peaks more than 1 deg apart -> full
           template on +- 1 deg at 0.02 deg -> +- 0.02 deg at 0.002 deg.
  power  = k / 464 with k = argmax_k S_k / sqrt(k), S_k = summed bar alpha of the first k track columns (the bar fill whose
           template fits best at any amplitude; a crisp bar ending at column K gives K).
drawn = both best scores > 0, i.e. the output carries some positive cyan / magenta evidence at all; otherwise there is no shot
(a miss). On clean renders the angle error has a median of about 0.003 deg.
"""
from __future__ import annotations

import math

import numpy as np

from nbm.tasks.billiards import physics as P
from nbm.tasks.billiards import render as RD

_BASE_U8: np.ndarray | None = None
GRID = np.arange(1440) * 0.25          # coarse angle grid (deg)
N_PEAKS, PEAK_SEP = 5, 1.0             # coarse peaks refined, minimal separation (deg)


def mark_alpha(img: np.ndarray, ref: np.ndarray, col) -> np.ndarray:
    """Least-squares alpha of colour `col` over `ref` per pixel (img = (1 - a) ref + a col); 0 where col is close to ref."""
    f, r = np.asarray(img, np.float32)[..., :3], np.asarray(ref, np.float32)[..., :3]
    dv = np.asarray(col, np.float32) - r
    den = (dv * dv).sum(-1)
    return np.where(den > 300.0, ((f - r) * dv).sum(-1) / np.maximum(den, 1e-6), 0.0)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2-D array at pixel-centre coordinates (x, y) (pixel k covers [k, k + 1)); points must be inside."""
    fx, fy = x - 0.5, y - 0.5
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, img.shape[0] - 2)
    wx, wy = np.clip(fx - x0, 0.0, 1.0), np.clip(fy - y0, 0.0, 1.0)
    return ((1 - wy) * ((1 - wx) * img[y0, x0] + wx * img[y0, x0 + 1]) + wy * ((1 - wx) * img[y0 + 1, x0] + wx * img[y0 + 1, x0 + 1]))


def read_shot(img: np.ndarray, ref: np.ndarray, cue_xy) -> dict:
    """The shot drawn in `img` (uint8 (CH, CW, 3): an image-model output or the video's DECIDE block) over its input picture `ref`
    (the rendered layout), with the cue ball at cue_xy (table coordinates). Returns angle (deg), power, line_score, bar_score,
    drawn (module doc)."""
    global _BASE_U8
    a32 = np.clip(mark_alpha(img, ref, RD.COL_AIM), -1.0, 1.5)  # float32; the bar is averaged at this precision
    a = a32.astype(np.float64)
    st = np.clip(mark_alpha(img, ref, RD.COL_STICK), -1.0, 1.5).astype(np.float64)
    u, v = RD.to_px(*cue_xy)
    if _BASE_U8 is None:
        _BASE_U8 = np.clip(np.rint(RD.base()), 0, 255).astype(np.uint8)
    balls = np.any(np.asarray(ref)[..., :3] != _BASE_U8, -1)  # ball pixels of the input: the aim line is drawn under them
    cp = (float(cue_xy[0]), float(cue_xy[1]))
    offs = np.arange(-3.0, 3.01, 0.5)
    wts = np.clip(RD.AIM_HALF + 0.5 - np.abs(offs), 0.0, 1.0)
    offs, wts = offs[wts > 0], wts[wts > 0]

    def pts(th: float, t0: float, t1: float, flip: bool, full: bool):
        """Sample points (x, y, weight) of a line template from t0 to t1 px along th (flip: backwards, the stick)."""
        if t1 <= t0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        t = np.arange(t0, t1, 1.0)
        ex, ey = math.cos(th), -math.sin(th)
        if flip:
            ex, ey = -ex, -ey
        o, w = (offs, wts) if full else (np.zeros(1), np.ones(1))
        x = (u + t[:, None] * ex - o[None] * ey).ravel()
        y = (v + t[:, None] * ey + o[None] * ex).ravel()
        w = np.broadcast_to(w[None], (len(t), len(w))).ravel()
        k = (x >= 0.5) & (x <= RD.CW - 0.5) & (y >= 0.5) & (y <= RD.AIM_YMAX - 0.5)
        return x[k], y[k], w[k]

    def score(ang: float, full: bool = True) -> float:
        """Normalised correlation of the alpha maps with the mark of angle `ang` (full: the 4-px profile, else the centre line)."""
        ang = ang % 360.0
        th = math.radians(ang)
        xa, ya, wa = pts(th, RD.AIM_START, RD.cushion_dist(cp, ang) * RD.SCALE, False, full)
        if len(xa):
            k = ~balls[np.clip(ya.astype(np.int64), 0, RD.CH - 1), np.clip(xa.astype(np.int64), 0, RD.CW - 1)]
            xa, ya, wa = xa[k], ya[k], wa[k]
        xs, ys, ws = pts(th, RD.AIM_START, RD.STICK_LEN * RD.SCALE, True, full)
        den = float((wa * wa).sum() + (ws * ws).sum())
        if den <= 0:
            return -np.inf
        return float(((_bilinear(a, xa, ya) * wa).sum() + (_bilinear(st, xs, ys) * ws).sum()) / math.sqrt(den))

    sc = np.array([score(g, False) for g in GRID])
    peaks: list[float] = []
    for j in np.argsort(-sc, kind="stable"):
        if len(peaks) == N_PEAKS:
            break
        if all(abs(P.ang_diff(GRID[j], q)) > PEAK_SEP for q in peaks):
            peaks.append(float(GRID[j]))
    best = (-np.inf, None)
    for q in peaks:
        for g in q + np.arange(-50, 51) * 0.02:
            s_ = score(g)
            if s_ > best[0]:
                best = (s_, g)
    q = best[1]
    for g in q + np.arange(-10, 11) * 0.002:
        s_ = score(g)
        if s_ > best[0]:
            best = (s_, g)
    bar = a32[RD.BAR_Y[0] + 1:RD.BAR_Y[1] - 1, RD.TRACK[0]:RD.TRACK[2]].mean(0).astype(np.float64)
    S = np.cumsum(bar)
    j = int(np.argmax(S / np.sqrt(np.arange(1, len(bar) + 1))))
    return {"angle": float(best[1] % 360.0), "power": (j + 1) / RD.TRACK_W, "line_score": float(best[0]),
            "bar_score": float(S[j] / math.sqrt(j + 1)), "drawn": bool(best[0] > 0 and S[j] > 0)}
