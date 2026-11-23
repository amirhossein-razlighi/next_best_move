"""Pictures of the billiards task: the table, the shot mark M(x, a) and the video clip of a shot.

Picture (512 x 288): dark background, brown rails, green cloth 464 x 232 px (232 px per table width, ball diameter ~14 px), black
pocket discs of exactly the capture radii, balls as flat antialiased discs with a 1-px darker rim, and a power track (a dark bar
464 px long) below the table.

Shot mark (= the image model's target = the video's DECIDE block), the "cushion format" of the paper:
  * a CYAN aim line from 2 px outside the cue ball along the shot direction to where the cue-ball CENTRE would meet a cushion
    (R from the cloth edge), drawn UNDER the balls (it is interrupted by every ball it crosses);
  * a MAGENTA cue stick behind the cue ball on the same axis (from 2 px outside it to 0.6 table widths behind its centre);
  * the power track filled CYAN from the left to p x 464 px (antialiased at the end).
Lines are 4 px wide, antialiased, and never drawn over the power track.

Video clip (17 blocks, nbm.models.video): block 0 = the layout, 1 = DECIDE (the layout + the shot mark), 2..15 = the simulated
balls at t = 0.2, 0.4, ..., 2.8 s, 16 = the final resting state (balls potted on the way vanish).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from nbm.tasks.billiards import physics as P

CW, CH = 512, 288                                                    # canvas (width, height); model.resolution = 288 = CH
SCALE = 232.0                                                        # px per table width
X0, Y0 = 24.0, 12.0                                                  # cloth top-left (pixel edges)
RAIL = (12, 0, 500, 256)                                             # outer rail box (x0, y0, x1, y1)
TRACK = (24, 264, 488, 280)                                          # power track box
TRACK_W = TRACK[2] - TRACK[0]                                        # 464
BAR_Y = (266, 278)                                                   # rows of the filled bar
STICK_LEN = 0.6                                                      # cue stick end, behind the cue centre (table widths)
AIM_HALF = 2.0                                                       # aim line / stick half width (px): 4 px wide
AIM_START = P.R * SCALE + 2.0                                        # lines start 2 px outside the cue ball (px)
AIM_YMAX = RAIL[3] + 2                                               # lines are clipped to rows < 258 (off the power track)
COL_BG = (30, 32, 36)
COL_RAIL = (104, 62, 30)
COL_CLOTH = (36, 118, 72)
COL_POCKET = (12, 12, 12)
COL_TRACK = (66, 66, 72)
COL_AIM = (0, 225, 255)                                              # cyan: aim line and power bar
COL_STICK = (236, 60, 200)                                           # magenta: cue stick
BALL_COLORS = {0: (246, 246, 238), 1: (214, 28, 28), 2: (244, 200, 28), 3: (34, 70, 210), 4: (128, 48, 168), 5: (250, 128, 24)}
COLOR_NAMES = {0: "cue", 1: "red", 2: "yellow", 3: "blue", 4: "purple", 5: "orange"}
FRAME_DT, N_MOTION = 0.2, 14                                         # video: t = 0.2 .. 2.8 s, then the final state

_YY, _XX = np.mgrid[0:CH, 0:CW].astype(np.float32) + 0.5            # pixel centres
_BASE: np.ndarray | None = None


def to_px(x: float, y: float) -> tuple[float, float]:
    """Table coordinates (widths, y up) -> pixel coordinates (x right, y down)."""
    return X0 + x * SCALE, Y0 + (P.W - y) * SCALE


def _blend(img: np.ndarray, alpha: np.ndarray, col) -> None:
    a = alpha[..., None]
    img[:] = img * (1 - a) + np.asarray(col, np.float32) * a


def _draw_disc(img: np.ndarray, cx: float, cy: float, r: float, col) -> None:
    x0, x1 = max(0, int(cx - r - 2)), min(CW, int(cx + r + 3))
    y0, y1 = max(0, int(cy - r - 2)), min(CH, int(cy + r + 3))
    d = np.hypot(_XX[y0:y1, x0:x1] - cx, _YY[y0:y1, x0:x1] - cy)
    _blend(img[y0:y1, x0:x1], np.clip(r + 0.5 - d, 0.0, 1.0), col)


def _draw_seg(img: np.ndarray, u: float, v: float, ex: float, ey: float, a0: float, a1: float, col) -> None:
    """Antialiased 4-px segment from (u, v) + a0 e to (u, v) + a1 e (px), clipped to rows < AIM_YMAX."""
    half = AIM_HALF
    bx0, bx1 = sorted((u + ex * a0, u + ex * a1))
    by0, by1 = sorted((v + ey * a0, v + ey * a1))
    x0, x1 = max(0, int(bx0 - half - 3)), min(CW, int(bx1 + half + 4))
    y0, y1 = max(0, int(by0 - half - 3)), min(AIM_YMAX, int(by1 + half + 4))
    if x1 > x0 and y1 > y0:
        qx, qy = _XX[y0:y1, x0:x1] - u, _YY[y0:y1, x0:x1] - v
        t = np.clip(qx * ex + qy * ey, a0, a1)
        d = np.hypot(qx - t * ex, qy - t * ey)
        _blend(img[y0:y1, x0:x1], np.clip(half + 0.5 - d, 0.0, 1.0), col)


def base() -> np.ndarray:
    """float32 (CH, CW, 3) picture of the empty table (cached; do not modify)."""
    global _BASE
    if _BASE is None:
        img = np.empty((CH, CW, 3), np.float32)
        img[:] = COL_BG
        x0, y0, x1, y1 = RAIL
        img[y0:y1, x0:x1] = COL_RAIL
        cx0, cy0 = int(X0), int(Y0)
        img[cy0:cy0 + int(P.W * SCALE), cx0:cx0 + int(P.L * SCALE)] = COL_CLOTH
        for (px, py), r in zip(P.POCKETS, P.POCKET_R):
            u, v = to_px(px, py)
            _draw_disc(img, u, v, r * SCALE, COL_POCKET)
        tx0, ty0, tx1, ty1 = TRACK
        img[ty0:ty1, tx0:tx1] = COL_TRACK
        _BASE = img
    return _BASE


def cushion_dist(p, ang: float) -> float:
    """Distance (table widths) from the cue centre p along `ang` (degrees) to where the centre is R from a cushion nose."""
    th = math.radians(ang)
    ex, ey = math.cos(th), math.sin(th)
    ts = []
    for d, lo, hi, x in ((ex, P.R, P.L - P.R, p[0]), (ey, P.R, P.W - P.R, p[1])):
        if d > 1e-12:
            ts.append((hi - x) / d)
        elif d < -1e-12:
            ts.append((lo - x) / d)
    return max(0.0, min(ts))


def render_f(xy, colors, alive=None, shot: tuple[float, float] | None = None) -> np.ndarray:
    """float32 (CH, CW, 3) picture of a layout; shot = (angle, power) adds the shot mark (module doc)."""
    img = base().copy()
    xy = P.as_xy(xy)
    al = np.ones(len(xy), bool) if alive is None else np.asarray(alive, bool)
    rp = P.R * SCALE
    if shot is not None:  # the aim line first: the balls are drawn over it
        u, v = to_px(*xy[0])
        th = math.radians(shot[0])
        _draw_seg(img, u, v, math.cos(th), -math.sin(th), AIM_START, cushion_dist(xy[0], shot[0]) * SCALE, COL_AIM)
    for i in range(len(xy)):
        if not al[i]:
            continue
        u, v = to_px(*xy[i])
        col = np.asarray(BALL_COLORS[int(colors[i])], np.float32)
        _draw_disc(img, u, v, rp, col * 0.55)  # rim
        _draw_disc(img, u, v, rp - 1.0, col)
    if shot is not None:
        ang, p = shot
        u, v = to_px(*xy[0])
        th = math.radians(ang)
        ex, ey = math.cos(th), -math.sin(th)
        _draw_seg(img, u, v, -ex, -ey, AIM_START, STICK_LEN * SCALE, COL_STICK)  # cue stick, behind the cue ball
        xe = TRACK[0] + float(np.clip(p, 0.0, 1.0)) * TRACK_W
        cov = np.clip(xe - (_XX[0, TRACK[0]:TRACK[2]] - 0.5), 0.0, 1.0)  # coverage of pixel column x by [TRACK0, xe)
        a = np.repeat(cov[None], BAR_Y[1] - BAR_Y[0], axis=0)
        _blend(img[BAR_Y[0]:BAR_Y[1], TRACK[0]:TRACK[2]], a, COL_AIM)
    return img


def render(xy, colors, alive=None, shot: tuple[float, float] | None = None) -> np.ndarray:
    """uint8 (CH, CW, 3) picture (render_f rounded once)."""
    return np.clip(np.rint(render_f(xy, colors, alive, shot)), 0, 255).astype(np.uint8)


def render_pil(xy, colors, alive=None, shot: tuple[float, float] | None = None) -> Image.Image:
    return Image.fromarray(render(xy, colors, alive, shot))


def shot_blocks(xy, colors, angle: float, power: float) -> np.ndarray:
    """The 17 video blocks (17, CH, CW, 3) uint8 of a shot from a layout (module doc)."""
    xy = P.as_xy(xy)
    al = np.ones(len(xy), np.int32)
    s = P.simulate(xy, al, angle, power, rec_t=FRAME_DT * np.arange(1, N_MOTION + 1))
    states = [(s["frames"][k], s["frames_alive"][k]) for k in range(N_MOTION)] + [(s["final_xy"], s["alive_final"])]
    blocks = [render(xy, colors, al), render(xy, colors, al, (angle, power))]
    blocks += [render(sxy, colors, sal) for sxy, sal in states]
    return np.stack(blocks)
