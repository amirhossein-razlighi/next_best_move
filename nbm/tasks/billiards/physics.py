"""The billiards table and its physics: constants, the C simulator (built on first use, loaded with ctypes), shot outcomes.

Table = cushion noses [0, 2] x [0, 1] (units = table widths, y up), ball radius R = 0.03, constant rolling deceleration
A = 0.75 / s^2, equal-mass ball-ball collisions with normal restitution 0.95, cushions with normal restitution 0.8, six pockets =
points (corners + middle of the long sides): a ball is potted the moment its centre comes within 0.085 (corner) / 0.07 (side) of
the point. No spin. A shot is (angle in degrees counter-clockwise from +x, power p in [0, 1]); the cue ball leaves at
v0 = VMAX p, VMAX = 3 (a full-power cue ball alone would roll 6 widths).

Ball 0 is the cue ball, ball 1 the target (the red ball). A shot SUCCEEDS iff the red ball is potted and the cue ball is not
(other balls may drop).

The simulator (sim/bilsim.c) is compiled with gcc the first time it is needed, into $NBM_OUTPUTS/cache/billiards/ (the file
name carries a hash of the source, so an edited source is rebuilt). The flags forbid fused multiply-add and fast math, so the
results are bit-identical on every x86-64 machine; the training labels and the evaluation depend on that.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nbm import paths

# ---------------------------------------------------------------- table and physics constants
L, W = 2.0, 1.0                        # table length / width (cushion noses)
R = 0.03                               # ball radius
A_DEC = 0.75                           # rolling deceleration (widths / s^2)
E_BALL, E_CUSH = 0.95, 0.8             # normal restitution of ball-ball and ball-cushion contacts
RC_CORNER, RC_SIDE = 0.085, 0.07       # pocket capture radius (corner / side pockets)
H_STEP, T_MAX, MAX_EV = 0.01, 30.0, 5000  # integration step (s), time cap (s), event cap
VMAX = 3.0                             # cue speed at power 1 (widths / s)
POCKETS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
POCKET_R = np.array([RC_CORNER, RC_SIDE, RC_CORNER, RC_CORNER, RC_SIDE, RC_CORNER])
PARAMS = np.array([L, W, R, A_DEC, E_BALL, E_CUSH, RC_CORNER, RC_SIDE, H_STEP, T_MAX, MAX_EV], dtype=np.float64)
# event counters returned by the simulator per shot (see bilsim.c)
INFO_KEYS = ("first_hit", "cue_cush_before", "tgt_cush", "tgt_hits", "tgt_first_hitter", "events", "overflow", "cue_cush",
             "collisions", "tgt_pocket", "cue_pocket", "n_obj_potted")
INFO_N = len(INFO_KEYS)

SOURCE = Path(__file__).with_name("sim") / "bilsim.c"
CFLAGS = ["-O2", "-march=x86-64", "-mno-fma", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared"]
_LIB = None


def library_path() -> Path:
    """Where the compiled simulator of the current source lives."""
    h = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    return paths.output_path("cache") / "billiards" / f"libbilsim_{h}.so"


def build() -> Path:
    """Compile sim/bilsim.c (if not compiled yet) and return the library path. Safe to call from many processes at once: each
    compiles into a temporary file and moves it into place atomically."""
    out = library_path()
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".so", dir=out.parent)
        os.close(fd)
        try:
            subprocess.check_call(["gcc", *CFLAGS, "-o", tmp, str(SOURCE), "-lm"])
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return out


def lib() -> ctypes.CDLL:
    """The loaded simulator (built on first use)."""
    global _LIB
    if _LIB is None:
        lb = ctypes.CDLL(str(build()))
        dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        lb.bil_sim.restype = ctypes.c_int
        lb.bil_sim.argtypes = [dp, ctypes.c_int, dp, ip, ctypes.c_double, ctypes.c_double, ctypes.c_int, dp, dp, ip, dp, ip, dp, ip,
                               dp]
        lb.bil_batch.restype = None
        lb.bil_batch.argtypes = [dp, ctypes.c_int, dp, ip, ctypes.c_int, dp, dp, ip, ip]
        if lb.bil_info_n() != INFO_N:
            raise RuntimeError(f"{SOURCE} reports {lb.bil_info_n()} event counters, expected {INFO_N}")
        _LIB = lb
    return _LIB


def _d(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _i(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def as_xy(xy) -> np.ndarray:
    """Ball centres as a contiguous float64 (n, 2) array (accepts a flat list of 2n numbers)."""
    return np.ascontiguousarray(np.asarray(xy, dtype=np.float64).reshape(-1, 2))


def _alive(alive, n: int) -> np.ndarray:
    return np.ascontiguousarray(np.ones(n, np.int32) if alive is None else np.asarray(alive, np.int32))


def simulate(xy, alive=None, angle: float = 0.0, power: float = 0.5, rec_t=None) -> dict:
    """One shot from the layout xy ((n, 2), ball 0 = cue, 1 = red); alive (n,) 0/1 (default: all on the table).
    Returns pocket (n,) (-1 or the pocket index), ptime, final_xy (n, 2), alive_final, info (INFO_KEYS -> int), t_rest, and the
    ball positions frames (len(rec_t), n, 2) / frames_alive at the times rec_t (seconds)."""
    xy = as_xy(xy)
    n = len(xy)
    al = _alive(alive, n)
    rt = np.ascontiguousarray(np.asarray([] if rec_t is None else rec_t, dtype=np.float64))
    nr = len(rt)
    rxy, ral = np.zeros((max(nr, 1), n, 2)), np.zeros((max(nr, 1), n), np.int32)
    oxy, opk, opt = np.zeros((n, 2)), np.zeros(n, np.int32), np.zeros(n)
    info, tr = np.zeros(INFO_N, np.int32), np.zeros(1)
    lib().bil_sim(_d(PARAMS), n, _d(xy), _i(al), float(angle), float(VMAX * power), nr, _d(rt), _d(rxy), _i(ral), _d(oxy), _i(opk),
                  _d(opt), _i(info), _d(tr))
    return {"pocket": opk.tolist(), "ptime": opt.tolist(), "final_xy": oxy,
            "alive_final": (al.astype(bool) & (opk < 0)).astype(np.int32), "info": dict(zip(INFO_KEYS, info.tolist())),
            "t_rest": float(tr[0]), "frames": rxy[:nr], "frames_alive": ral[:nr]}


def batch(xy, alive, angles, powers) -> tuple[np.ndarray, np.ndarray]:
    """Many shots from one layout -> (potted mask (ns,) int, bit i = ball i potted; info (ns, INFO_N))."""
    xy = as_xy(xy)
    n = len(xy)
    al = _alive(alive, n)
    ang = np.ascontiguousarray(np.asarray(angles, dtype=np.float64))
    v = np.ascontiguousarray(VMAX * np.asarray(powers, dtype=np.float64))
    if ang.shape != v.shape:
        raise ValueError(f"{ang.shape} angles vs {v.shape} powers")
    pm, info = np.zeros(len(ang), np.int32), np.zeros((len(ang), INFO_N), np.int32)
    lib().bil_batch(_d(PARAMS), n, _d(xy), _i(al), len(ang), _d(ang), _d(v), _i(pm), _i(info))
    return pm, info


def success_mask(pmask) -> np.ndarray:
    """Potted masks -> success (red potted, cue ball not)."""
    pmask = np.asarray(pmask)
    return ((pmask & 2) != 0) & ((pmask & 1) == 0)


def shot_ok(xy, alive, angle: float, power: float) -> bool:
    pm, _ = batch(xy, alive, [angle], [power])
    return bool(success_mask(pm)[0])


def direct_mask(pmask, info) -> np.ndarray:
    """Successful DIRECT pots: the cue ball hits the red first without a cushion, the red drops without a cushion or another
    contact."""
    i = {k: info[:, j] for j, k in enumerate(INFO_KEYS)}
    return success_mask(pmask) & (i["first_hit"] == 1) & (i["cue_cush_before"] == 0) & (i["tgt_cush"] == 0) & (i["tgt_hits"] == 1)


def shot_type(info: dict) -> str:
    """Class of a successful shot from its event counters: combo > kick > bank > kiss > direct."""
    if info["tgt_first_hitter"] not in (0, -1):
        return "combo"
    if info["cue_cush_before"] > 0:
        return "kick"
    if info["tgt_cush"] > 0:
        return "bank"
    if info["tgt_hits"] > 1:
        return "kiss"
    return "direct"


def ang_diff(a: float, b: float) -> float:
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d if d != -180.0 else 180.0


def mirror_xy(xy, how: str) -> np.ndarray:
    """An exact table symmetry (pockets map to pockets): 'lr' (x -> L - x), 'tb' (y -> W - y) or 'both'."""
    xy = as_xy(xy).copy()
    if how in ("lr", "both"):
        xy[:, 0] = L - xy[:, 0]
    if how in ("tb", "both"):
        xy[:, 1] = W - xy[:, 1]
    return xy

