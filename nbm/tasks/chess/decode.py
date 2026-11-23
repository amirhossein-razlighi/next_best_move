"""Reading a move back from a generated picture (the shared decoder D and the paper's threshold-free readout).

Per square, the change out - board is fitted by least squares on (BLUE - board) and (RED - board) (nbm.marks.fit_strengths;
the inner 76% of the square, INNER_MARGIN dropped at each edge); a video DECIDE block has a third regressor for the green dots.
The coefficients divided by alpha are the blue / red strengths of the 64 squares. Threshold-free readout: the strongest blue
square is the origin and the strongest red square the destination, however faint (ties go to the lower square index). The
rules are applied afterwards: if that (origin, destination) pair is not a legal move the answer is NO_MOVE, which is always
wrong (and votes as one token in a majority vote). Promotion pieces cannot be drawn, so moves compare on their first four
UCI characters.
"""
from __future__ import annotations

import chess
import numpy as np
from PIL import Image

from nbm import marks, readout
from nbm.tasks.chess.board import dot_coverage, render_board, square_boxes

INNER_MARGIN = 0.12  # fraction of a square ignored at each edge (grid edges, VAE blur)
NO_MOVE = "none"


def strengths(out: Image.Image, cond: Image.Image, dots: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(blue (64,), red (64,)) mark strengths of a generated picture against its board picture (python-chess square order).
    dots=True also fits the green dots of a DECIDE block (as a nuisance regressor)."""
    size = cond.size[0]
    colors = [marks.BLUE, marks.RED] + ([marks.GREEN] if dots else [])
    cov = [None, None] + ([dot_coverage(size)] if dots else [])
    coef, _ = marks.fit_strengths(marks.as_array(out, cond.size), marks.as_array(cond), square_boxes(size), colors,
                                  margin=int(size / 8 * INNER_MARGIN), coverages=cov)
    return marks.to_strength(coef[:, 0]), marks.to_strength(coef[:, 1])


def read_move(out: Image.Image, board: chess.Board, cond: Image.Image | None = None, dots: bool = False,
              require_mark: bool = False) -> dict:
    """Threshold-free decision of one picture: {"move": "e2e4" or NO_MOVE, "origin", "destination", "blue_peak", "red_peak"}.

    cond: the board picture (rendered from `board` if None). require_mark (video DECIDE blocks): a picture whose strongest
    blue or red strength is exactly 0 (nothing drawn at all) gives NO_MOVE, as in the paper's video readout."""
    cond = render_board(board, out.size[0]) if cond is None else cond
    blue, red = strengths(out, cond, dots)
    origin, dest = readout.strongest(blue), readout.strongest(red)
    legal = {(m.from_square, m.to_square) for m in board.legal_moves}
    drawn = not require_mark or (blue.max() > 0 and red.max() > 0)
    move = chess.square_name(origin) + chess.square_name(dest) if drawn and (origin, dest) in legal else NO_MOVE
    return {"move": move, "origin": chess.square_name(origin), "destination": chess.square_name(dest),
            "blue_peak": float(blue.max()), "red_peak": float(red.max())}


def read_image_move(out: Image.Image, fen: str, cond: Image.Image | None = None) -> dict:
    """Image model: the move marked on the generated board."""
    return read_move(out, chess.Board(fen), cond)


def read_clip_move(decide: Image.Image, fen: str, cond: Image.Image | None = None) -> dict:
    """Video model: the first move, read from the clip's first DECIDE block (block 1) against the starting board."""
    board = chess.Board(fen)
    if cond is None:
        cond = render_board(board, decide.size[0])
    elif decide.size != cond.size:
        decide = decide.resize(cond.size, Image.BILINEAR)
    return read_move(decide, board, cond, dots=True, require_mark=True)


def same_move(pred: str, answer: str) -> bool:
    """Moves compare on origin and destination (the picture cannot encode a promotion piece)."""
    return pred != NO_MOVE and pred[:4] == answer[:4]
