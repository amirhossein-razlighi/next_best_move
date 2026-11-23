"""Chess pictures: the board, the move marks M(x, a) and the decide / imagine clip.

Conventions (the paper's, fixed): the board is drawn from White's side (a1 bottom left) without coordinates or border, so the
square of file f and rank r is the pixel box x in [f s, (f + 1) s), y in [(7 - r) s, (8 - r) s) with s = size / 8. The move
a = (origin, destination) is drawn by tinting the origin square BLUE and the destination square RED (alpha blend, nbm.marks).

Video clip (17 blocks, nbm.models.video): block 0 = the board; for every ply k = 1 .. 8 of the puzzle line (the solver's moves
and the opponent's replies) block 2k - 1 = DECIDE (the board before ply k with a GREEN dot on every legal destination of the
moved piece, then the blue / red tints on top) and block 2k = IMAGINE (the board after ply k); after the line every remaining
block repeats the final board. The dots are the "legal squares green" of the video prompt; the decoder fits them as a nuisance
colour and reads only the blue / red marks.
"""
from __future__ import annotations

import io
from functools import lru_cache

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image

from nbm import marks

DOT_ALPHA = 0.6
DOT_RADIUS = 0.3  # dot radius as a fraction of the square side
MAX_PLIES = 8  # plies of the puzzle line shown in a clip (1 + 2 x 8 = 17 blocks)


def image_prompt(side: str) -> str:
    """Image model prompt; side = "white" / "black" (the side to move)."""
    return f"{side.capitalize()} to move. Highlight the best move: the origin square in blue and the destination square in red."


def video_prompt(side: str) -> str:
    """Video model prompt; side = "white" / "black" (the side to move in the first block)."""
    return (f"{side.capitalize()} to move. Show the best continuation: at each step highlight the move "
            "(origin blue, legal squares green, destination red), then show the board after it.")


def render_board(board: chess.Board, size: int) -> Image.Image:
    """The board as an RGB picture of size x size pixels (python-chess SVG rendered by cairosvg)."""
    svg = chess.svg.board(board, size=size, orientation=chess.WHITE, coordinates=False, borders=False)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGB")


def square_box(square: int, size: int) -> marks.Box:
    """Pixel box (x0, y0, x1, y1) of a python-chess square index (a1 = 0 ... h8 = 63)."""
    s = size / 8
    f, r = chess.square_file(square), chess.square_rank(square)
    return int(round(f * s)), int(round((7 - r) * s)), int(round((f + 1) * s)), int(round((8 - r) * s))


def square_boxes(size: int) -> list[marks.Box]:
    """The 64 square boxes in python-chess square order."""
    return [square_box(sq, size) for sq in range(64)]


@lru_cache(maxsize=None)
def disc_mask(h: int, w: int) -> np.ndarray:
    """(h, w) anti-aliased coverage in [0, 1] of the dot centred in an h x w square (4 x 4 supersampling)."""
    ss = 4
    y, x = (np.mgrid[0:h * ss, 0:w * ss] + 0.5) / ss
    inside = (x - w / 2) ** 2 + (y - h / 2) ** 2 <= (DOT_RADIUS * min(h, w)) ** 2
    return inside.reshape(h, ss, w, ss).mean(axis=(1, 3)).astype(np.float32)


@lru_cache(maxsize=None)
def dot_coverage(size: int) -> np.ndarray:
    """(size, size) coverage map with the dot drawn in every square (the decoder's regressor for the green colour)."""
    cov = np.zeros((size, size), np.float32)
    for x0, y0, x1, y1 in square_boxes(size):
        cov[y0:y1, x0:x1] = disc_mask(y1 - y0, x1 - x0)
    return cov


def draw_dots(img: Image.Image, squares) -> Image.Image:
    """A green dot (alpha DOT_ALPHA) in each of the given squares."""
    size = img.size[0]
    cov = np.zeros((size, size), np.float32)
    for sq in squares:
        x0, y0, x1, y1 = square_box(sq, size)
        cov[y0:y1, x0:x1] = disc_mask(y1 - y0, x1 - x0)
    return marks.blend(img, cov, marks.GREEN, DOT_ALPHA)


def legal_destinations(board: chess.Board, from_square: int) -> set[int]:
    """Destination squares of the legal moves of the piece on from_square (the promotions of a pawn share one square)."""
    return {m.to_square for m in board.legal_moves if m.from_square == from_square}


def move_marks(cond: Image.Image, move: chess.Move, dots=()) -> Image.Image:
    """The condition picture with the move drawn: green dots on `dots` first, then origin BLUE and destination RED."""
    base = draw_dots(cond, dots) if dots else cond
    size = cond.size[0]
    return marks.tint(base, [(square_box(move.from_square, size), marks.BLUE), (square_box(move.to_square, size), marks.RED)])


def image_pair(fen: str, move_uci: str, size: int) -> tuple[Image.Image, Image.Image]:
    """Image model example: (board, board with the move marked)."""
    cond = render_board(chess.Board(fen), size)
    return cond, move_marks(cond, chess.Move.from_uci(move_uci))


def clip_blocks(fen: str, line: list[str], size: int) -> list[Image.Image]:
    """The 17 blocks of the decide / imagine clip of a puzzle line (module docstring); the line is cut at MAX_PLIES plies."""
    board = chess.Board(fen)
    blocks = [render_board(board, size)]
    for u in list(line)[:MAX_PLIES]:
        mv = chess.Move.from_uci(u)
        if mv not in board.legal_moves:
            raise ValueError(f"illegal move {u} in {board.fen()}")
        cond = render_board(board, size)
        blocks.append(move_marks(cond, mv, legal_destinations(board, mv.from_square)))
        board.push(mv)
        blocks.append(render_board(board, size))
    final = render_board(board, size)
    while len(blocks) < 1 + 2 * MAX_PLIES:
        blocks.append(final.copy())
    return blocks
