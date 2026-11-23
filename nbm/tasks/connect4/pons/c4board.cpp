// Batch Connect Four analyser on top of Pascal Pons' solver (https://github.com/PascalPons/connect4, AGPL-3.0), compiled
// against its sources by build.sh (the solver itself is not part of this repository).
// stdin, one query per line:
//   B <42 chars>   board, column-major, each column bottom -> top (index = 6 * col + row); '.' empty, 'x' first player (Red),
//                  'o' second player (Blue). Side to move by parity (Red iff #x == #o). The caller guarantees a valid,
//                  non-terminal board (gravity, counts, no four).
//   S <moves>      a move sequence (columns 1..7, as Pons' c4solver), for cross-checks.
// stdout, one line per query: the 7 move scores for the side to move (Pons' convention; -1000 = full column), space
// separated, or ERR. Score of a move: > 0 win, 0 draw, < 0 loss; a win with the mover's own k-th stone scores 22 - k, a loss
// to the opponent's k-th stone -(22 - k).
#define private public  // the board is set directly from its bitboards (Position has no public setter)
#include "Solver.hpp"
#undef private
#include <iostream>
#include <string>
#include <vector>

using namespace GameSolver::Connect4;

int main(int argc, char** argv) {
  Solver solver;
  std::string book = argc > 1 ? argv[1] : "7x6.book";
  solver.loadBook(book);
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.size() < 2) { std::cout << "ERR\n" << std::flush; continue; }
    Position P;
    if (line[0] == 'S') {
      std::string seq = line.substr(2);
      if (P.play(seq) != seq.size()) { std::cout << "ERR\n" << std::flush; continue; }
    } else if (line[0] == 'B') {
      std::string b = line.substr(2);
      if (b.size() != 42) { std::cout << "ERR\n" << std::flush; continue; }
      int nx = 0, no = 0;
      for (char c : b) { nx += c == 'x'; no += c == 'o'; }
      char cur = nx == no ? 'x' : 'o';
      Position::position_t cp = 0, mask = 0;
      for (int col = 0; col < 7; col++)
        for (int row = 0; row < 6; row++) {
          char c = b[6 * col + row];
          if (c == '.') continue;
          Position::position_t bit = Position::position_t(1) << (col * (Position::HEIGHT + 1) + row);
          mask |= bit;
          if (c == cur) cp |= bit;
        }
      P.current_position = cp;
      P.mask = mask;
      P.moves = nx + no;
    } else { std::cout << "ERR\n" << std::flush; continue; }
    std::vector<int> s = solver.analyze(P, false);
    for (int i = 0; i < 7; i++) std::cout << (i ? " " : "") << s[i];
    std::cout << "\n" << std::flush;
  }
}
