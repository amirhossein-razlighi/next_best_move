#!/bin/bash
# Build the Connect Four solver driver used by the data builder (nbm/tasks/connect4/solver.py).
#   bash nbm/tasks/connect4/pons/build.sh [DEST]        DEST default: $NBM_CONNECT4_SOLVER or ./tools/connect4
# 1. clones Pascal Pons' solver (https://github.com/PascalPons/connect4, AGPL-3.0) at the commit the paper's labels used,
# 2. downloads its 7x6 opening book (release asset "book" of the same repository) and checks its sha256,
# 3. compiles c4board.cpp (batch driver: one board per line -> 7 move scores) and Pons' own c4solver against the sources.
# Result: DEST/bin/c4board, DEST/bin/c4solver, DEST/book/7x6.book. Needs git, curl (or wget) and g++ (C++11).
set -euo pipefail
REPO=https://github.com/PascalPons/connect4.git
COMMIT=d6ba50d8aaf2308c769d9bf2abd42d90f34baf41
BOOK_URL=https://github.com/PascalPons/connect4/releases/download/book/7x6.book
BOOK_SHA256=f346cd449626fb81da93be0958e017ee854e5f74d85b6d38062357f5403aec53
DEST=${1:-${NBM_CONNECT4_SOLVER:-tools/connect4}}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DEST/src" "$DEST/bin" "$DEST/book"
SRC=$DEST/src/connect4
if [ ! -d "$SRC/.git" ]; then
    git clone --quiet "$REPO" "$SRC"
fi
git -C "$SRC" checkout --quiet "$COMMIT"
BOOK=$DEST/book/7x6.book
if [ ! -f "$BOOK" ]; then
    if command -v curl >/dev/null; then curl -sSL -o "$BOOK.tmp" "$BOOK_URL"; else wget -q -O "$BOOK.tmp" "$BOOK_URL"; fi
    mv "$BOOK.tmp" "$BOOK"
fi
echo "$BOOK_SHA256  $BOOK" | sha256sum -c --quiet - || { echo "opening book checksum mismatch: $BOOK"; exit 1; }
g++ --std=c++11 -O3 -DNDEBUG -march=x86-64 -I "$SRC" -o "$DEST/bin/c4board" "$HERE/c4board.cpp" "$SRC/Solver.cpp"
g++ --std=c++11 -O3 -DNDEBUG -march=x86-64 -I "$SRC" -o "$DEST/bin/c4solver" "$SRC/main.cpp" "$SRC/Solver.cpp"
echo "built $DEST/bin/c4board and $DEST/bin/c4solver (book $BOOK)"
printf 'S 4\n' | "$DEST/bin/c4board" "$BOOK" | sed 's/^/self-test (scores after the first move in the centre): /'
