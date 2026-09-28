"""Build the Boxoban level banks used by envs/sokoban_env.py.

Downloads google-deepmind/boxoban-levels (master.zip, ~33 MB) unless the archive is already at
data/boxoban/boxoban-levels.zip, parses every level file and writes one .npz per split:

  data/boxoban/<split>.npz   with  walls, boxes, targets  [N, 10, 10] uint8   and  player [N, 2] uint8 (row, col)

Splits used in the paper (default): unfiltered_train (900k), unfiltered_valid (100k), unfiltered_test (1k).
The other Boxoban splits (medium_train, medium_valid, hard) can be built with --splits but are not used here. Files are read in sorted order and levels in file
order, so the level index is stable (evaluations refer to levels by this index).

Usage (repo root):  python data_scripts/build_boxoban_banks.py [--out data/boxoban] [--splits unfiltered_train unfiltered_valid unfiltered_test]
"""
import argparse
import io
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np

URL = 'https://github.com/google-deepmind/boxoban-levels/archive/refs/heads/master.zip'
SPLITS = {'unfiltered_train': 'unfiltered/train', 'unfiltered_valid': 'unfiltered/valid', 'unfiltered_test': 'unfiltered/test',
          'medium_train': 'medium/train', 'medium_valid': 'medium/valid', 'hard': 'hard'}
M = N = 10


def parse_levels(text):
    """Yield (walls, boxes, targets, player) for every level in one boxoban .txt file."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith(';'):
            i += 1; continue
        rows = lines[i + 1:i + 1 + M]
        assert len(rows) == M and all(len(r) == N for r in rows), (lines[i], rows)
        g = np.array([list(r) for r in rows])
        walls = (g == '#'); boxes = (g == '$') | (g == '*'); targets = (g == '.') | (g == '*') | (g == '+')
        player = np.argwhere((g == '@') | (g == '+'))
        assert len(player) == 1, lines[i]
        yield walls, boxes, targets, player[0]
        i += 1 + M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'boxoban'))
    ap.add_argument('--splits', nargs='*', default=['unfiltered_train', 'unfiltered_valid', 'unfiltered_test'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zpath = os.path.join(args.out, 'boxoban-levels.zip')
    if not os.path.exists(zpath):
        print('downloading', URL, flush=True)
        with urllib.request.urlopen(URL) as r, open(zpath, 'wb') as f:
            f.write(r.read())
    z = zipfile.ZipFile(zpath)
    names = z.namelist(); root = names[0].split('/')[0]
    for split in args.splits:
        sub = SPLITS[split]
        files = sorted(n for n in names if re.fullmatch(rf'{root}/{sub}/\d+\.txt', n))
        W, B, T, P = [], [], [], []
        for fn in files:
            for w, b, t, p in parse_levels(z.read(fn).decode()):
                W.append(w); B.append(b); T.append(t); P.append(p)
        out = os.path.join(args.out, f'{split}.npz')
        np.savez_compressed(out, walls=np.array(W, np.uint8), boxes=np.array(B, np.uint8), targets=np.array(T, np.uint8), player=np.array(P, np.uint8))
        print(f'{split:16s} {len(W):7d} levels from {len(files)} files -> {out}', flush=True)


if __name__ == '__main__':
    main()
