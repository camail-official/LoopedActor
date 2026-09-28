"""Build the Rush Hour level banks used in the paper from Fogleman's puzzle database.

Input: rush.txt, one puzzle per line: <moves> <36-char board> <cluster size>, 2,577,412 rows (see README for the source).
Board chars: o empty, x wall, A red car (always row 2, horizontal), B..Z other pieces (length 2 or 3).
Output grid encoding (int8, [6,6]): 0 empty, -1 wall, 1 red car, 2..16 other pieces relabelled in row-major
first-occurrence order (stable, no dependence on the database letters). A move = one piece slid any distance.

Only puzzles with at most 15 moves are used (the paper's setting):
  easy_train  all <= 15-move puzzles minus the two held-out sets (1,979,836 levels)   training distribution
  easy_valid  10k random <= 15-move puzzles (held out; in-training evaluation)
  easy_test   10k random <= 15-move puzzles (held out; reported numbers use the first 2000)
The split is a fixed permutation (numpy RandomState(0)) of the <= 15-move rows, identical to the paper's banks.
Files: <out>/<split>.npz with grid [N,6,6] int8, moves [N] int16, cluster [N] int32, index [N] int32 (row in rush.txt).
Usage (repo root): python data_scripts/build_rushhour_banks.py --src rush.txt [--out data/rushhour]
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='path to rush.txt')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'rushhour'))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rows = [l.split() for l in open(args.src)]
N = len(rows); print('puzzles', N)
moves = np.array([int(r[0]) for r in rows], np.int16); cluster = np.array([int(r[2]) for r in rows], np.int32)
grid = np.zeros((N, 6, 6), np.int8)
for i, r in enumerate(rows):
    b = r[1]; assert len(b) == 36, (i, b)
    g = np.zeros(36, np.int8); nxt = 2; seen = {}
    for j, ch in enumerate(b):
        if ch == 'o': continue
        if ch == 'x': g[j] = -1; continue
        if ch == 'A': g[j] = 1; continue
        if ch not in seen: seen[ch] = nxt; nxt += 1
        g[j] = seen[ch]
    grid[i] = g.reshape(6, 6)
rng = np.random.RandomState(0)
easy = np.flatnonzero(moves <= 15); rng.shuffle(easy)
splits = {'easy_valid': easy[:10000], 'easy_test': easy[10000:20000], 'easy_train': np.sort(easy[20000:])}
for name, idx in splits.items():
    np.savez_compressed(f'{args.out}/{name}.npz', grid=grid[idx], moves=moves[idx], cluster=cluster[idx], index=idx.astype(np.int32))
    print(f'{name:12s} n={len(idx):8d} moves {moves[idx].min()}-{moves[idx].max()} mean {moves[idx].mean():.1f}')
