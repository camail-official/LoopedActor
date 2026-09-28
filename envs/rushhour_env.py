"""Rush Hour (6x6 sliding-block puzzle) environment in JAX, on Fogleman's database of all
"interesting" 6x6 configurations (https://www.michaelfogleman.com/rush/, 2,577,412 puzzles with exact
minimum move counts). Level banks are built by data_scripts/build_rushhour_banks.py into
<DATA_DIR>/<split>.npz (grid [N,6,6] int8, moves [N] int16 = Fogleman min moves, cluster, index).
Splits: easy_train / easy_valid / easy_test (moves <= 15), medium (16-24), hard (25-34), expert (>= 35).

Grid encoding: 0 empty, -1 wall, 1 red car (row 2, horizontal, length 2), 2..16 other pieces (length 2 or 3,
horizontal or vertical). A piece slides one cell per step along its own axis; it cannot pass through pieces,
walls or the border. The puzzle is solved when the red car reaches the right edge (its right end in column 5).

Observation: per-cell channels [occupied, red, wall, horizontal, vertical, link_right, link_down], cell-major flat
vector of size 6*6*7 (the actor reshapes it to [36, 7] tokens). link_right/link_down mark that this cell and its
right/down neighbour belong to the same piece, which makes piece boundaries unambiguous (e.g. BBCC vs BBBC).
No goal vector (goal_size 0): the exit is fixed geometry.

Actions (per-cell head, 2 per cell): a = cell * 2 + sign, sign 0 = move the piece occupying `cell` one step
towards smaller index along its axis (up / left), sign 1 = towards larger index (down / right). Choosing an
empty or wall cell, or a blocked move, is a no-op (step penalty still applies).
Reward: -step_penalty per step, +solve_reward when solved (episode ends), plus potential-based shaping
shaping_weight * (phi(s') - phi(s)) with phi = -(number of occupied cells between the red car and the exit) - (red car distance
to the exit). A random policy solves 0/2000 levels on every split, so the sparse solve reward alone gives PPO no signal;
potential-based shaping keeps the optimal policy unchanged (Ng et al. 1999). Episode length set in make_env.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from ml_collections import config_dict

from envs.utils import State

DATA_DIR = os.environ.get('RUSHHOUR_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'rushhour'))
M = N = 6
OBS_CHANNELS = 7
RED = 1
WALL = -1


@struct.dataclass
class RushHourData:
    grid: jax.Array   # [6, 6] int8: 0 empty, -1 wall, 1 red, 2.. pieces
    level: jax.Array  # () int32 index into the bank it was drawn from
    moves: jax.Array  # () int32 Fogleman minimum move count of the level (diagnostic only)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        m=M,
        n=N,
        episode_length=150,
        train_split='easy_train',
        eval_split='hard',
        max_train_levels=0,  # 0 = all levels of the split
        max_eval_levels=0,
        step_penalty=0.1,
        solve_reward=10.0,
        shaping_weight=1.0,  # potential-based shaping on (blockers, red-car distance); 0 = sparse reward only
    )


def load_bank(split, max_levels=0):
    z = np.load(os.path.join(DATA_DIR, f'{split}.npz'))
    n = len(z['grid']) if max_levels <= 0 else min(max_levels, len(z['grid']))
    return dict(grid=jnp.asarray(z['grid'][:n].astype(np.int8)), moves=jnp.asarray(z['moves'][:n].astype(np.int32)))


def piece_masks(grid):
    """Per-cell piece structure from the id grid: occupied, horizontal, vertical, link_right, link_down (all [6,6] bool)."""
    occ = grid > 0
    same_r = jnp.zeros_like(occ).at[:, :-1].set(occ[:, :-1] & (grid[:, :-1] == grid[:, 1:]))   # same piece as right neighbour
    same_d = jnp.zeros_like(occ).at[:-1, :].set(occ[:-1, :] & (grid[:-1, :] == grid[1:, :]))   # same piece as down neighbour
    same_l = jnp.zeros_like(occ).at[:, 1:].set(same_r[:, :-1])
    same_u = jnp.zeros_like(occ).at[1:, :].set(same_d[:-1, :])
    horizontal = same_r | same_l
    vertical = same_d | same_u
    return occ, horizontal, vertical, same_r, same_d


def potential(grid):
    """phi(s) = -(occupied cells in row 2 right of the red car) - (red car distance to the exit), both >= 0."""
    row = grid[2]
    red_right = jnp.max(jnp.where(row == RED, jnp.arange(N), -1))  # column of the red car's right end
    ahead = jnp.arange(N) > red_right
    blockers = jnp.sum(ahead & (row != 0))
    dist = (N - 1) - red_right
    return -(blockers + dist).astype(jnp.float32)


def solved(grid):
    """Red car's right end in the last column of row 2."""
    return grid[2, N - 1] == RED


class RushHourEnv:
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        self._config = config
        self.m, self.n = config.m, config.n
        self.grid_size = self.m * self.n
        self.obs_channels = OBS_CHANNELS
        self.action_head = 'per_cell'
        self.train_bank = load_bank(config.train_split, config.max_train_levels)
        self.eval_bank = load_bank(config.eval_split, config.max_eval_levels)
        self.num_train_levels = int(self.train_bank['grid'].shape[0])
        self.num_eval_levels = int(self.eval_bank['grid'].shape[0])

    @property
    def action_size(self): return self.grid_size * 2

    @property
    def observation_size(self): return self.grid_size * OBS_CHANNELS

    @property
    def goal_size(self): return 0

    def _obs(self, data: RushHourData):
        g = data.grid
        occ, hor, ver, link_r, link_d = piece_masks(g)
        cells = jnp.stack([occ, g == RED, g == WALL, hor, ver, link_r, link_d], axis=-1)  # [6, 6, 7]
        return cells.reshape(-1).astype(jnp.float32)

    def _make_state(self, rng, grid, idx, moves):
        data = RushHourData(grid=grid, level=idx, moves=moves)
        return State(
            data=data, obs=self._obs(data),
            reward=jnp.array(0.0, jnp.float32), done=jnp.array(0.0, jnp.float32),
            metrics={'success': 0.0, 'reward': 0.0},  # (level min-move count lives in data.moves; episode metrics are summed over steps)
            info={'rng': rng, 'target_goal': jnp.zeros((0,), jnp.float32)},
        )

    def _reset_from_bank(self, rng, bank, num_levels):
        idx = jax.random.randint(rng, (), 0, num_levels)
        return self._make_state(rng, bank['grid'][idx], idx, bank['moves'][idx])

    def reset(self, rng: jax.Array) -> State:
        return self._reset_from_bank(rng, self.train_bank, self.num_train_levels)

    def eval_reset(self, rng: jax.Array) -> State:
        return self._reset_from_bank(rng, self.eval_bank, self.num_eval_levels)

    def reset_level(self, bank, idx):
        """Deterministic reset to a given level index of a bank (for per-level evaluation)."""
        return self._make_state(jax.random.PRNGKey(0), bank['grid'][idx], jnp.asarray(idx, jnp.int32), bank['moves'][idx])

    def move(self, grid, action):
        """Apply action = cell * 2 + sign to the id grid. Returns (new_grid, moved: bool)."""
        cell, sign = action // 2, action % 2
        r, c = cell // self.n, cell % self.n
        pid = grid[r, c]
        is_piece = pid > 0
        mask = grid == pid
        _, hor, _, _, _ = piece_masks(grid)
        horizontal = hor[r, c]
        step = 2 * sign - 1  # -1 = up/left, +1 = down/right
        # shifted footprint (roll along the piece's axis, then reject wrap-around)
        shifted_h = jnp.roll(mask, step, axis=1)
        shifted_v = jnp.roll(mask, step, axis=0)
        shifted = jnp.where(horizontal, shifted_h, shifted_v)
        # wrap-around detection: a footprint cell would leave the grid iff the piece touches the border it moves towards
        edge_h = jnp.where(step > 0, mask[:, -1].any(), mask[:, 0].any())
        edge_v = jnp.where(step > 0, mask[-1, :].any(), mask[0, :].any())
        at_edge = jnp.where(horizontal, edge_h, edge_v)
        others = (grid != 0) & ~mask  # walls and other pieces
        blocked = (shifted & others).any()
        ok = is_piece & ~at_edge & ~blocked
        new_grid = jnp.where(ok, jnp.where(mask, jnp.int8(0), grid), grid)
        new_grid = jnp.where(ok, jnp.where(shifted, pid.astype(jnp.int8), new_grid), grid)
        return new_grid, ok

    def step(self, state: State, action: jax.Array) -> State:
        d = state.data
        new_grid, moved = self.move(d.grid, action)
        done_now = solved(new_grid)
        cfg = self._config
        reward = (-cfg.step_penalty + cfg.solve_reward * done_now.astype(jnp.float32)
                  + cfg.shaping_weight * (potential(new_grid) - potential(d.grid)))
        new_data = RushHourData(grid=new_grid, level=d.level, moves=d.moves)
        success = done_now.astype(jnp.float32)
        return State(
            data=new_data, obs=self._obs(new_data), reward=reward, done=success,
            metrics={**state.metrics, 'success': success, 'reward': reward},
            info={**state.info, 'target_goal': state.info['target_goal']},
        )


def render_ascii(grid):
    """Fogleman-style board string (o empty, x wall, A red, B.. pieces)."""
    g = np.asarray(grid)
    def ch(v):
        return 'o' if v == 0 else 'x' if v < 0 else chr(ord('A') + v - 1)
    return '\n'.join(''.join(ch(v) for v in row) for row in g)
