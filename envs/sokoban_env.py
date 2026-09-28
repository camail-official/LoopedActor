"""Sokoban (Boxoban levels) environment in JAX.

Boxoban (Guez et al. 2018): 10x10 rooms, 4 boxes, 4 targets, wall border. Level
banks are parsed from google-deepmind/boxoban-levels into
data/boxoban/<split>.npz (walls/boxes/targets [N,10,10] uint8, player [N,2]).
Splits: unfiltered_{train,valid,test}, medium_{train,valid}, hard.

Training resets sample a level from `train_split`; evaluation resets
(`eval_reset`) sample from `eval_split` - by default the HARD set, so the
"test" metrics in ppo.py directly measure transfer to the difficulty tier
that defeats feed-forward policies.

Observation: per-cell channels [wall, box, target, player], cell-major flat
vector of size m*n*4 (the actor reshapes it to [m*n, 4] tokens). No goal
vector (goal_size 0): targets are part of the observation.
Actions: 0 up, 1 down, 2 left, 3 right (pushing is implicit).
Reward (gym-sokoban / Guez et al.): -0.1 per step, +1 for a box pushed onto a
target, -1 for a box pushed off, +10 when all boxes are on targets (episode
ends). Episode length 120 (set in make_env).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from ml_collections import config_dict

from envs.utils import State

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'boxoban')
DIRS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)  # up, down, left, right
OBS_CHANNELS = 4


@struct.dataclass
class SokobanData:
    walls: jax.Array    # [m, n] bool
    boxes: jax.Array    # [m, n] bool
    targets: jax.Array  # [m, n] bool
    player: jax.Array   # [2] int32 (row, col)
    level: jax.Array    # () int32 index into the bank it was drawn from
    on_target: jax.Array  # () int32 number of boxes on targets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        m=10,
        n=10,
        episode_length=120,
        train_split='unfiltered_train',
        eval_split='hard',
        max_train_levels=0,  # 0 = all levels of the split
        max_eval_levels=0,
        step_penalty=0.1,
        box_reward=1.0,
        solve_reward=10.0,
    )


def load_bank(split, max_levels=0):
    z = np.load(os.path.join(DATA_DIR, f'{split}.npz'))
    n = len(z['walls']) if max_levels <= 0 else min(max_levels, len(z['walls']))
    return dict(walls=jnp.asarray(z['walls'][:n].astype(bool)), boxes=jnp.asarray(z['boxes'][:n].astype(bool)),
                targets=jnp.asarray(z['targets'][:n].astype(bool)), player=jnp.asarray(z['player'][:n].astype(np.int32)))


class SokobanEnv:
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        self._config = config
        self.m, self.n = config.m, config.n
        self.grid_size = self.m * self.n
        self.obs_channels = OBS_CHANNELS
        self.train_bank = load_bank(config.train_split, config.max_train_levels)
        self.eval_bank = load_bank(config.eval_split, config.max_eval_levels)
        self.num_train_levels = int(self.train_bank['walls'].shape[0])
        self.num_eval_levels = int(self.eval_bank['walls'].shape[0])

    @property
    def action_size(self): return 4

    @property
    def observation_size(self): return self.grid_size * OBS_CHANNELS

    @property
    def goal_size(self): return 0

    def _obs(self, data: SokobanData):
        player_grid = jnp.zeros((self.m, self.n), bool).at[data.player[0], data.player[1]].set(True)
        cells = jnp.stack([data.walls, data.boxes, data.targets, player_grid], axis=-1)  # [m, n, 4]
        return cells.reshape(-1).astype(jnp.float32)

    def _reset_from_bank(self, rng, bank, num_levels):
        idx = jax.random.randint(rng, (), 0, num_levels)
        walls, boxes, targets, player = bank['walls'][idx], bank['boxes'][idx], bank['targets'][idx], bank['player'][idx]
        data = SokobanData(walls=walls, boxes=boxes, targets=targets, player=player, level=idx,
                           on_target=jnp.sum(boxes & targets).astype(jnp.int32))
        return State(
            data=data, obs=self._obs(data),
            reward=jnp.array(0.0, jnp.float32), done=jnp.array(0.0, jnp.float32),
            metrics={'success': 0.0, 'reward': 0.0, 'boxes_on_target': data.on_target.astype(jnp.float32)},
            info={'rng': rng, 'target_goal': jnp.zeros((0,), jnp.float32)},
        )

    def reset(self, rng: jax.Array) -> State:
        return self._reset_from_bank(rng, self.train_bank, self.num_train_levels)

    def eval_reset(self, rng: jax.Array) -> State:
        return self._reset_from_bank(rng, self.eval_bank, self.num_eval_levels)

    def step(self, state: State, action: jax.Array) -> State:
        d = state.data
        delta = DIRS[action]
        p1 = jnp.clip(d.player + delta, 0, jnp.array([self.m - 1, self.n - 1]))
        p2 = jnp.clip(p1 + delta, 0, jnp.array([self.m - 1, self.n - 1]))
        wall1 = d.walls[p1[0], p1[1]]
        box1 = d.boxes[p1[0], p1[1]]
        blocked2 = d.walls[p2[0], p2[1]] | d.boxes[p2[0], p2[1]] | jnp.all(p2 == p1)
        push = box1 & ~blocked2
        move = ~wall1 & (~box1 | push)
        new_boxes = jnp.where(push, d.boxes.at[p1[0], p1[1]].set(False).at[p2[0], p2[1]].set(True), d.boxes)
        new_player = jnp.where(move, p1, d.player)
        on_target = jnp.sum(new_boxes & d.targets).astype(jnp.int32)
        solved = on_target == jnp.sum(d.targets)
        cfg = self._config
        reward = (-cfg.step_penalty + cfg.box_reward * (on_target - d.on_target).astype(jnp.float32)
                  + cfg.solve_reward * solved.astype(jnp.float32))
        new_data = SokobanData(walls=d.walls, boxes=new_boxes, targets=d.targets, player=new_player, level=d.level, on_target=on_target)
        success = solved.astype(jnp.float32)
        return State(
            data=new_data, obs=self._obs(new_data), reward=reward, done=success,
            metrics={**state.metrics, 'success': success, 'reward': reward, 'boxes_on_target': on_target.astype(jnp.float32)},
            info={**state.info, 'target_goal': state.info['target_goal']},
        )
