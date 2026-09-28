"""Boxoban environment + bank-parser tests (CPU, a few seconds)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_scripts.build_boxoban_banks import parse_levels  # noqa: E402
from envs.sokoban_env import DATA_DIR, SokobanData, SokobanEnv, default_config  # noqa: E402
from envs.utils import State  # noqa: E402

LEVEL = """; 0
##########
######## #
#######  #
#######$ #
#######  #
######. .#
###### $.#
#####  #$#
#####. $@#
##########
"""


def test_parse_level():
    (walls, boxes, targets, player), = list(parse_levels(LEVEL))
    assert walls.shape == (10, 10) and walls[0].all() and walls[:, 0].all()
    assert boxes.sum() == 4 and targets.sum() == 4
    assert tuple(player) == (8, 8) and not walls[8, 8]


def _state(env, walls, boxes, targets, player):
    d = SokobanData(walls=jnp.asarray(walls), boxes=jnp.asarray(boxes), targets=jnp.asarray(targets), player=jnp.asarray(player, jnp.int32),
                    level=jnp.int32(0), on_target=jnp.sum(jnp.asarray(boxes) & jnp.asarray(targets)).astype(jnp.int32))
    return State(data=d, obs=env._obs(d), reward=jnp.float32(0), done=jnp.float32(0),
                 metrics={'success': 0.0, 'reward': 0.0, 'boxes_on_target': 0.0},
                 info={'rng': jax.random.PRNGKey(0), 'target_goal': jnp.zeros((0,), jnp.float32)})


def test_dynamics_and_reward():
    """Hand-built 10x10 room: one box, one target; up = 0, down = 1, left = 2, right = 3."""
    walls = np.ones((10, 10), bool); walls[1:9, 1:9] = False
    boxes = np.zeros((10, 10), bool); boxes[4, 4] = True
    targets = np.zeros((10, 10), bool); targets[4, 6] = True
    cfg = default_config(); cfg.train_split = cfg.eval_split = 'unfiltered_test'
    if not os.path.exists(os.path.join(DATA_DIR, 'unfiltered_test.npz')):
        pytest.skip('data/boxoban/unfiltered_test.npz missing (python data_scripts/build_boxoban_banks.py)')
    env = SokobanEnv(cfg); step = jax.jit(env.step)
    s = _state(env, walls, boxes, targets, (4, 3))
    assert s.obs.shape == (400,) and env.observation_size == 400 and env.action_size == 4
    s = step(s, 0)                              # up into free space: plain move
    assert tuple(np.asarray(s.data.player)) == (3, 3) and float(s.reward) == pytest.approx(-0.1)
    s = step(s, 0); s = step(s, 0)              # two more ups: (2,3) then bump the wall at row 0 -> stays at (1,3)
    assert tuple(np.asarray(s.data.player)) == (1, 3)
    s = _state(env, walls, boxes, targets, (4, 3))
    s = step(s, 3)                              # right: push box (4,4) -> (4,5), player to (4,4)
    assert bool(s.data.boxes[4, 5]) and tuple(np.asarray(s.data.player)) == (4, 4) and float(s.done) == 0
    s = step(s, 3)                              # push onto the target: +1 box reward + 10 solve - 0.1 step, episode done
    assert bool(s.data.boxes[4, 6]) and float(s.reward) == pytest.approx(10.9) and float(s.done) == 1
    # pushing against a wall does not move box or player
    walls2 = walls.copy(); boxes2 = np.zeros((10, 10), bool); boxes2[4, 8] = True
    s = _state(env, walls2, boxes2, targets, (4, 7)); s = step(s, 3)
    assert bool(s.data.boxes[4, 8]) and tuple(np.asarray(s.data.player)) == (4, 7)


def test_bank_reset_shapes():
    if not os.path.exists(os.path.join(DATA_DIR, 'unfiltered_test.npz')):
        pytest.skip('data/boxoban/unfiltered_test.npz missing (python data_scripts/build_boxoban_banks.py)')
    cfg = default_config(); cfg.train_split = cfg.eval_split = 'unfiltered_test'
    env = SokobanEnv(cfg)
    assert env.num_eval_levels == 1000
    s = jax.jit(env.eval_reset)(jax.random.PRNGKey(1))
    assert s.obs.shape == (400,) and int(jnp.sum(s.data.boxes)) == 4 and int(jnp.sum(s.data.targets)) == 4
