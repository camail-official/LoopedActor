"""Tests for the feedforward-stack (untied blocks) baselines of GCFPRMActor.

Covers: parameter-tree identities (num_blocks=1 is iso-params with the
looped actor; num_blocks=8 adds exactly 7 extra untied cores), one finite
update for both baselines, the official
sample_actions API, sample_actions_with_info in fixed mode (constant
iteration count), and the config guards (feedforward + klhalt is invalid).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_cube_actor import (
    ACTION_DIM,
    OBS_DIM,
    make_cube_agent,
    make_cube_batch,
)


def n_params(tree):
    return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(tree))


def actor_params(agent):
    return agent.network.params['modules_actor']


def test_single_block_is_iso_params():
    looped = make_cube_agent()
    single = make_cube_agent(fprm_num_blocks=1, fprm_train_iters=1)
    lp, sp = actor_params(looped), actor_params(single)
    assert 'core' in lp and 'core' not in sp
    assert 'core_0' in sp
    # Identical structure block-for-block -> identical actor param count.
    l_shapes = jax.tree_util.tree_map(jnp.shape, lp['core'])
    s_shapes = jax.tree_util.tree_map(jnp.shape, sp['core_0'])
    assert l_shapes == s_shapes
    assert n_params(lp) == n_params(sp)


def test_multi_block_params_are_untied():
    looped = make_cube_agent()
    multi = make_cube_agent(fprm_num_blocks=8)
    lp, mp = actor_params(looped), actor_params(multi)
    for i in range(8):
        assert f'core_{i}' in mp
    core_params = n_params(lp['core'])
    assert n_params(mp) == n_params(lp) + 7 * core_params
    # Untied: block params must be independently initialized (not shared).
    c0 = jax.tree_util.tree_leaves(mp['core_0'])
    c1 = jax.tree_util.tree_leaves(mp['core_1'])
    assert any(
        not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(c0, c1)
    )


@pytest.mark.parametrize(
    'overrides',
    [
        dict(fprm_num_blocks=1, fprm_train_iters=1),
        dict(fprm_num_blocks=8),
    ],
)
def test_one_update_finite(overrides):
    agent = make_cube_agent(max_grad_norm=1.0, **overrides)
    batch = make_cube_batch(batch_size=8, seed=1)
    agent, info = agent.update(batch)
    assert np.isfinite(info['actor/actor_loss'])
    assert 'actor/bc_loss' in info
    for k, v in info.items():
        assert np.all(np.isfinite(v)), k


def test_sample_actions_and_info():
    agent = make_cube_agent(fprm_num_blocks=8)
    batch = make_cube_batch(batch_size=4, seed=2)
    a = agent.sample_actions(
        batch['observations'][0], batch['actor_goals'][0],
        seed=jax.random.PRNGKey(0), temperature=0.0,
    )
    assert a.shape == (ACTION_DIM,) and np.all(np.isfinite(a))
    acts, info = agent.sample_actions_with_info(
        batch['observations'], batch['actor_goals'],
        grid_shape=None, deterministic=True, halt_mode='fixed', num_iters=4,
    )
    assert acts.shape == (4, ACTION_DIM) and np.all(np.isfinite(acts))
    # Fixed compute: iteration count is the block count, num_iters ignored.
    assert np.all(np.asarray(info['iterations']) == 8)
    assert np.all(np.isfinite(np.asarray(info['rms_residual'])))


def test_feedforward_rejects_klhalt():
    with pytest.raises(AssertionError):
        make_cube_agent(fprm_num_blocks=8, fprm_halt_on_policy_kl=1e-3)
