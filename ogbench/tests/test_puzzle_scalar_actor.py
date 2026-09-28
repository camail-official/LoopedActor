"""Tests for the puzzle_scalar-tokenizer path of GCFPRMActor / GCIQLFPRMAgent.

Mirrors tests/test_cube_actor.py for the raw-observation puzzle-3x3 setup
(TQL-style per-scalar tokens, full-observation goals, obs dim 55 = 19
proprio + 9 buttons x 4 features). Covers: agent construction, the
parameter tree (per-scalar tokenizer, no grid conv), one finite update under
both actor objectives, sample_actions APIs with grid_shape=None, the config
guards, and the layout sanity checks (cube layout rejects puzzle obs dims
and vice versa).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.cube_tokenizer import (
    FEATURES_PER_BUTTON,
    FEATURES_PER_CUBE,
    NUM_PROPRIO_FEATURES,
    CubeTokenizer,
)

OBS_DIM = NUM_PROPRIO_FEATURES + 9 * FEATURES_PER_BUTTON  # puzzle-3x3: 55
CUBE_OBS_DIM = NUM_PROPRIO_FEATURES + 2 * FEATURES_PER_CUBE  # cube-double: 37
ACTION_DIM = 5


def assert_info_finite(info):
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), (k, v)


def make_puzzle_batch(batch_size=4, seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'next_observations': rng.normal(size=(batch_size, OBS_DIM)).astype(
            np.float32
        ),
        'actions': rng.uniform(-1, 1, size=(batch_size, ACTION_DIM)).astype(
            np.float32
        ),
        'actor_goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'value_goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'rewards': -np.ones(batch_size, np.float32),
        'masks': np.ones(batch_size, np.float32),
    }


def make_puzzle_config(**overrides):
    from agents.gciql_fprm import get_config

    config = get_config()
    config['fprm_tokenizer'] = 'puzzle_scalar'
    config['fprm_use_grid_conv'] = False
    # Small model for test speed; head_dim 32/4 = 8.
    config['fprm_d_model'] = 32
    config['fprm_train_iters'] = 4
    for k, v in overrides.items():
        config[k] = v
    return config


def make_puzzle_agent(seed=0, **overrides):
    from agents.gciql_fprm import GCIQLFPRMAgent

    return GCIQLFPRMAgent.create(
        seed, make_puzzle_batch(), make_puzzle_config(**overrides)
    )


@pytest.fixture(scope='module')
def agent():
    return make_puzzle_agent()


@pytest.fixture(scope='module')
def klhalt_agent():
    return make_puzzle_agent(fprm_halt_on_policy_kl=1e-3)


# --- Layout sanity checks ----------------------------------------------------


def test_puzzle_layout_rejects_cube_obs_dim():
    tok = CubeTokenizer(32, CUBE_OBS_DIM, layout='puzzle')
    obs = jnp.zeros((2, CUBE_OBS_DIM))
    with pytest.raises(AssertionError):
        tok.init(jax.random.PRNGKey(0), obs)


def test_cube_layout_rejects_puzzle_obs_dim():
    # A button-block observation whose non-proprio part is not a multiple of the 9 cube features.
    dim = NUM_PROPRIO_FEATURES + 5 * FEATURES_PER_BUTTON
    tok = CubeTokenizer(32, dim, layout='cube')
    obs = jnp.zeros((2, dim))
    with pytest.raises(AssertionError):
        tok.init(jax.random.PRNGKey(0), obs)


# --- Construction and parameter tree ----------------------------------------


def test_grid_shape_is_none(agent):
    assert agent.config['grid_shape'] is None


def test_actor_uses_per_scalar_tokenizer(agent):
    tok = agent.network.params['modules_actor']['tokenizer']
    assert tok['positional_embedding'].shape == (OBS_DIM, 32)
    assert tok['state_embedding'].shape == (32,)
    assert tok['goal_embedding'].shape == (32,)
    assert 'grid_depthwise_conv' not in agent.network.params['modules_actor']['core']


def test_value_nets_use_full_obs_goals(agent):
    # Value MLP input = obs (55) + full-observation goal (55) = 110.
    kernel = agent.network.params['modules_value']['value_net']['Dense_0']['kernel']
    assert kernel.shape[0] == 2 * OBS_DIM


def test_grid_conv_rejected_for_puzzle_scalar():
    from agents.gciql_fprm import GCIQLFPRMAgent

    config = make_puzzle_config()
    config['fprm_use_grid_conv'] = True
    with pytest.raises(AssertionError):
        GCIQLFPRMAgent.create(0, make_puzzle_batch(), config)


def test_compact_goals_rejected_for_puzzle_scalar():
    from agents.gciql_fprm import GCIQLFPRMAgent

    batch = make_puzzle_batch()
    batch['actor_goals'] = batch['actor_goals'][:, :16]
    batch['value_goals'] = batch['value_goals'][:, :16]
    with pytest.raises(AssertionError):
        GCIQLFPRMAgent.create(0, batch, make_puzzle_config())


# --- Training ----------------------------------------------------------------


def test_one_update_finite_fixed_depth(agent):
    batch = make_puzzle_batch()
    new_agent, info = agent.update(batch)
    for k, v in info.items():
        assert np.isfinite(float(v)), (k, v)
    assert 'actor/mse' in info and 'actor/think_iters' not in info


def test_one_update_finite_klhalt(klhalt_agent):
    batch = make_puzzle_batch()
    new_agent, info = klhalt_agent.update(batch)
    assert_info_finite(info)
    assert 'actor/think_iters' in info


def test_one_update_finite_with_grad_clip():
    clipped = make_puzzle_agent(max_grad_norm=1.0, fprm_halt_on_policy_kl=1e-3)
    batch = make_puzzle_batch()
    new_agent, info = clipped.update(batch)
    assert_info_finite(info)


def test_one_update_finite_single_block():
    single = make_puzzle_agent(
        max_grad_norm=1.0,
        fprm_num_blocks=1,
        fprm_train_iters=1,
    )
    batch = make_puzzle_batch()
    new_agent, info = single.update(batch)
    assert_info_finite(info)


def test_update_changes_tokenizer_params(agent):
    batch = make_puzzle_batch()
    new_agent, _ = agent.update(batch)
    before = agent.network.params['modules_actor']['tokenizer']
    after = new_agent.network.params['modules_actor']['tokenizer']
    changed = jax.tree_util.tree_map(
        lambda a, b: bool(jnp.any(a != b)), before, after
    )
    assert any(jax.tree_util.tree_leaves(changed))


# --- Evaluation APIs ---------------------------------------------------------


def test_sample_actions_unbatched(agent):
    obs = np.zeros(OBS_DIM, np.float32)
    goal = np.zeros(OBS_DIM, np.float32)
    action = agent.sample_actions(obs, goal, seed=jax.random.PRNGKey(0))
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(np.asarray(action)) <= 1.0)


def test_sample_actions_with_info_policy_kl(klhalt_agent):
    batch = make_puzzle_batch(batch_size=3)
    actions, info = klhalt_agent.sample_actions_with_info(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=None,
        halt_mode='policy_kl',
        max_iters=16,
    )
    assert actions.shape == (3, ACTION_DIM)
    iters = np.asarray(info['iterations'])
    assert np.all(iters >= 1) and np.all(iters <= 16)


# --- Existing paths untouched ------------------------------------------------


def test_default_config_stays_puzzle():
    from agents.gciql_fprm import get_config

    assert get_config()['fprm_tokenizer'] == 'puzzle'


def test_cube_tokenizer_default_layout_is_cube():
    assert CubeTokenizer(32, CUBE_OBS_DIM).layout == 'cube'
