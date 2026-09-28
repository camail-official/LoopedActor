"""Cube tokenizer tests: shapes, goal-modality semantics, parameter layout."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.cube_tokenizer import (
    FEATURES_PER_CUBE,
    NUM_PROPRIO_FEATURES,
    PREFIX_LEN,
    CubeTokenizer,
    num_cubes_from_obs_dim,
)

H = 32
D_DOUBLE = NUM_PROPRIO_FEATURES + 2 * FEATURES_PER_CUBE  # 37
D_TRIPLE = NUM_PROPRIO_FEATURES + 3 * FEATURES_PER_CUBE  # 46
ATOL = 1e-6


def make_inputs(batch, obs_dim, seed=0):
    rng = np.random.default_rng(seed)
    obs = jnp.asarray(rng.normal(size=(batch, obs_dim)), jnp.float32)
    goal = jnp.asarray(rng.normal(size=(batch, obs_dim)), jnp.float32)
    return obs, goal


@pytest.fixture(scope='module')
def tokenizer_and_params():
    tok = CubeTokenizer(d_model=H, obs_dim=D_DOUBLE)
    obs, goal = make_inputs(2, D_DOUBLE)
    params = tok.init(jax.random.PRNGKey(0), obs, goal)
    return tok, params


def test_obs_dim_inference():
    assert num_cubes_from_obs_dim(D_DOUBLE) == 2
    assert num_cubes_from_obs_dim(D_TRIPLE) == 3
    with pytest.raises(AssertionError):
        num_cubes_from_obs_dim(D_DOUBLE + 1)
    with pytest.raises(AssertionError):
        num_cubes_from_obs_dim(NUM_PROPRIO_FEATURES)


def test_shapes_with_and_without_goal(tokenizer_and_params):
    tok, params = tokenizer_and_params
    obs, goal = make_inputs(3, D_DOUBLE, seed=1)

    tokens, mask = tok.apply(params, obs, goal)
    assert tokens.shape == (3, PREFIX_LEN + 2 * D_DOUBLE, H)
    assert mask.shape == tokens.shape[:2] and bool(mask.all())

    tokens_ng, mask_ng = tok.apply(params, obs)
    assert tokens_ng.shape == (3, PREFIX_LEN + D_DOUBLE, H)
    assert bool(mask_ng.all())
    assert np.all(np.isfinite(np.asarray(tokens)))


def test_param_tree_independent_of_goal_arg():
    tok = CubeTokenizer(d_model=H, obs_dim=D_DOUBLE)
    obs, goal = make_inputs(2, D_DOUBLE)
    with_goal = tok.init(jax.random.PRNGKey(0), obs, goal)
    without_goal = tok.init(jax.random.PRNGKey(0), obs)
    tree_a = jax.tree.map(lambda x: (x.shape, x.dtype), with_goal)
    tree_b = jax.tree.map(lambda x: (x.shape, x.dtype), without_goal)
    assert tree_a == tree_b


def test_goal_tokenized_exactly_like_state(tokenizer_and_params):
    """Same input -> goal tokens differ from state tokens only by the
    (constant) modality-embedding difference: shared scalar projection and
    shared positional table."""
    tok, params = tokenizer_and_params
    obs, _ = make_inputs(2, D_DOUBLE, seed=2)

    tokens, _ = tok.apply(params, obs, obs)
    state_tokens = tokens[:, PREFIX_LEN:PREFIX_LEN + D_DOUBLE]
    goal_tokens = tokens[:, PREFIX_LEN + D_DOUBLE:]

    diff = np.asarray(goal_tokens - state_tokens)
    # Constant across batch and positions...
    assert np.allclose(diff, diff[0, 0], atol=ATOL)
    # ...and equal to goal_embedding - state_embedding.
    p = params['params']
    expected = np.asarray(p['goal_embedding'] - p['state_embedding'])
    assert np.allclose(diff[0, 0], expected, atol=ATOL)


def test_state_tokens_unchanged_by_goal_presence(tokenizer_and_params):
    tok, params = tokenizer_and_params
    obs, goal = make_inputs(2, D_DOUBLE, seed=3)
    with_goal, _ = tok.apply(params, obs, goal)
    without_goal, _ = tok.apply(params, obs)
    np.testing.assert_allclose(
        np.asarray(with_goal[:, : PREFIX_LEN + D_DOUBLE]),
        np.asarray(without_goal),
        atol=ATOL,
    )


def test_readout_is_input_independent(tokenizer_and_params):
    tok, params = tokenizer_and_params
    obs_a, goal_a = make_inputs(2, D_DOUBLE, seed=4)
    obs_b, goal_b = make_inputs(2, D_DOUBLE, seed=5)
    tokens_a, _ = tok.apply(params, obs_a, goal_a)
    tokens_b, _ = tok.apply(params, obs_b, goal_b)
    np.testing.assert_allclose(
        np.asarray(tokens_a[:, 0]), np.asarray(tokens_b[:, 0]), atol=ATOL
    )


def test_wrong_obs_dim_asserts(tokenizer_and_params):
    tok, params = tokenizer_and_params
    obs, _ = make_inputs(2, D_TRIPLE)
    with pytest.raises(AssertionError):
        tok.apply(params, obs)


def test_triple_cube_instance():
    tok = CubeTokenizer(d_model=H, obs_dim=D_TRIPLE)
    obs, goal = make_inputs(2, D_TRIPLE)
    params = tok.init(jax.random.PRNGKey(0), obs, goal)
    tokens, _ = tok.apply(params, obs, goal)
    assert tokens.shape == (2, PREFIX_LEN + 2 * D_TRIPLE, H)
    assert (
        params['params']['positional_embedding'].shape == (D_TRIPLE, H)
    )
