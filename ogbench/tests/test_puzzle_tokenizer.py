"""Tokenizer and cross-size shape tests (spec Section 20.3, tests 11-19)."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.fprm import FPRMConfig, FPRMCore, grid_conv_tokens
from utils.puzzle_tokenizer import (
    FEATURES_PER_BUTTON,
    NUM_CELL_FEATURES,
    NUM_ROBOT_FEATURES,
    PREFIX_LEN,
    PuzzleTokenizer,
    SPACING_HALF,
    X_CENTER,
    build_cell_features,
    cell_geometry,
    effector_xyz_from_robot,
)

H = 32


def make_obs_goal(B, R, C, rng=None):
    rng = np.random.default_rng(0 if rng is None else rng)
    obs = rng.normal(size=(B, NUM_ROBOT_FEATURES + FEATURES_PER_BUTTON * R * C))
    # Make the one-hot slots valid one-hots.
    cells = obs[:, NUM_ROBOT_FEATURES:].reshape(B, R * C, FEATURES_PER_BUTTON)
    bits = rng.integers(0, 2, size=(B, R * C))
    cells[:, :, 0] = 1.0 - bits
    cells[:, :, 1] = bits
    obs[:, NUM_ROBOT_FEATURES:] = cells.reshape(B, -1)
    goal = rng.integers(0, 2, size=(B, R * C)).astype(np.float64)
    return jnp.asarray(obs, jnp.float32), jnp.asarray(goal, jnp.float32), bits


# --- Tests 11-13: shapes across grid sizes -----------------------------------


@pytest.mark.parametrize(
    'R,C,obs_dim,goal_dim,seq',
    [
        (3, 3, 55, 9, 11),
        (4, 4, 83, 16, 18),
        (4, 5, 99, 20, 22),
        (4, 6, 115, 24, 26),
    ],
)
def test_tokenizer_shapes(R, C, obs_dim, goal_dim, seq):
    B = 3
    obs, goal, _ = make_obs_goal(B, R, C)
    assert obs.shape == (B, obs_dim)
    assert goal.shape == (B, goal_dim)

    tok = PuzzleTokenizer(d_model=H)
    variables = tok.init(jax.random.PRNGKey(0), obs, goal, (R, C))
    tokens, mask = tok.apply(variables, obs, goal, (R, C))
    assert tokens.shape == (B, seq, H)
    assert mask.shape == (B, seq)
    assert bool(jnp.all(mask))


def test_target_onehot_shape():
    B, R, C = 2, 3, 3
    obs, goal, _ = make_obs_goal(B, R, C)
    target_idx = goal.reshape(B, R, C).astype(jnp.int32)
    target_onehot = jax.nn.one_hot(target_idx, 2, dtype=jnp.float32)
    assert target_onehot.shape == (B, 3, 3, 2)  # not (B, 3, 3, 1, 2)


def test_same_parameter_tree_across_sizes():
    """Test 13/19: one parameter tree initialized on 3x3 applies to all sizes."""
    obs3, goal3, _ = make_obs_goal(2, 3, 3)
    tok = PuzzleTokenizer(d_model=H)
    variables = tok.init(jax.random.PRNGKey(0), obs3, goal3, (3, 3))

    for R, C in [(4, 5), (4, 6)]:
        obs, goal, _ = make_obs_goal(2, R, C)
        tokens, mask = tok.apply(variables, obs, goal, (R, C))
        assert tokens.shape == (2, PREFIX_LEN + R * C, H)

    # No parameter shape depends on the grid size.
    shapes = jax.tree_util.tree_map(lambda x: x.shape, variables['params'])
    flat = jax.tree_util.tree_leaves(shapes, is_leaf=lambda x: isinstance(x, tuple))
    for shape in flat:
        assert 9 not in shape and 16 not in shape and 20 not in shape, shapes


# --- Test 14: row-major mapping ----------------------------------------------


@pytest.mark.parametrize('R,C', [(3, 3), (4, 5)])
def test_row_major_mapping(R, C):
    B = 1
    obs, goal, bits = make_obs_goal(B, R, C)
    _, cell_features = build_cell_features(obs, goal, (R, C))
    # Button feature i (flat) must land at cell (i // C, i % C).
    for i in range(R * C):
        r, c = i // C, i % C
        # current_onehot[..., 1] is the current bit.
        assert float(cell_features[0, r, c, 1]) == float(bits[0, i])
        # button_joint_pos is feature index 5 in the cell vector.
        flat_cell = obs[0, NUM_ROBOT_FEATURES + i * 4 + 2]
        assert float(cell_features[0, r, c, 5]) == pytest.approx(float(flat_cell))


# --- Test 15: physical coordinate formula ------------------------------------


@pytest.mark.parametrize('R,C', [(3, 3), (4, 5), (4, 6)])
def test_physical_coordinates(R, C):
    _, physical_xy, _ = cell_geometry((R, C))
    for r in range(R):
        for c in range(C):
            expect_x = 0.425 - 0.05 * (R - 1) + 0.1 * r
            expect_y = -0.05 * (C - 1) + 0.1 * c
            assert float(physical_xy[r, c, 0]) == pytest.approx(expect_x, abs=1e-6)
            assert float(physical_xy[r, c, 1]) == pytest.approx(expect_y, abs=1e-6)

    # Spec ranges: 3x3 x in [0.325, 0.525], y in [-0.10, 0.10].
    if (R, C) == (3, 3):
        assert float(physical_xy[..., 0].min()) == pytest.approx(0.325)
        assert float(physical_xy[..., 0].max()) == pytest.approx(0.525)
        assert float(physical_xy[..., 1].min()) == pytest.approx(-0.10)
        assert float(physical_xy[..., 1].max()) == pytest.approx(0.10)


def test_normalized_coordinates_bounds():
    for R, C in [(3, 3), (4, 6)]:
        norm_rc, _, _ = cell_geometry((R, C))
        np.testing.assert_allclose(np.asarray(norm_rc).min(), -1.0)
        np.testing.assert_allclose(np.asarray(norm_rc).max(), 1.0)


# --- Test 16: boundary flags -------------------------------------------------


def test_boundary_flags():
    R, C = 4, 5
    _, _, flags = cell_geometry((R, C))  # order: top, bottom, left, right
    corners = {
        (0, 0): [1, 0, 1, 0],
        (0, C - 1): [1, 0, 0, 1],
        (R - 1, 0): [0, 1, 1, 0],
        (R - 1, C - 1): [0, 1, 0, 1],
    }
    for (r, c), expect in corners.items():
        np.testing.assert_array_equal(np.asarray(flags[r, c]), expect)
    # Interior cell.
    np.testing.assert_array_equal(np.asarray(flags[1, 2]), [0, 0, 0, 0])
    # Edge (non-corner) cells.
    np.testing.assert_array_equal(np.asarray(flags[0, 2]), [1, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(flags[2, 0]), [0, 0, 1, 0])


# --- Test 17: no row wrap in the grid convolution ----------------------------


def test_conv_no_row_wrap():
    """An impulse at the end of one row must not reach the next row's first
    cell (they are not 2-D neighbors), unlike a flattened 1-D convolution."""
    R, C, Hc = 3, 4, 4
    B = 1
    L = PREFIX_LEN + R * C

    conv = nn.Conv(
        features=Hc,
        kernel_size=(3, 3),
        padding='SAME',
        feature_group_count=Hc,
        use_bias=False,
        kernel_init=nn.initializers.ones,  # every neighbor contributes
    )
    z = jnp.zeros((B, L, Hc))
    # Impulse at cell (0, C-1) = flat cell index C-1.
    z = z.at[0, PREFIX_LEN + (C - 1), :].set(1.0)

    variables = conv.init(jax.random.PRNGKey(0), z[:, PREFIX_LEN:].reshape(B, R, C, Hc))
    out = grid_conv_tokens(
        lambda g: conv.apply(variables, g), z, (R, C), PREFIX_LEN
    )

    grid_out = np.asarray(out[0, PREFIX_LEN:].reshape(R, C, Hc))
    # 2-D neighbors of (0, C-1): (0, C-2), (1, C-2), (1, C-1) and itself.
    assert grid_out[0, C - 1, 0] != 0
    assert grid_out[1, C - 1, 0] != 0
    assert grid_out[0, C - 2, 0] != 0
    # NOT a neighbor: next row's first cell (1, 0) (flat offset +1 from
    # (0, C-1) in flattened order, which a 1-D conv would corrupt).
    assert grid_out[1, 0, 0] == 0
    # Prefix tokens unchanged.
    np.testing.assert_array_equal(np.asarray(out[0, :PREFIX_LEN]), 0.0)


# --- Test 18: environment round trip (obs bits, goal dtype, effector) --------


@pytest.mark.slow
def test_env_round_trip_3x3():
    import ogbench

    env = ogbench.make_env_and_datasets(
        'puzzle-3x3-play-oraclerep-v0', env_only=True
    )
    obs, info = env.reset(seed=0, options={'task_id': 1, 'render_goal': False})

    # Goal comes back float64 and must be cast (test 24).
    assert np.asarray(info['goal']).dtype == np.float64
    goal = jnp.asarray(np.asarray(info['goal'], dtype=np.float32))
    obs_b = jnp.asarray(np.asarray(obs, dtype=np.float32))[None]
    goal_b = goal[None]

    # Take a few steps to move the arm and get nonzero velocities.
    for _ in range(10):
        obs, _, _, _, info = env.step(env.action_space.sample() * 0.3)
    obs_b = jnp.asarray(np.asarray(obs, dtype=np.float32))[None]

    _, cell_features = build_cell_features(obs_b, goal_b, (3, 3))
    # Decoded current bit must match info['button_states'] (row-major).
    decoded_bits = np.asarray(cell_features[0, :, :, 1]).reshape(-1)
    np.testing.assert_array_equal(
        decoded_bits.astype(np.int64), np.asarray(info['button_states'])
    )

    # Effector recovery: obs[12:15] = (effector_pos - [0.425, 0, 0]) * 10.
    ob_info = env.unwrapped.compute_ob_info()
    recovered = np.asarray(effector_xyz_from_robot(obs_b[:, :19])[0])
    np.testing.assert_allclose(
        recovered, np.asarray(ob_info['proprio/effector_pos']), atol=1e-5
    )

    # Physical button coordinates match the simulator's button site x/y.
    _, physical_xy, _ = cell_geometry((3, 3))
    for i in range(9):
        site_id = env.unwrapped._button_site_ids[i]
        site_pos = env.unwrapped._data.site_xpos[site_id]
        r, c = i // 3, i % 3
        np.testing.assert_allclose(
            np.asarray(physical_xy[r, c]), site_pos[:2], atol=1e-3
        )


# --- Test 19: cross-size parameter reuse through the full core ---------------


class TokenizerCore(nn.Module):
    config: FPRMConfig

    @nn.compact
    def __call__(self, obs, goal, grid_shape):
        tokens, mask = PuzzleTokenizer(self.config.d_model, name='tokenizer')(
            obs, goal, grid_shape
        )
        core = FPRMCore(self.config, name='core')
        return core(tokens, tokens, mask, grid_shape, PREFIX_LEN)


def test_parameter_reuse_full_core():
    cfg = FPRMConfig(d_model=H)
    obs3, goal3, _ = make_obs_goal(2, 3, 3)
    module = TokenizerCore(cfg)
    variables = module.init(jax.random.PRNGKey(0), obs3, goal3, (3, 3))
    n_params_3x3 = sum(x.size for x in jax.tree_util.tree_leaves(variables))

    out3 = module.apply(variables, obs3, goal3, (3, 3))
    assert out3.shape == (2, 11, H)

    obs4, goal4, _ = make_obs_goal(2, 4, 5)
    out4 = module.apply(variables, obs4, goal4, (4, 5))
    assert out4.shape == (2, 22, H)

    # Applying to a larger grid must not require new parameters: re-initializing
    # on it yields the identical tree structure and shapes.
    variables4 = module.init(jax.random.PRNGKey(0), obs4, goal4, (4, 5))
    shapes3 = jax.tree_util.tree_map(lambda x: x.shape, variables)
    shapes4 = jax.tree_util.tree_map(lambda x: x.shape, variables4)
    assert shapes3 == shapes4
    n_params_4x5 = sum(x.size for x in jax.tree_util.tree_leaves(variables4))
    assert n_params_3x3 == n_params_4x5


def test_deterministic_outputs():
    """Identical inputs give identical outputs (Stage 0 gate)."""
    cfg = FPRMConfig(d_model=H)
    obs, goal, _ = make_obs_goal(2, 3, 3)
    module = TokenizerCore(cfg)
    variables = module.init(jax.random.PRNGKey(0), obs, goal, (3, 3))
    out1 = module.apply(variables, obs, goal, (3, 3))
    out2 = module.apply(variables, obs, goal, (3, 3))
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_cell_feature_count():
    obs, goal, _ = make_obs_goal(1, 3, 3)
    _, cell_features = build_cell_features(obs, goal, (3, 3))
    assert cell_features.shape[-1] == NUM_CELL_FEATURES == 17


def test_mismatch_feature():
    obs, goal, bits = make_obs_goal(1, 3, 3)
    _, cell_features = build_cell_features(obs, goal, (3, 3))
    target = np.asarray(goal[0]).reshape(3, 3)
    current = np.asarray(bits[0]).reshape(3, 3)
    mismatch = np.asarray(cell_features[0, :, :, 4])
    np.testing.assert_array_equal(mismatch, np.abs(current - target))


def test_relative_xy():
    obs, goal, _ = make_obs_goal(1, 3, 3)
    robot = obs[:, :NUM_ROBOT_FEATURES]
    effector = effector_xyz_from_robot(robot)[0, :2]
    _, cell_features = build_cell_features(obs, goal, (3, 3))
    _, physical_xy, _ = cell_geometry((3, 3))
    rel = np.asarray(cell_features[0, :, :, 11:13])
    np.testing.assert_allclose(
        rel, np.asarray(physical_xy) - np.asarray(effector)[None, None, :],
        atol=1e-6,
    )
