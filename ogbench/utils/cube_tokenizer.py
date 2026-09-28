"""Per-scalar tokenizer (TQL-style) for OGBench manipspace environments.

Builds the actor token sequence
    [READOUT, s_0, s_1, ..., s_{D-1}]                     (reward-based)
    [READOUT, s_0, ..., s_{D-1}, g_0, ..., g_{D-1}]       (goal-conditioned)
from the raw state observation and (optionally) a goal observation.

Following TQL ("Scaling Q-Functions with Transformers by Preventing
Attention Collapse", arXiv:2602.01439), every scalar dimension of the
observation becomes one token: a single shared Dense(1 -> H) projects each
scalar, a learnable positional embedding distinguishes the dimensions, and a
learnable modality embedding distinguishes state tokens from goal tokens.
Goals are tokenized exactly like observations (same scalar projection, same
positional table) so a goal-conditioned policy sees goal state through the
identical representation; only the modality embedding differs.

Unlike the structured PuzzleTokenizer, the positional table makes parameter
shapes depend on the observation dimension, so one parameter tree does NOT
transfer across entity counts (cube-double D=37, cube-triple D=46;
puzzle-3x3 D=55). This is the TQL trade-off: no
hand-crafted per-entity features, fully generic within one environment.

The tokenizer itself is layout-agnostic; the `layout` field only selects
which observation-dimension sanity check applies.

Observation layouts (verified against the pinned OGBench source,
ogbench/manipspace/envs/{cube_env,puzzle_env}.py::compute_observation):
    obs[:19]  robot/proprio features (identical across manipspace envs):
        [0:6]   joint_pos
        [6:12]  joint_vel
        [12:15] (effector_pos - [0.425, 0, 0]) * 10
        [15]    cos(effector_yaw)
        [16]    sin(effector_yaw)
        [17]    gripper_opening * 3
        [18]    gripper_contact
    layout='cube':   obs[19:] per-cube [scaled pos (3), quat (4), cos(yaw),
                     sin(yaw)], 9 features per cube, cube order 0..k-1.
    layout='puzzle': obs[19:] per-button [state one-hot (2),
                     press pos * 120 (1), press vel (1)], 4 features per
                     button, row-major button order (3x3 -> 9 buttons,
                     D = 19 + 36 = 55).
"""

import flax.linen as nn
import jax.numpy as jnp

NUM_PROPRIO_FEATURES = 19
FEATURES_PER_CUBE = 9
FEATURES_PER_BUTTON = 4
PREFIX_LEN = 1  # [READOUT]
SCENE_OBS_DIM = NUM_PROPRIO_FEATURES + FEATURES_PER_CUBE + 2 * FEATURES_PER_BUTTON + 4  # 40


def default_init(scale=1.0):
    """OGBench default kernel initializer (fan-average uniform)."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def num_cubes_from_obs_dim(obs_dim):
    """Infer the cube count from a state observation dimension."""
    residual = obs_dim - NUM_PROPRIO_FEATURES
    assert residual > 0 and residual % FEATURES_PER_CUBE == 0, (
        f'obs_dim {obs_dim} does not match proprio ({NUM_PROPRIO_FEATURES}) '
        f'+ k * {FEATURES_PER_CUBE}'
    )
    return residual // FEATURES_PER_CUBE


def num_buttons_from_obs_dim(obs_dim):
    """Infer the button count from a puzzle state observation dimension."""
    residual = obs_dim - NUM_PROPRIO_FEATURES
    assert residual > 0 and residual % FEATURES_PER_BUTTON == 0, (
        f'obs_dim {obs_dim} does not match proprio ({NUM_PROPRIO_FEATURES}) '
        f'+ k * {FEATURES_PER_BUTTON}'
    )
    return residual // FEATURES_PER_BUTTON


class CubeTokenizer(nn.Module):
    """Tokenize (observation[, goal]) into [B, 1 + D (+ D), H] actor tokens.

    `token_mask` is all-true for the exact-length batches used here; it is
    kept in the API for parity with PuzzleTokenizer.
    """

    d_model: int
    obs_dim: int  # positional table size; asserted against the inputs
    layout: str = 'cube'  # 'cube' | 'puzzle' | 'scene' | 'maze': observation-layout sanity check
    # Paired layout: ONE token per scalar carrying both the state and the
    # goal value, token_i = Dense(2 -> H)([s_i, g_i]) + pos_i, sequence
    # [READOUT, t_0, ..., t_{D-1}] (1 + D tokens instead of 1 + 2D). Puts
    # s_i and g_i in the same token so per-scalar state/goal comparisons
    # (e.g. the button XOR that puzzle needs) are computable by the token
    # MLP without attention having to pair token i with token D + i via
    # the positional table. Requires goals; no modality embeddings.
    paired: bool = False
    # Optional trailing noise tokens (one-step flow policies, FQL): when
    # noise_dim > 0 and `noise` [B, noise_dim] is passed, each noise scalar
    # becomes one token Dense(1 -> H) + noise_pos_j + noise_embedding,
    # appended after the state (and goal) tokens. Parameters are created only
    # when noise_dim > 0, so the goal-conditioned parameter tree is unchanged.
    noise_dim: int = 0

    def _noise_tokens(self, noise):
        H = self.d_model
        assert noise is not None and noise.shape[-1] == self.noise_dim, (
            None if noise is None else noise.shape, self.noise_dim
        )
        noise_projection = nn.Dense(
            H, kernel_init=default_init(1.0), name='noise_projection'
        )
        noise_positional = self.param(
            'noise_positional_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.noise_dim, H),
        )
        noise_embedding = self.param(
            'noise_embedding', nn.initializers.normal(stddev=0.02), (H,)
        )
        tokens = noise_projection(noise[..., None])
        tokens = tokens + noise_positional[None].astype(tokens.dtype)
        return tokens + noise_embedding[None, None].astype(tokens.dtype)

    @nn.compact
    def __call__(self, observations, goals=None, noise=None):
        B = observations.shape[0]
        D = self.obs_dim
        H = self.d_model
        assert observations.shape == (B, D), (observations.shape, D)
        if self.layout == 'cube':
            num_cubes_from_obs_dim(D)  # layout sanity check
        elif self.layout == 'scene':
            # scene-play: 19 proprio + 1 cube (9) + 2 buttons (2 x 4) + drawer
            # (pos, vel) + window (pos, vel) = 40.
            assert D == SCENE_OBS_DIM, (D, SCENE_OBS_DIM)
        elif self.layout == 'maze':
            # Locomotion mazes (pointmaze 2, antmaze 29, humanoidmaze 69,
            # antsoccer 42): flat joint-state vectors, no entity structure to
            # check. Goals are full observations of the same dim.
            pass
        else:
            assert self.layout == 'puzzle', self.layout
            num_buttons_from_obs_dim(D)

        if self.paired:
            assert goals is not None, 'paired tokenizer requires goals'
            assert goals.shape == (B, D), (goals.shape, D)
            pair_projection = nn.Dense(
                H, kernel_init=default_init(1.0), name='pair_projection'
            )
            positional = self.param(
                'positional_embedding',
                nn.initializers.normal(stddev=0.02),
                (D, H),
            )
            readout = self.param(
                'readout', nn.initializers.normal(stddev=0.02), (1, 1, H)
            )
            pairs = jnp.stack([observations, goals], axis=-1)  # [B, D, 2]
            tokens = pair_projection(pairs) + positional[None].astype(
                observations.dtype
            )
            parts = [jnp.broadcast_to(readout.astype(tokens.dtype), (B, 1, H)), tokens]
            if self.noise_dim > 0:
                parts.append(self._noise_tokens(noise))
            tokens = jnp.concatenate(parts, axis=1)
            token_mask = jnp.ones(tokens.shape[:2], bool)
            return tokens, token_mask

        # All parameters are created unconditionally so the parameter tree
        # does not depend on whether goals are passed.
        scalar_projection = nn.Dense(
            H, kernel_init=default_init(1.0), name='scalar_projection'
        )
        positional = self.param(
            'positional_embedding', nn.initializers.normal(stddev=0.02), (D, H)
        )
        state_embedding = self.param(
            'state_embedding', nn.initializers.normal(stddev=0.02), (H,)
        )
        goal_embedding = self.param(
            'goal_embedding', nn.initializers.normal(stddev=0.02), (H,)
        )
        readout = self.param(
            'readout', nn.initializers.normal(stddev=0.02), (1, 1, H)
        )

        def tokenize(x, modality_embedding):
            # [B, D] -> [B, D, H]: shared scalar projection + shared
            # positional table + modality embedding.
            tokens = scalar_projection(x[..., None])
            tokens = tokens + positional[None].astype(tokens.dtype)
            return tokens + modality_embedding[None, None].astype(tokens.dtype)

        parts = [
            jnp.broadcast_to(
                readout.astype(observations.dtype), (B, 1, H)
            ),
            tokenize(observations, state_embedding),
        ]
        if goals is not None:
            assert goals.shape == (B, D), (goals.shape, D)
            parts.append(tokenize(goals, goal_embedding))
        if self.noise_dim > 0:
            parts.append(self._noise_tokens(noise))

        tokens = jnp.concatenate(parts, axis=1)
        token_mask = jnp.ones(tokens.shape[:2], bool)
        return tokens, token_mask
