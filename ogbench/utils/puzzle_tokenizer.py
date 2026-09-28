"""Variable-size, parameter-shared puzzle tokenizer (spec Section 7).

Builds the actor token sequence
    [READOUT, ROBOT, CELL_00, CELL_01, ..., CELL_(R-1,C-1)]
from the raw OGBench puzzle state observation and the binary oracle goal.
Cell order matches OGBench row-major button order.

One shared dense projection processes every cell, so no parameter shape
depends on R, C, or R*C; deterministic normalized/physical geometry features
replace learned positions. This makes the tokenizer usable across grid sizes
with one parameter tree, but it is not mathematically size-equivariant:
results must be labeled "geometry-augmented oraclerep".

Observation layout (verified against the pinned OGBench source,
ogbench/manipspace/envs/puzzle_env.py::compute_observation):
    obs[:19]  robot/proprio features:
        [0:6]   joint_pos
        [6:12]  joint_vel
        [12:15] (effector_pos - [0.425, 0, 0]) * 10
        [15]    cos(effector_yaw)
        [16]    sin(effector_yaw)
        [17]    gripper_opening * 3
        [18]    gripper_contact
    obs[19:]  per-button [one-hot(2), joint_pos * 120, joint_vel], row-major.

Button placement (puzzle_env.py::add_objects):
    physical_x = 0.425 - 0.05 * (R - 1) + 0.1 * row
    physical_y =       - 0.05 * (C - 1) + 0.1 * col
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

NUM_ROBOT_FEATURES = 19
FEATURES_PER_BUTTON = 4
PREFIX_LEN = 2  # [READOUT, ROBOT]
NUM_CELL_FEATURES = 17

# OGBench button placement constants.
SPACING_HALF = 0.05
X_CENTER = 0.425
XYZ_SCALER = 10.0


def default_init(scale=1.0):
    """OGBench default kernel initializer (fan-average uniform)."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def cell_geometry(
    grid_shape,
    dtype=jnp.float32,
    coord_transform=None,
    logical_coord_mode='normalized_extent',
):
    """Deterministic per-cell geometry, shape [R, C, ...] each.

    Returns (normalized_row_col [R,C,2], physical_xy [R,C,2],
    boundary_flags [R,C,4] with order top/bottom/left/right).

    `coord_transform` (utils.coord_adapter.CoordinateTransform, eval-only)
    maps the physical coordinates world->virtual; None keeps the exact
    original computation. `logical_coord_mode` selects the lattice-index
    features: 'normalized_extent' (current behavior, [-1, 1]) or
    'metric_lattice' (i - (N-1)/2, constant neighbor spacing).
    """
    R, C = grid_shape
    rows = jnp.arange(R, dtype=dtype)
    cols = jnp.arange(C, dtype=dtype)
    row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')

    if logical_coord_mode == 'normalized_extent':
        # Normalized coordinates in [-1, 1] (a single row/col maps to 0).
        norm_row = jnp.where(R > 1, 2.0 * row_grid / max(R - 1, 1) - 1.0, 0.0)
        norm_col = jnp.where(C > 1, 2.0 * col_grid / max(C - 1, 1) - 1.0, 0.0)
    elif logical_coord_mode == 'metric_lattice':
        norm_row = row_grid - (R - 1) / 2.0
        norm_col = col_grid - (C - 1) / 2.0
    else:
        raise ValueError(f'Unknown logical_coord_mode: {logical_coord_mode!r}')
    normalized_row_col = jnp.stack([norm_row, norm_col], axis=-1)

    physical_x = X_CENTER - SPACING_HALF * (R - 1) + 2 * SPACING_HALF * row_grid
    physical_y = -SPACING_HALF * (C - 1) + 2 * SPACING_HALF * col_grid
    physical_xy = jnp.stack([physical_x, physical_y], axis=-1)
    if coord_transform is not None and not coord_transform.is_identity:
        from utils.coord_adapter import transform_points_xy

        physical_xy = transform_points_xy(physical_xy, coord_transform)

    top = (row_grid == 0).astype(dtype)
    bottom = (row_grid == R - 1).astype(dtype)
    left = (col_grid == 0).astype(dtype)
    right = (col_grid == C - 1).astype(dtype)
    boundary_flags = jnp.stack([top, bottom, left, right], axis=-1)

    return normalized_row_col, physical_xy, boundary_flags


def effector_xyz_from_robot(robot):
    """Recover the physical effector position from the robot features."""
    return robot[..., 12:15] / XYZ_SCALER + jnp.array([X_CENTER, 0.0, 0.0])


def transform_robot_features(robot, coord_transform):
    """World -> virtual for the effector XY channels of the robot features.

    Channels 12/13 store 10 * (effector_xy - center), so the affine map
    p_virtual = c + scale * (p - c) + shift becomes
    ch_virtual = scale * ch + 10 * shift. All other channels pass through.
    """
    sx, sy = coord_transform.scale_xy
    tx, ty = coord_transform.shift_xy
    robot = robot.at[..., 12].set(sx * robot[..., 12] + XYZ_SCALER * tx)
    robot = robot.at[..., 13].set(sy * robot[..., 13] + XYZ_SCALER * ty)
    return robot


def build_cell_features(
    observations,
    goals,
    grid_shape,
    coord_transform=None,
    logical_coord_mode='normalized_extent',
):
    """Parse the raw observation/goal into per-cell features [B, R, C, 17].

    Feature order:
        current_onehot (2), target_onehot (2), mismatch (1),
        button_joint_pos (1), button_joint_vel (1), normalized_row_col (2),
        physical_xy (2), relative_xy (2), boundary_flags (4).

    With a non-identity `coord_transform` (evaluation-only), the physical
    cell coordinates, the effector XY channels of the returned robot
    features, and hence the relative coordinates are all expressed in the
    virtual (transformed) frame; every other feature is unchanged.
    """
    R, C = grid_shape
    B = observations.shape[0]
    assert observations.shape[-1] == NUM_ROBOT_FEATURES + FEATURES_PER_BUTTON * R * C, (
        observations.shape,
        grid_shape,
    )
    assert goals.shape[-1] == R * C, (goals.shape, grid_shape)

    robot = observations[:, :NUM_ROBOT_FEATURES]
    if coord_transform is not None and not coord_transform.is_identity:
        robot = transform_robot_features(robot, coord_transform)
    cells = observations[:, NUM_ROBOT_FEATURES:].reshape(B, R, C, FEATURES_PER_BUTTON)

    current_onehot = cells[..., :2]
    button_joint_pos = cells[..., 2:3]
    button_joint_vel = cells[..., 3:4]

    target_idx = goals.reshape(B, R, C).astype(jnp.int32)
    target_onehot = jax.nn.one_hot(target_idx, 2, dtype=observations.dtype)
    target_bit = target_idx[..., None].astype(observations.dtype)

    current_bit = current_onehot[..., 1:2]
    mismatch = jnp.abs(current_bit - target_bit)

    normalized_row_col, physical_xy, boundary_flags = cell_geometry(
        grid_shape,
        observations.dtype,
        coord_transform=coord_transform,
        logical_coord_mode=logical_coord_mode,
    )
    normalized_row_col = jnp.broadcast_to(normalized_row_col[None], (B, R, C, 2))
    physical_xy_b = jnp.broadcast_to(physical_xy[None], (B, R, C, 2))
    boundary_flags = jnp.broadcast_to(boundary_flags[None], (B, R, C, 4))

    effector_xy = effector_xyz_from_robot(robot)[..., :2]  # [B, 2]
    relative_xy = physical_xy_b - effector_xy[:, None, None, :]

    cell_features = jnp.concatenate(
        [
            current_onehot,       # 2
            target_onehot,        # 2
            mismatch,             # 1
            button_joint_pos,     # 1
            button_joint_vel,     # 1
            normalized_row_col,   # 2
            physical_xy_b,        # 2
            relative_xy,          # 2
            boundary_flags,       # 4
        ],
        axis=-1,
    )
    assert cell_features.shape == (B, R, C, NUM_CELL_FEATURES)
    return robot, cell_features


class PuzzleTokenizer(nn.Module):
    """Tokenize (observation, oracle goal) into [B, 2 + R*C, H] actor tokens.

    No parameter shape depends on the grid size. `token_mask` is all-true for
    the exact-length batches used here; it is kept in the API for future
    padded/mixed-size batching.
    """

    d_model: int

    @nn.compact
    def __call__(
        self,
        observations,
        goals,
        grid_shape,
        coord_transform=None,
        logical_coord_mode='normalized_extent',
    ):
        R, C = grid_shape
        B = observations.shape[0]
        H = self.d_model

        robot, cell_features = build_cell_features(
            observations,
            goals,
            grid_shape,
            coord_transform=coord_transform,
            logical_coord_mode=logical_coord_mode,
        )

        cell_tokens = nn.Dense(
            H, kernel_init=default_init(1.0), name='cell_projection'
        )(cell_features)
        cell_tokens = cell_tokens.reshape(B, R * C, H)

        robot_token = nn.Dense(
            H, kernel_init=default_init(1.0), name='robot_projection'
        )(robot)
        robot_token = robot_token[:, None, :]

        readout = self.param(
            'readout', nn.initializers.normal(stddev=0.02), (1, 1, H)
        )
        readout = jnp.broadcast_to(readout.astype(cell_tokens.dtype), (B, 1, H))

        tokens = jnp.concatenate([readout, robot_token, cell_tokens], axis=1)
        token_mask = jnp.ones((B, PREFIX_LEN + R * C), bool)
        return tokens, token_mask
