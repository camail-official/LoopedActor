"""FPRM actor-critic for grid puzzles (Boxoban, Rush Hour) with policy-KL adaptive halting.

The concatenated (grid, goal) observation is tokenized into
[READOUT, CELL_00, ..., CELL_(m-1,n-1)] (row-major, matching the grid
action indexing), and the FPRM core is iterated with the fixed-point
solver. The actor logit for button i is decoded from cell token i's RMS-normed
latent by one shared Dense(1) head (per-cell decoding keeps the parameter tree
board-size independent, unlike a global Dense(m*n) head); the value is decoded
from the readout latent.

Adaptive halting (the same rule during training and evaluation): after every
core call the actor head is decoded from the latent, and an example stops
thinking once the KL divergence between its consecutive policy readouts,
KL(pi_i || pi_{i-1}), drops below `halt_kl` (after `min_think_iters` calls) —
i.e. the agent stops thinking when further thinking no longer changes its
mind. Halted examples are frozen inside the differentiable scan
(`run_halted_scan`), so their gradient path ends at the step where they
halted; everything up to `max_think_iters` core calls is executed for the
batch.

Training backpropagates through every executed iteration (full BPTT); the
final actor/value readout is sown as 'logits'/'value' intermediates for the
PPO loss (ppo.py).

No parameter shape depends on m, n, or the iteration cap: one parameter tree
supports other board sizes and larger test-time caps (separate XLA
compilations, identical parameters).
"""

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from models.fprm import (
    FPRMConfig,
    FPRMCore,
    STOP_CONVERGED,
    init_solver_state,
    make_latent_init,
    rms_norm,
    run_halted_scan,
)

PREFIX_LEN = 1  # [READOUT]


def flat_grid_cell_features(grid, goal, m, n):
    """Per-cell feature tensor [B, m*n, 11] from flat binary grid and goal.

    Features per cell: current one-hot (2), target one-hot (2), mismatch (1),
    normalized row/col in [-1, 1] (2), boundary flags (4).
    """
    B = grid.shape[0]
    cur = grid.reshape(B, m * n, 1)
    tgt = goal.reshape(B, m * n, 1)
    cur_onehot = jnp.concatenate([1.0 - cur, cur], axis=-1)
    tgt_onehot = jnp.concatenate([1.0 - tgt, tgt], axis=-1)
    mismatch = jnp.abs(cur - tgt)

    rows = jnp.arange(m * n) // n
    cols = jnp.arange(m * n) % n
    norm_row = jnp.where(m > 1, 2.0 * rows / jnp.maximum(m - 1, 1) - 1.0, 0.0)
    norm_col = jnp.where(n > 1, 2.0 * cols / jnp.maximum(n - 1, 1) - 1.0, 0.0)
    top = (rows == 0).astype(jnp.float32)
    bottom = (rows == m - 1).astype(jnp.float32)
    left = (cols == 0).astype(jnp.float32)
    right = (cols == n - 1).astype(jnp.float32)
    geometry = jnp.stack([norm_row, norm_col, top, bottom, left, right], axis=-1)
    geometry = jnp.broadcast_to(geometry[None], (B, m * n, 6))

    return jnp.concatenate(
        [cur_onehot, tgt_onehot, mismatch, geometry], axis=-1
    )


def grid_geometry(m, n, B):
    rows = jnp.arange(m * n) // n
    cols = jnp.arange(m * n) % n
    norm_row = jnp.where(m > 1, 2.0 * rows / jnp.maximum(m - 1, 1) - 1.0, 0.0)
    norm_col = jnp.where(n > 1, 2.0 * cols / jnp.maximum(n - 1, 1) - 1.0, 0.0)
    top = (rows == 0).astype(jnp.float32); bottom = (rows == m - 1).astype(jnp.float32)
    left = (cols == 0).astype(jnp.float32); right = (cols == n - 1).astype(jnp.float32)
    geometry = jnp.stack([norm_row, norm_col, top, bottom, left, right], axis=-1)
    return jnp.broadcast_to(geometry[None], (B, m * n, 6))


def grid_channel_features(x_flat, m, n, channels):
    """Per-cell features [B, m*n, channels + 6] from a cell-major flat observation
    of m*n*channels values (e.g. Sokoban wall/box/target/player planes)."""
    B = x_flat.shape[0]
    cells = x_flat.reshape(B, m * n, channels)
    return jnp.concatenate([cells, grid_geometry(m, n, B)], axis=-1)


def tokenize(x_flat, m, n, obs_channels):
    """Cell feature tensor for either input mode (0 = flat grid + goal; > 0 = cell-major planes)."""
    if obs_channels > 0:
        assert x_flat.shape[-1] == m * n * obs_channels, (x_flat.shape, m, n, obs_channels)
        return grid_channel_features(x_flat, m, n, obs_channels)
    grid_size = m * n
    assert x_flat.shape[-1] == 2 * grid_size, (x_flat.shape, m, n)
    return flat_grid_cell_features(x_flat[:, :grid_size], x_flat[:, grid_size:], m, n)


class FPRMThinkerActorValue(nn.Module):
    """Shared FPRM trunk with categorical actor head and value head."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    # Halting criterion: 'kl' stops thinking once KL(pi_i || pi_{i-1}) between
    # consecutive actor readouts drops below halt_kl; 'latent_residual' is the
    # original FPRM-paper rule — stop once the latent max-token residual
    # computed by the solver drops below halt_residual_thresh.
    halt_criterion: str = 'kl'
    # Halting tolerance: stop thinking once KL(pi_i || pi_{i-1}) between
    # consecutive actor readouts drops below this value.
    halt_kl: float = 1e-3
    # Halting tolerance for halt_criterion='latent_residual' (the paper's
    # fp_thresh).
    halt_residual_thresh: float = 0.1
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    # Input mode: 0 = flat grid + goal (2*m*n features); >0 = cell-major planes (m*n*obs_channels), used by Boxoban / Rush Hour.
    obs_channels: int = 0
    # 'per_cell': one logit per cell token (one action per cell); 'readout': Dense(output_dim_1) on the readout token (Sokoban moves).
    action_head: str = 'per_cell'

    def setup(self):
        self.cell_projection = nn.Dense(self.config.d_model)
        self.readout = self.param(
            'readout',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.d_model),
        )
        self.core = FPRMCore(self.config)
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        # per_cell: shared per-cell logit head (no parameter depends on m * n); readout: logits from the readout token.
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
        # per_cell: k = output_dim_1 // (m*n) logits per cell token (k logits per cell), flattened cell-major.
        self.actor_head = nn.Dense(self.output_dim_1 // (self.m * self.n) if self.action_head == 'per_cell' else self.output_dim_1, bias_init=nn.initializers.zeros)
        self.value_head = nn.Dense(
            self.output_dim_2, bias_init=nn.initializers.zeros
        )
        self.latent_init = make_latent_init(self.config)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]

        cell_features = tokenize(x_flat, self.m, self.n, self.obs_channels)
        cell_tokens = self.cell_projection(cell_features)
        readout_token = jnp.broadcast_to(
            self.readout, (B, 1, self.config.d_model)
        )
        tokens = jnp.concatenate([readout_token, cell_tokens], axis=1)
        token_mask = jnp.ones(tokens.shape[:2], bool)
        per_cell = self.action_head == 'per_cell'

        def heads(z):
            readout_latent = rms_norm(z[:, 0], eps=self.config.rms_norm_eps)
            cell_latents = rms_norm(
                z[:, PREFIX_LEN:], eps=self.config.rms_norm_eps
            )
            if per_cell:
                logits = self.actor_head(cell_latents).reshape(cell_latents.shape[0], -1)  # [B, m*n*k]
            else:
                logits = self.actor_head(readout_latent)  # [B, output_dim_1]
            assert logits.shape[-1] == self.output_dim_1, (
                logits.shape, self.output_dim_1
            )
            logits = logits.reshape(input_shape[:-1] + (-1,))
            value = self.value_head(readout_latent)
            value = value.reshape(input_shape[:-1] + (-1,))
            return logits, jnp.squeeze(value, axis=-1)

        state = init_solver_state(self.latent_init, tokens, self.config)

        # Force actor-head param creation before the lifted scan (the KL halt
        # fn closes over the raw kernel/bias arrays: it runs inside the scan,
        # where calling a sibling module would break the transform).
        _ = self.actor_head(jnp.zeros((1, self.config.d_model)))

        assert self.halt_criterion in ('kl', 'latent_residual'), \
            self.halt_criterion
        if self.halt_criterion == 'kl':
            ah = self.actor_head.variables['params']
            ah_kernel, ah_bias = ah['kernel'], ah['bias']

            def halt_residual_fn(z_prev, z_new):
                def log_pi(z):
                    if per_cell:
                        cells = rms_norm(
                            z[:, PREFIX_LEN:], eps=self.config.rms_norm_eps
                        )
                        logits = (cells @ ah_kernel + ah_bias).reshape(cells.shape[0], -1)
                    else:
                        ro = rms_norm(z[:, 0], eps=self.config.rms_norm_eps)
                        logits = ro @ ah_kernel + ah_bias
                    return jax.nn.log_softmax(logits, axis=-1)

                lp_new, lp_prev = log_pi(z_new), log_pi(z_prev)
                return jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), axis=-1)

            fp_thresh = self.halt_kl
        else:
            # Original FPRM halting: run_halted_scan compares the solver's
            # latent max-token residual against fp_thresh.
            halt_residual_fn = None
            fp_thresh = self.halt_residual_thresh

        # One differentiable scan over all iterations (full BPTT through each
        # example's executed iterations); only the final readout is sown.
        state, info = run_halted_scan(
            self.core,
            state,
            tokens,
            token_mask,
            grid_shape=(self.m, self.n),
            config=self.config,
            max_iters=self.max_think_iters,
            min_iters=self.min_think_iters,
            fp_thresh=fp_thresh,
            train=True,
            prefix_len=PREFIX_LEN,
            halt_residual_fn=halt_residual_fn,
        )
        logits, value = heads(state.z)
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)

        # Diagnostics, retrievable via apply(..., mutable=['intermediates']).
        # `state.residual` holds each example's policy KL at its halt step.
        self.sow('intermediates', 'fp_iterations', state.iterations)
        self.sow('intermediates', 'fp_residual', state.residual)
        self.sow(
            'intermediates', 'fp_converged',
            state.stop_reason == STOP_CONVERGED,
        )
        self.sow('intermediates', 'fp_stop_reason', state.stop_reason)

        return distrax.Categorical(logits=logits), value
