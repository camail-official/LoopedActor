"""Flax/JAX implementation of the FPRM core and fixed-point solver.

Ported from the public FPRM implementation (pinned):
    https://github.com/nilskiKonjIzDunava/fprm @ 4fd7ab116b1361c4fb040b26b270f19a3d5319b4
    - models/layers.py       (rms_norm, Attention, SwiGLU, CastedLinear init)
    - models/transformer.py  (FixedPointTransformerBlock, FixedPointTransformer)
    - models/fixed_point_reasoning/model_utils.py (FixedPointOptimizer)

Deliberate deviations from the public code:
    - The grid convolution takes an explicit rectangular (R, C) instead of
      inferring a square grid from sqrt(sequence_length).
    - Every iteration replaces the latent by the core's proposal (no mixing
      with the previous latent); `fp_thresh` is used only for halting.
    - Attention uses an explicit token mask (all-true for the exact-length
      batches used here; kept for future padded batching).
    - No vocabulary/puzzle embeddings, no q_head/ACT, no variational or
      weight dropout, no Jacobian regularization (all out of scope).

`run_halted_scan` is a reverse-differentiable fixed-length scan with
per-example halting, usable inside an RL training loss. The halting residual
is pluggable via `halt_residual_fn` — the KL-halting example replaces the
latent residual with the KL divergence between consecutive policy readouts.
"""

import dataclasses
import math
from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

# Stop-reason integer enum (host-side code converts to strings).
STOP_ACTIVE = 0
STOP_CONVERGED = 1
STOP_ITERATION_CAP = 2
STOP_NONFINITE = 3

STOP_REASON_NAMES = {
    STOP_ACTIVE: 'active',
    STOP_CONVERGED: 'converged',
    STOP_ITERATION_CAP: 'iteration_cap',
    STOP_NONFINITE: 'nonfinite',
}


@dataclasses.dataclass(frozen=True)
class FPRMConfig:
    """Static configuration for the FPRM core and solver."""

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 1
    expansion: float = 4.0
    softmax_temp: float = 1.0
    rms_norm_eps: float = 1e-5

    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25
    # Coupled residual scaling (a1/b1 block residuals + a2/b2 input mixing).
    # False -> plain pre-norm residuals (h = h + f(h); h = z + input_tokens):
    # no contractivity, intended for the non-looped baselines only.
    residual_scaling: bool = True

    use_grid_conv: bool = True
    conv_kernel: Tuple[int, int] = (3, 3)
    conv_bias: bool = False

    fp_thresh: float = 0.1
    eps: float = 1e-8
    adaptive_min_iters: int = 2

    init_mode: str = 'fixed_random'  # 'fixed_random' | 'zeros'
    init_std: float = 6.0
    init_seed: int = 0


def rms_norm(x, eps):
    """Unparameterized RMSNorm, matching the public implementation."""
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    y = x32 * jax.lax.rsqrt(variance + eps)
    return y.astype(dtype)


def round_up_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def trunc_normal_init(stddev):
    """Truncated-normal initializer matching the public `trunc_normal_init_`.

    The public PyTorch helper replicates JAX semantics (std-corrected, truncated
    at +-2 sigma), so Flax's `truncated_normal` is the exact counterpart.
    """
    return nn.initializers.truncated_normal(stddev=stddev)


def pytorch_depthwise_conv_init(key, shape, dtype=jnp.float32):
    """PyTorch nn.Conv2d default (Kaiming-uniform, a=sqrt(5)) for a bias-free
    depthwise kernel: bound = 1/sqrt(fan_in) with fan_in = kH*kW*(in/groups).

    Flax depthwise kernel shape: [kH, kW, in_channels/groups(=1), features].
    """
    fan_in = shape[0] * shape[1] * shape[2]
    bound = 1.0 / math.sqrt(fan_in)
    return jax.random.uniform(key, shape, dtype, -bound, bound)


class FPRMAttention(nn.Module):
    """Noncausal multi-head self-attention with bias-free projections.

    Matches the public Attention module (no RoPE, no QK norm) with
    logit scale 1 / (sqrt(head_dim) * softmax_temp), plus an explicit
    token mask (invalid keys -> -inf; invalid query outputs zeroed).
    """

    config: FPRMConfig

    @nn.compact
    def __call__(self, x, token_mask):
        cfg = self.config
        B, L, H = x.shape
        num_heads = cfg.num_heads
        head_dim = H // num_heads
        assert head_dim * num_heads == H

        qkv = nn.Dense(
            3 * H,
            use_bias=False,
            kernel_init=trunc_normal_init(1.0 / math.sqrt(H)),
            name='qkv_proj',
        )(x)
        qkv = qkv.reshape(B, L, 3 * num_heads, head_dim)
        query = qkv[:, :, :num_heads]
        key = qkv[:, :, num_heads : 2 * num_heads]
        value = qkv[:, :, 2 * num_heads :]

        scale = 1.0 / (math.sqrt(head_dim) * cfg.softmax_temp)
        logits = jnp.einsum('bqhd,bkhd->bhqk', query, key) * scale
        logits = jnp.where(token_mask[:, None, None, :], logits, -jnp.inf)
        probs = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', probs, value)
        out = out.reshape(B, L, H)

        out = nn.Dense(
            H,
            use_bias=False,
            kernel_init=trunc_normal_init(1.0 / math.sqrt(H)),
            name='o_proj',
        )(out)
        # Zero invalid queries (future padded batching; all-true masks here).
        out = jnp.where(token_mask[..., None], out, 0.0)
        return out


class FPRMSwiGLU(nn.Module):
    """SwiGLU MLP matching the public implementation."""

    config: FPRMConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        H = cfg.d_model
        inter = round_up_to_multiple(round(cfg.expansion * H * 2 / 3), 256)

        gate_up = nn.Dense(
            2 * inter,
            use_bias=False,
            kernel_init=trunc_normal_init(1.0 / math.sqrt(H)),
            name='gate_up_proj',
        )(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        y = jax.nn.silu(gate) * up
        y = nn.Dense(
            H,
            use_bias=False,
            kernel_init=trunc_normal_init(1.0 / math.sqrt(inter)),
            name='down_proj',
        )(y)
        return y


class FPRMBlock(nn.Module):
    """Pre-norm Transformer block with coupled residual scaling.

    h = a1 * h + b1 * attn(rms_norm(h))
    h = a1 * h + b1 * swiglu(rms_norm(h))
    """

    config: FPRMConfig

    @nn.compact
    def __call__(self, h, token_mask, alpha_1, beta_1):
        cfg = self.config
        out = FPRMAttention(cfg, name='self_attn')(
            rms_norm(h, cfg.rms_norm_eps), token_mask
        )
        h = alpha_1 * h + beta_1 * out
        out = FPRMSwiGLU(cfg, name='mlp')(rms_norm(h, cfg.rms_norm_eps))
        h = alpha_1 * h + beta_1 * out
        return h


def grid_conv_tokens(conv: nn.Module, z, grid_shape, prefix_len):
    """Apply a 2-D convolution to the grid tokens of a token sequence.

    Splits the prefix tokens, reshapes the cell tokens to the explicit
    rectangular [B, R, C, H] grid, applies the convolution, and reassembles.
    Never infers the grid from sqrt(sequence_length): the public code does and
    therefore fails for rectangular boards. A 1-D convolution over flattened
    cells would create false row-wrap adjacency; this preserves 2-D topology.
    """
    B, L, H = z.shape
    R, C = grid_shape
    assert L == prefix_len + R * C, (L, prefix_len, R, C)
    prefix = z[:, :prefix_len]
    grid = z[:, prefix_len:].reshape(B, R, C, H)
    grid = conv(grid)
    return jnp.concatenate([prefix, grid.reshape(B, R * C, H)], axis=1)


class FPRMCore(nn.Module):
    """One recurrent FPRM core call: grid conv + input mixing + L blocks.

    Maps (hidden_states [B, L, H], input_tokens [B, L, H]) -> [B, L, H].
    The whole module is reused at every recurrent iteration (weight tying
    across iterations); the `num_layers` blocks within one call have
    distinct parameters.
    """

    config: FPRMConfig

    @nn.compact
    def __call__(self, z, input_tokens, token_mask, grid_shape, prefix_len=1):
        cfg = self.config
        B, L, H = z.shape
        R, C = grid_shape
        assert L == prefix_len + R * C, (L, prefix_len, R, C)

        # Rectangular depthwise grid convolution on the cell tokens.
        if cfg.use_grid_conv:
            conv = nn.Conv(
                features=H,
                kernel_size=cfg.conv_kernel,
                padding='SAME',
                feature_group_count=H,
                use_bias=cfg.conv_bias,
                kernel_init=pytorch_depthwise_conv_init,
                name='grid_depthwise_conv',
            )
            z = grid_conv_tokens(conv, z, grid_shape, prefix_len)

        if cfg.residual_scaling:
            # Input-independent residual scaling (public 'input-independent').
            alpha_1_logit = self.param(
                'alpha_1_logit',
                nn.initializers.constant(_logit(cfg.alpha_1_init)),
                (H,),
            )
            alpha_2_logit = self.param(
                'alpha_2_logit',
                nn.initializers.constant(_logit(cfg.alpha_2_init)),
                (H,),
            )
            a1 = jax.nn.sigmoid(alpha_1_logit)[None, None, :].astype(z.dtype)
            a2 = jax.nn.sigmoid(alpha_2_logit)[None, None, :].astype(z.dtype)
            two_l = 2 * cfg.num_layers
            b2 = 1.0 - a2 * a1**two_l
            b1 = b2 * (1.0 - a1) / (1.0 - a1**two_l + 1e-5)
        else:
            a1 = a2 = b1 = b2 = 1.0

        # Input mixing (repeated input injection).
        h = a2 * z + b2 * input_tokens

        for i in range(cfg.num_layers):
            h = FPRMBlock(cfg, name=f'block_{i}')(h, token_mask, a1, b1)

        return h


def _logit(p):
    return math.log(p / (1.0 - p))


@flax.struct.dataclass
class FPSolverState:
    z: jax.Array  # [B, L, H]
    residual: jax.Array  # [B], float32
    iterations: jax.Array  # [B], int32
    active: jax.Array  # [B], bool
    nonfinite: jax.Array  # [B], bool
    stop_reason: jax.Array  # [B], int32


@flax.struct.dataclass
class FPInfo:
    iterations: jax.Array
    residual: jax.Array
    rms_residual: jax.Array
    paper_global_residual: jax.Array
    converged: jax.Array
    hit_cap: jax.Array
    stop_reason: jax.Array
    nonfinite: jax.Array


def make_latent_init(config: FPRMConfig):
    """Deterministic latent initialization vector.

    The vector is created once from `init_seed` and is shape-independent of
    batch/sequence; it should be stored in the checkpoint as a non-trainable
    constant.
    """
    if config.init_mode == 'zeros':
        return jnp.zeros((config.d_model,), jnp.float32)
    elif config.init_mode == 'fixed_random':
        key = jax.random.PRNGKey(config.init_seed)
        # Matches jax.nn.initializers.truncated_normal semantics (std-corrected).
        return trunc_normal_init(config.init_std)(
            key, (config.d_model,), jnp.float32
        )
    else:
        raise ValueError(f'Unknown init_mode: {config.init_mode}')


def init_solver_state(init_vector, input_tokens, config: FPRMConfig):
    """Fresh solver state for a batch (reset per minibatch / per decision)."""
    B, L, H = input_tokens.shape
    z0 = jnp.broadcast_to(
        init_vector[None, None, :].astype(input_tokens.dtype), (B, L, H)
    )
    return FPSolverState(
        z=z0,
        residual=jnp.full((B,), jnp.inf, jnp.float32),
        iterations=jnp.zeros((B,), jnp.int32),
        active=jnp.ones((B,), bool),
        nonfinite=jnp.zeros((B,), bool),
        stop_reason=jnp.full((B,), STOP_ACTIVE, jnp.int32),
    )


def compute_residuals(z, proposal, token_mask, eps):
    """All three residual definitions, without gradients.

    Returns (public_code_max_token_residual, rms_residual,
    paper_global_residual), each [B] float32.
    """
    z_sg = jax.lax.stop_gradient(z).astype(jnp.float32)
    p_sg = jax.lax.stop_gradient(proposal).astype(jnp.float32)

    numerator = jnp.max(jnp.abs(z_sg - p_sg), axis=-1)  # [B, L]
    denominator = jnp.max(jnp.abs(p_sg), axis=-1) + eps  # [B, L]
    token_residual_finite = numerator / denominator

    token_residual_for_max = jnp.where(token_mask, token_residual_finite, -jnp.inf)
    max_residual = jnp.max(token_residual_for_max, axis=-1)

    valid = token_mask.astype(jnp.float32)
    # Zero masked tokens before squaring: a masked token with a near-zero
    # proposal can have a non-finite ratio, and inf * 0 = nan.
    token_residual_masked = jnp.where(token_mask, token_residual_finite, 0.0)
    rms_residual = jnp.sqrt(
        jnp.sum(jnp.square(token_residual_masked), axis=-1)
        / jnp.maximum(jnp.sum(valid, axis=-1), 1.0)
    )

    masked_diff = jnp.where(token_mask[..., None], jnp.abs(z_sg - p_sg), 0.0)
    masked_prop = jnp.where(token_mask[..., None], jnp.abs(p_sg), 0.0)
    paper_global_residual = jnp.max(masked_diff, axis=(-2, -1)) / (
        jnp.max(masked_prop, axis=(-2, -1)) + eps
    )

    return max_residual, rms_residual, paper_global_residual


def solver_step(state: FPSolverState, proposal, token_mask, config: FPRMConfig):
    """One fixed-point solver step: the proposal becomes the new latent.

    Residual is computed between the previous latent and the proposal. Solver
    metadata is non-differentiable; gradients flow through z -> proposal.

    Returns (new_state, (rms_residual, paper_global_residual)).
    """
    residual, rms_residual, paper_global = compute_residuals(
        state.z, proposal, token_mask, config.eps
    )

    new_z = proposal

    # Nonfinite detection: proposal and residual.
    finite = jnp.all(jnp.isfinite(proposal), axis=(-2, -1)) & jnp.isfinite(residual)
    nonfinite = state.nonfinite | ~finite

    return state.replace(
        z=new_z,
        residual=residual,
        iterations=state.iterations + 1,
        nonfinite=nonfinite,
    ), (rms_residual, paper_global)


def _frozen_where(active, new_state: FPSolverState, old_state: FPSolverState):
    """Select new state only for active examples (per-example freezing)."""

    def sel(new, old):
        mask = active.reshape((-1,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(sel, new_state, old_state)


def _classify_stop(state: FPSolverState, config: FPRMConfig, min_iters,
                   max_iters):
    """Stop condition + reason, with the priority
    nonfinite > converged > iteration_cap."""
    converged = (state.iterations >= min_iters) & (
        state.residual < config.fp_thresh
    )
    capped = state.iterations >= max_iters

    reason = jnp.full_like(state.stop_reason, STOP_ACTIVE)
    reason = jnp.where(capped, STOP_ITERATION_CAP, reason)
    reason = jnp.where(converged, STOP_CONVERGED, reason)
    reason = jnp.where(state.nonfinite, STOP_NONFINITE, reason)

    should_stop = state.nonfinite | converged | capped
    return should_stop, reason


def run_halted_scan(core: nn.Module, state: FPSolverState, input_tokens,
                    token_mask, grid_shape, config: FPRMConfig,
                    max_iters, min_iters=None, fp_thresh=None, train=True,
                    prefix_len=1, remat=True, num_steps=None,
                    halt_residual_fn=None):
    """Fixed-length scan with per-example halting, differentiable.

    Runs exactly `num_steps` core calls, but an example whose halting residual
    falls below `fp_thresh` (after `min_iters` steps) or which goes nonfinite
    is frozen: its latent and all solver
    metadata keep their halted values through `jnp.where`, so its gradient
    path ends at the step where it halted. Implemented with `nn.scan` and
    therefore supports reverse-mode differentiation, at the cost of always
    executing `num_steps` core calls for the whole batch.

    Returns (final_state, FPInfo). The FPInfo rms/paper residuals are those of
    each example's final executed step.

    `num_steps` (default `max_iters`) is the scan length of THIS call, while
    `max_iters` stays the global iteration cap used by `_classify_stop`
    against the cumulative `state.iterations`. Passing a pre-advanced `state`
    with `num_steps < max_iters` therefore executes one segment of a longer
    halted run with identical halting semantics.

    `halt_residual_fn` (optional) replaces the stopping residual: a callable
    (z_prev, z_new) -> [B] whose (stop_gradient'ed) value is written into
    `state.residual` after each solver step, so `_classify_stop` compares IT
    against `fp_thresh` instead of the latent max-token residual. Used for
    the policy-KL halting criterion in fprm_thinker.

    Must be called from inside a Linen module (uses nn.scan with
    variable_broadcast='params' so a single parameter tree is reused).
    """
    if min_iters is None:
        min_iters = config.adaptive_min_iters
    if num_steps is None:
        num_steps = max_iters
    if fp_thresh is not None:
        config = dataclasses.replace(config, fp_thresh=fp_thresh)

    B = state.z.shape[0]
    init_carry = (
        state,
        jnp.full((B,), jnp.inf, jnp.float32),  # rms residual at final step
        jnp.full((B,), jnp.inf, jnp.float32),  # paper-global residual
    )

    def body(core, carry, _):
        st, rms_prev, paper_prev = carry
        proposal = core(st.z, input_tokens, token_mask, grid_shape, prefix_len)
        stepped, (rms_res, paper_res) = solver_step(
            st, proposal, token_mask, config
        )
        # A nonfinite step keeps the last finite latent.
        stepped = stepped.replace(
            z=jnp.where(stepped.nonfinite[:, None, None], st.z, stepped.z)
        )
        if halt_residual_fn is not None:
            stepped = stepped.replace(
                residual=jax.lax.stop_gradient(
                    halt_residual_fn(st.z, stepped.z)
                )
            )
        should_stop, reason = _classify_stop(stepped, config, min_iters, max_iters)
        stepped = stepped.replace(
            active=~should_stop,
            stop_reason=jnp.where(should_stop, reason, STOP_ACTIVE),
        )
        # Freeze examples that had already halted before this step.
        new_st = _frozen_where(st.active, stepped, st)
        active = st.active
        new_rms = jnp.where(active, rms_res, rms_prev)
        new_paper = jnp.where(active, paper_res, paper_prev)
        return (new_st, new_rms, new_paper), None

    if core.is_initializing():
        # nn.scan cannot create parameters at init; run the body once instead.
        (final_state, rms_residual, paper_residual), _ = body(
            core, init_carry, jnp.zeros((), jnp.int32)
        )
    else:
        # Rematerialize each iteration in the backward pass: activation memory
        # stays at one core call regardless of the number of iterations.
        scan_body = nn.remat(body, prevent_cse=False) if remat else body
        scan = nn.scan(
            scan_body,
            variable_broadcast='params',
            split_rngs={'params': False},
            length=num_steps,
        )
        (final_state, rms_residual, paper_residual), _ = scan(
            core, init_carry, jnp.arange(num_steps)
        )

    info = FPInfo(
        iterations=final_state.iterations,
        residual=final_state.residual,
        rms_residual=rms_residual,
        paper_global_residual=paper_residual,
        converged=final_state.stop_reason == STOP_CONVERGED,
        hit_cap=final_state.stop_reason == STOP_ITERATION_CAP,
        stop_reason=final_state.stop_reason,
        nonfinite=final_state.nonfinite,
    )
    return final_state, info
