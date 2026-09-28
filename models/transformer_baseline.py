"""Feedforward Transformer baselines for the looped FPRM.

Both baselines reuse the FPRM tokenizer, core block, latent initialization,
and decode heads (models/fprm_thinker.py), but replace the fixed-point
iteration by a static feedforward stack: no solver, no halting, and a fixed
amount of computation for every state.

- `num_blocks=1` — "single block": the FPRM core applied exactly once.
  Identical parameter count to the looped FPRM (which reuses this one core
  at every think iteration), but a single unit of computation per decision.
- `num_blocks=8` — "multi block": 8 FPRM core calls with DISTINCT parameters
  (no weight tying). Matches the looped FPRM's default compute per decision
  (8 core calls) at ~8x the parameter count.

Each core call mixes the input tokens back in (the FPRM input injection), so
the multi-block stack is exactly the unrolled FPRM iteration with the weight
tying removed and the fixed-point solver replaced by plain composition.

For compatibility with the shared PPO loss, the final actor/value readout is
sown as 'logits'/'value' and 'fp_iterations' is the constant `num_blocks`.
"""

import distrax
import flax.linen as nn
import jax.numpy as jnp

from models.fprm import FPRMConfig, FPRMCore, make_latent_init, rms_norm
from models.fprm_thinker import PREFIX_LEN, tokenize


class TransformerActorValue(nn.Module):
    """Non-looped stack of FPRM cores with categorical actor and value heads."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    num_blocks: int = 1
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    obs_channels: int = 0          # see FPRMThinkerActorValue
    action_head: str = 'per_cell'  # see FPRMThinkerActorValue

    def setup(self):
        self.cell_projection = nn.Dense(self.config.d_model)
        self.readout = self.param(
            'readout',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.d_model),
        )
        # Distinct parameters per core call (no weight tying across blocks).
        self.cores = [
            FPRMCore(self.config, name=f'core_{i}')
            for i in range(self.num_blocks)
        ]
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
        # per_cell: k = output_dim_1 // (m*n) logits per cell token (k=1 for one logit per cell, k=2 for Rush Hour), flattened cell-major.
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

        def readout(z):
            readout_latent = rms_norm(z[:, 0], eps=self.config.rms_norm_eps)
            cell_latents = rms_norm(
                z[:, PREFIX_LEN:], eps=self.config.rms_norm_eps
            )
            if self.action_head == 'per_cell':
                logits = self.actor_head(cell_latents).reshape(cell_latents.shape[0], -1)  # [B, m*n*k]
            else:
                logits = self.actor_head(readout_latent)
            assert logits.shape[-1] == self.output_dim_1, (
                logits.shape, self.output_dim_1
            )
            logits = logits.reshape(input_shape[:-1] + (-1,))
            value = self.value_head(readout_latent)
            value = jnp.squeeze(value.reshape(input_shape[:-1] + (-1,)), axis=-1)
            return logits, value

        z = jnp.broadcast_to(
            self.latent_init[None, None, :].astype(tokens.dtype), tokens.shape
        )
        for core in self.cores:
            z = core(z, tokens, token_mask, (self.m, self.n), PREFIX_LEN)
        logits, value = readout(z)
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow(
            'intermediates', 'fp_iterations',
            jnp.full((B,), self.num_blocks, jnp.int32),
        )

        return distrax.Categorical(logits=logits), value
