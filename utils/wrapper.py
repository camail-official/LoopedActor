from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import struct

from envs.utils import State


class Wrapper():
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, rng) -> State:
        return self.env.reset(rng)

    def eval_reset(self, rng) -> State:
        return self.env.eval_reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def goal_size(self) -> int:
        return self.env.goal_size

    def __getattr__(self, name):
        return getattr(self.env, name)


class VmapWrapper(Wrapper):
    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def eval_reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.eval_reset)(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
    def __init__(self, env, episode_length: int):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        new_info = {
            **state.info,
            'steps': jnp.zeros((), dtype=jnp.int32),
            'truncation': jnp.zeros((), dtype=jnp.float32),
        }
        return state.replace(info=new_info)

    def eval_reset(self, rng: jax.Array) -> State:
        state = self.env.eval_reset(rng)
        new_info = {
            **state.info,
            'steps': jnp.zeros((), dtype=jnp.int32),
            'truncation': jnp.zeros((), dtype=jnp.float32),
        }
        return state.replace(info=new_info)

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        steps = state.info['steps'] + 1
        timed_out = (steps >= self.episode_length)
        done = jnp.maximum(state.done, timed_out.astype(jnp.float32))
        truncation = (timed_out & (state.done < 0.5)).astype(jnp.float32)

        new_info = {**state.info, 'steps': steps, 'truncation': truncation}
        return state.replace(done=done, info=new_info)


class AutoResetWrapper(Wrapper):
    def step(self, state: State, action: jax.Array) -> State:
        # 1. Update the RNG in the info dict before the step
        # This ensures the step itself and any subsequent reset have fresh keys
        rng, step_rng = jax.random.split(state.info["rng"])
        state = state.replace(info={**state.info, "rng": rng})

        # 2. Take the step
        next_state = self.env.step(state, action)

        # 3. If the result of that step is 'done', trigger reset
        # Otherwise, keep the next_state
        return jax.lax.cond(
            next_state.done.astype(bool),
            lambda _: self.env.reset(step_rng).replace(
                # We preserve the terminal reward and done signal
                # so the Learner/Logger sees the final outcome
                reward=next_state.reward,
                done=next_state.done,
                metrics=next_state.metrics,
            ),
            lambda _: next_state,
            operand=None,
        )


@struct.dataclass
class EvalMetrics:
    episode_metrics: Dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array


class EvalWrapper(Wrapper):
    def reset(self, rng: jax.Array):
        reset_state = self.env.reset(rng)
        return self._init_eval_metrics(reset_state)

    def eval_reset(self, rng: jax.Array):
        reset_state = self.env.eval_reset(rng)
        return self._init_eval_metrics(reset_state)

    def _init_eval_metrics(self, reset_state):
        new_metrics = {**reset_state.metrics, 'reward': reset_state.reward}
        reset_state = reset_state.replace(metrics=new_metrics)

        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )

        new_info = {**reset_state.info, 'eval_metrics': eval_metrics}
        return reset_state.replace(info=new_info)

    def step(self, state, action):
        state_metrics = state.info['eval_metrics']

        state = self.env.step(state, action)

        new_metrics = {**state.metrics, 'reward': state.reward}
        state = state.replace(metrics=new_metrics)

        episode_steps = jnp.where(
            state_metrics.active_episodes.astype(bool),
            state.info['steps'],
            state_metrics.episode_steps,
        )

        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            state.metrics,
        )

        active_episodes = state_metrics.active_episodes * (1.0 - state.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )

        new_info = {**state.info, 'eval_metrics': eval_metrics}
        return state.replace(info=new_info)


def wrap_env(env, episode_length: int, train=True) -> Wrapper:
    env = EpisodeWrapper(env, episode_length)
    if train:
        env = AutoResetWrapper(env)
    env = VmapWrapper(env)
    return env
