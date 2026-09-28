import re
from typing import Any, Dict

import jax
from flax import struct


@struct.dataclass
class State:
    """Environment state for training and inference."""
    data: Any
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


def make_env(args):
    """Build the environment class and config from an env id (sokoban-<train>-<eval> | rushhour-<train>-<eval>)."""
    sok = re.fullmatch(r"sokoban-([a-z_]+)-([a-z_]+)", args.env_id)  # sokoban-<train_split>-<eval_split>
    if sok is not None:
        from envs.sokoban_env import SokobanEnv, default_config as sokoban_config
        config = sokoban_config()
        config.train_split, config.eval_split = sok.group(1), sok.group(2)
        config.max_train_levels = getattr(args, 'sokoban_max_train_levels', 0)
        config.episode_length = getattr(args, 'sokoban_episode_length', 120)
        return SokobanEnv, config
    rh = re.fullmatch(r"rushhour-([a-z_]+)-([a-z_]+)", args.env_id)  # rushhour-<train_split>-<eval_split>
    if rh is not None:
        from envs.rushhour_env import RushHourEnv, default_config as rushhour_config
        config = rushhour_config()
        config.train_split, config.eval_split = rh.group(1), rh.group(2)
        config.max_train_levels = getattr(args, 'rushhour_max_train_levels', 0)
        config.episode_length = getattr(args, 'rushhour_episode_length', 150)
        config.shaping_weight = getattr(args, 'rushhour_shaping_weight', 1.0)
        return RushHourEnv, config
    raise ValueError(f"Environment {args.env_id} not supported (expected sokoban-<train>-<eval> or rushhour-<train>-<eval>)")
