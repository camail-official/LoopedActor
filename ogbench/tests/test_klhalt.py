"""Tests for the policy-KL halted actor objective.

Covers: default-off behavior, checkpoint-compatible parameter tree, halting
mechanism in both directions, eval-path parity with fixed depth, and one
finite update.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest



def make_example_batch(batch_size=2):
    rng = np.random.default_rng(0)
    return {
        'observations': rng.normal(size=(batch_size, 55)).astype(np.float32),
        'next_observations': rng.normal(size=(batch_size, 55)).astype(np.float32),
        'actions': rng.uniform(-1, 1, size=(batch_size, 5)).astype(np.float32),
        'actor_goals': rng.integers(0, 2, size=(batch_size, 9)).astype(np.float32),
        'value_goals': rng.integers(0, 2, size=(batch_size, 9)).astype(np.float32),
        'rewards': -np.ones(batch_size, np.float32),
        'masks': np.ones(batch_size, np.float32),
    }


def make_batch(batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.normal(size=(batch_size, 55)).astype(np.float32),
        'next_observations': rng.normal(size=(batch_size, 55)).astype(np.float32),
        'actions': rng.uniform(-1, 1, size=(batch_size, 5)).astype(np.float32),
        'actor_goals': rng.integers(0, 2, size=(batch_size, 9)).astype(np.float32),
        'value_goals': rng.integers(0, 2, size=(batch_size, 9)).astype(np.float32),
        'rewards': -rng.integers(0, 2, size=batch_size).astype(np.float32),
        'masks': np.ones(batch_size, np.float32),
    }


def make_klhalt_agent(seed=0, kl=1e-3):
    from agents.gciql_fprm import GCIQLFPRMAgent, get_config

    config = get_config()
    config['fprm_halt_on_policy_kl'] = kl
    return GCIQLFPRMAgent.create(seed, make_example_batch(), config)


def make_default_agent(seed=0):
    from agents.gciql_fprm import GCIQLFPRMAgent, get_config

    return GCIQLFPRMAgent.create(seed, make_example_batch(), get_config())


@pytest.fixture(scope='module')
def klhalt_agent():
    return make_klhalt_agent()


@pytest.fixture(scope='module')
def default_agent():
    return make_default_agent()


# --- Default-off behavior ----------------------------------------------------


def test_default_agent_trains_at_fixed_depth(default_agent):
    batch = make_batch()
    dist, info = default_agent.network.select('actor')(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        train=True,
        return_train_info=True,
    )
    assert dist.mode().shape == (8, 5)
    assert 'think_iters' not in info


# --- Checkpoint compatibility ------------------------------------------------


def test_param_tree_identical_to_default(klhalt_agent, default_agent):
    """klhalt adds no parameters: fixed-depth checkpoints stay loadable."""
    a = jax.tree_util.tree_map(lambda x: x.shape, klhalt_agent.network.params)
    b = jax.tree_util.tree_map(lambda x: x.shape, default_agent.network.params)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert a == b


# --- Training path -----------------------------------------------------------


def test_klhalt_train_path_reports_halting(klhalt_agent):
    batch = make_batch()
    dist, info = klhalt_agent.network.select('actor')(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        train=True,
        return_train_info=True,
    )
    assert dist.mode().shape == (8, 5)
    assert float(info['think_iters']) <= 8.0
    assert 0.0 <= float(info['kl_converged_frac']) <= 1.0


def test_train_halting_freezes_examples():
    """A huge threshold halts everything at min_iters; the training readout
    is then the readout after exactly min_iters core calls."""
    agent = make_klhalt_agent(kl=1e6)
    batch = make_batch()
    dist, info = agent.network.select('actor')(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        train=True,
        return_train_info=True,
    )
    assert float(info['think_iters']) == 2.0
    assert float(info['kl_converged_frac']) == 1.0
    a_fixed, _ = agent.sample_actions_with_info(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        halt_mode='fixed',
        num_iters=2,
    )
    np.testing.assert_allclose(
        np.asarray(dist.mode()), np.asarray(a_fixed), rtol=1e-5, atol=1e-6
    )


def test_tiny_threshold_runs_to_cap():
    agent = make_klhalt_agent(kl=1e-30)
    batch = make_batch()
    _, info = agent.network.select('actor')(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        train=True,
        return_train_info=True,
    )
    assert float(info['think_iters']) == 8.0
    assert float(info['kl_converged_frac']) == 0.0


def test_one_jit_update_finite(klhalt_agent):
    batch = make_batch()
    new_agent, info = klhalt_agent.update(batch)
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), (k, v)
    assert 'actor/think_iters' in info and 'actor/mse' in info
    before = klhalt_agent.network.params['modules_actor']['mean_head']['kernel']
    after = new_agent.network.params['modules_actor']['mean_head']['kernel']
    assert not np.allclose(np.asarray(before), np.asarray(after))


def test_latent_init_zero_grad(klhalt_agent):
    batch = make_batch()

    def loss(grad_params):
        l, _ = klhalt_agent.actor_loss(batch, grad_params)
        return l

    grads = jax.grad(loss)(klhalt_agent.network.params)
    latent_grad = grads['modules_actor']['latent_init']
    assert float(jnp.abs(latent_grad).sum()) == 0.0


# --- Evaluation path ---------------------------------------------------------


def test_eval_policy_kl_matches_fixed_when_never_halting(klhalt_agent):
    batch = make_batch(batch_size=4)
    a_fixed, _ = klhalt_agent.sample_actions_with_info(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        halt_mode='fixed',
        num_iters=8,
    )
    a_kl, info = klhalt_agent.sample_actions_with_info(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        halt_mode='policy_kl',
        max_iters=8,
        fp_thresh=1e-30,
    )
    np.testing.assert_allclose(
        np.asarray(a_fixed), np.asarray(a_kl), rtol=1e-5, atol=1e-6
    )
    assert np.all(np.asarray(info['iterations']) == 8)


def test_eval_policy_kl_halts_and_reports(klhalt_agent):
    batch = make_batch(batch_size=4)
    _, info = klhalt_agent.sample_actions_with_info(
        batch['observations'],
        batch['actor_goals'],
        grid_shape=(3, 3),
        halt_mode='policy_kl',
        max_iters=8,
        fp_thresh=1e6,
    )
    assert np.all(np.asarray(info['iterations']) == 2)
    assert np.all(np.asarray(info['converged']))
    # action_change_last_step_l2 = sqrt(2 * KL residual).
    np.testing.assert_allclose(
        np.asarray(info['action_change_last_step_l2']),
        np.sqrt(2.0 * np.asarray(info['residual'])),
        rtol=1e-6,
    )


def test_sample_actions_uses_klhalt(klhalt_agent):
    obs = np.random.default_rng(0).normal(size=(55,)).astype(np.float32)
    goal = (
        np.random.default_rng(1).integers(0, 2, size=(9,)).astype(np.float32)
    )
    action = klhalt_agent.sample_actions(
        obs, goal, seed=jax.random.PRNGKey(0)
    )
    assert action.shape == (5,)
    assert np.all(np.isfinite(np.asarray(action)))
