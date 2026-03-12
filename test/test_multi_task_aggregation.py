import jax.numpy as jnp
import pytest


# --- Lookup table tests ---

def test_brax_reference_rewards_contains_known_envs():
    from tensorneat.problem.rl.multi_task import BRAX_REFERENCE_REWARDS

    assert "hopper" in BRAX_REFERENCE_REWARDS
    assert "walker2d" in BRAX_REFERENCE_REWARDS
    assert BRAX_REFERENCE_REWARDS["hopper"] > 0
    assert BRAX_REFERENCE_REWARDS["walker2d"] > 0


def test_taskspec_max_reward_default():
    """max_reward defaults to 1.0 when not provided."""
    from unittest.mock import MagicMock
    from tensorneat.problem.rl.multi_task import TaskSpec

    task = TaskSpec(env=MagicMock(), obs_size=11, act_size=3)
    assert task.max_reward == 1.0


# --- WeightedSum tests ---

def test_weighted_sum_equal_weights():
    from tensorneat.problem.rl.multi_task import WeightedSum

    agg = WeightedSum()
    fitnesses = jnp.array([3000.0, 5000.0])
    weights = jnp.array([1.0, 1.0])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 8000.0)


def test_weighted_sum_unequal_weights():
    from tensorneat.problem.rl.multi_task import WeightedSum

    agg = WeightedSum()
    fitnesses = jnp.array([3000.0, 5000.0])
    weights = jnp.array([0.5, 0.5])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 4000.0)


# --- NormalizedWeightedSum tests ---

def test_normalized_weighted_sum_equal_contribution():
    """Tasks at their max reward should contribute equally."""
    from tensorneat.problem.rl.multi_task import NormalizedWeightedSum, BRAX_REFERENCE_REWARDS

    max_rewards = jnp.array([BRAX_REFERENCE_REWARDS["hopper"], BRAX_REFERENCE_REWARDS["walker2d"]])
    agg = NormalizedWeightedSum(max_rewards=max_rewards)
    fitnesses = max_rewards  # Both at max
    weights = jnp.array([1.0, 1.0])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 2.0)


def test_normalized_weighted_sum_half_performance():
    """Half performance on both tasks should give 1.0."""
    from tensorneat.problem.rl.multi_task import NormalizedWeightedSum, BRAX_REFERENCE_REWARDS

    max_rewards = jnp.array([BRAX_REFERENCE_REWARDS["hopper"], BRAX_REFERENCE_REWARDS["walker2d"]])
    agg = NormalizedWeightedSum(max_rewards=max_rewards)
    fitnesses = max_rewards / 2
    weights = jnp.array([1.0, 1.0])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 1.0)


def test_normalized_weighted_sum_unequal_weights():
    """Weights still apply after normalization."""
    from tensorneat.problem.rl.multi_task import NormalizedWeightedSum, BRAX_REFERENCE_REWARDS

    max_rewards = jnp.array([BRAX_REFERENCE_REWARDS["hopper"], BRAX_REFERENCE_REWARDS["walker2d"]])
    agg = NormalizedWeightedSum(max_rewards=max_rewards)
    fitnesses = max_rewards
    weights = jnp.array([2.0, 1.0])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 3.0)


def test_normalized_weighted_sum_zero_fitness():
    from tensorneat.problem.rl.multi_task import NormalizedWeightedSum, BRAX_REFERENCE_REWARDS

    max_rewards = jnp.array([BRAX_REFERENCE_REWARDS["hopper"], BRAX_REFERENCE_REWARDS["walker2d"]])
    agg = NormalizedWeightedSum(max_rewards=max_rewards)
    fitnesses = jnp.array([0.0, 0.0])
    weights = jnp.array([1.0, 1.0])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 0.0)


def test_normalized_weighted_sum_negative_fitness():
    from tensorneat.problem.rl.multi_task import NormalizedWeightedSum, BRAX_REFERENCE_REWARDS

    hopper_max = BRAX_REFERENCE_REWARDS["hopper"]
    walker_max = BRAX_REFERENCE_REWARDS["walker2d"]
    max_rewards = jnp.array([hopper_max, walker_max])
    agg = NormalizedWeightedSum(max_rewards=max_rewards)
    fitnesses = jnp.array([-hopper_max * 0.1, walker_max * 0.5])
    weights = jnp.array([1.0, 1.0])
    result = agg.aggregate(fitnesses, weights)
    assert jnp.isclose(result, 0.4)


def test_normalized_weighted_sum_rejects_zero_max_reward():
    from tensorneat.problem.rl.multi_task import NormalizedWeightedSum

    with pytest.raises(AssertionError):
        NormalizedWeightedSum(max_rewards=jnp.array([0.0, 5000.0]))
