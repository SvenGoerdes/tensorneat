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


# --- Auto-selection tests ---

def test_multi_task_auto_selects_normalized_aggregator():
    """MultiTaskBraxEnv auto-selects NormalizedWeightedSum when max_reward is set."""
    from unittest.mock import MagicMock
    from tensorneat.problem.rl.multi_task import MultiTaskBraxEnv, TaskSpec, NormalizedWeightedSum, WeightedSum

    mock_env1 = MagicMock()
    mock_env1.env_name = "hopper"
    mock_env2 = MagicMock()
    mock_env2.env_name = "walker2d"

    # With max_reward set -> NormalizedWeightedSum
    tasks_normalized = [
        TaskSpec(env=mock_env1, obs_size=11, act_size=3, max_reward=3000.0),
        TaskSpec(env=mock_env2, obs_size=17, act_size=6, max_reward=5000.0),
    ]
    mt = MultiTaskBraxEnv(tasks=tasks_normalized)
    assert isinstance(mt.aggregator, NormalizedWeightedSum)

    # Without max_reward -> WeightedSum
    tasks_default = [
        TaskSpec(env=mock_env1, obs_size=11, act_size=3),
        TaskSpec(env=mock_env2, obs_size=17, act_size=6),
    ]
    mt2 = MultiTaskBraxEnv(tasks=tasks_default)
    assert isinstance(mt2.aggregator, WeightedSum)


def test_multi_task_explicit_aggregator_overrides_auto():
    """Explicit aggregator should override auto-selection."""
    from unittest.mock import MagicMock
    from tensorneat.problem.rl.multi_task import MultiTaskBraxEnv, TaskSpec, WeightedSum

    mock_env1 = MagicMock()
    mock_env1.env_name = "hopper"
    mock_env2 = MagicMock()
    mock_env2.env_name = "walker2d"

    explicit_agg = WeightedSum()
    tasks = [
        TaskSpec(env=mock_env1, obs_size=11, act_size=3, max_reward=3000.0),
        TaskSpec(env=mock_env2, obs_size=17, act_size=6, max_reward=5000.0),
    ]
    mt = MultiTaskBraxEnv(tasks=tasks, aggregator=explicit_agg)
    assert mt.aggregator is explicit_agg


# --- per_task_evaluate dual logging tests ---

def test_per_task_evaluate_returns_raw_and_normalized():
    """per_task_evaluate returns both raw and normalized fitness keys."""
    from unittest.mock import MagicMock
    from tensorneat.problem.rl.multi_task import MultiTaskBraxEnv, TaskSpec, BRAX_REFERENCE_REWARDS
    from tensorneat.common import State
    import jax

    mock_env1 = MagicMock()
    mock_env1.env_name = "hopper"
    mock_env1.evaluate = MagicMock(return_value=jnp.array(1500.0))
    mock_env2 = MagicMock()
    mock_env2.env_name = "walker2d"
    mock_env2.evaluate = MagicMock(return_value=jnp.array(2500.0))

    hopper_max = BRAX_REFERENCE_REWARDS["hopper"]
    walker_max = BRAX_REFERENCE_REWARDS["walker2d"]
    tasks = [
        TaskSpec(env=mock_env1, obs_size=11, act_size=3, max_reward=hopper_max),
        TaskSpec(env=mock_env2, obs_size=17, act_size=6, max_reward=walker_max),
    ]
    mt = MultiTaskBraxEnv(tasks=tasks)

    dummy_state = State()
    dummy_key = jax.random.PRNGKey(0)
    dummy_act_func = lambda state, params, obs: jnp.zeros(6)

    result = mt.per_task_evaluate(dummy_state, dummy_key, dummy_act_func, None)

    # Raw keys
    assert "fitness/hopper" in result
    assert "fitness/walker2d" in result
    # Normalized keys
    assert "fitness_normalized/hopper" in result
    assert "fitness_normalized/walker2d" in result
    # Check values
    assert abs(result["fitness/hopper"] - 1500.0) < 1e-5
    assert abs(result["fitness_normalized/hopper"] - 1500.0 / hopper_max) < 1e-5
    assert abs(result["fitness_normalized/walker2d"] - 2500.0 / walker_max) < 1e-5
