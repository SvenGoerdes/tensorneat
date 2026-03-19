from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp

from ..base import BaseProblem
from .rl_jit import RLEnv
from tensorneat.common import State, StatefulBaseClass


BRAX_REFERENCE_REWARDS = {
    "hopper": 3000.0,
    "walker2d": 5000.0,
    "ant": 6000.0,
    "halfcheetah": 8000.0,
    "humanoid": 6000.0,
    "reacher": 5.0,
}


@dataclass
class TaskSpec:
    """Configuration for one sub-task in a multi-task setup."""

    env: RLEnv
    obs_size: int
    act_size: int
    weight: float = 1.0
    max_reward: float = 1.0


class FitnessAggregator(StatefulBaseClass):
    """Base class for combining per-task fitnesses into a scalar."""

    def aggregate(self, fitnesses, weights):
        raise NotImplementedError


class WeightedSum(FitnessAggregator):
    """Weighted sum of per-task fitnesses."""

    def aggregate(self, fitnesses, weights):
        return jnp.dot(fitnesses, weights)


class NormalizedWeightedSum(FitnessAggregator):
    """Normalize per-task fitness by max_reward before weighting.

    Each task's fitness is divided by its max_reward so tasks with
    different reward scales contribute equally to the aggregate.
    """

    def __init__(self, max_rewards: jnp.ndarray):
        assert jnp.all(jnp.array(max_rewards) > 0), "max_rewards must be positive"
        self.max_rewards = max_rewards

    def aggregate(self, fitnesses, weights):
        normalized = fitnesses / self.max_rewards
        return jnp.dot(normalized, weights)


class NormalizedMin(FitnessAggregator):
    """Minimum of normalized per-task fitnesses.

    Forces evolution to improve the worst-performing task.
    Prevents task dominance by a single easy task.
    """

    def __init__(self, max_rewards: jnp.ndarray):
        assert jnp.all(jnp.array(max_rewards) > 0), "max_rewards must be positive"
        self.max_rewards = max_rewards

    def aggregate(self, fitnesses, weights):
        normalized = fitnesses / self.max_rewards
        return jnp.min(normalized)


class NormalizedProduct(FitnessAggregator):
    """Product of normalized per-task fitnesses.

    Rewards balanced performance across all tasks; a near-zero score
    on any single task collapses the overall fitness.
    """

    def __init__(self, max_rewards: jnp.ndarray):
        assert jnp.all(jnp.array(max_rewards) > 0), "max_rewards must be positive"
        self.max_rewards = max_rewards

    def aggregate(self, fitnesses, weights):
        normalized = fitnesses / self.max_rewards
        return jnp.prod(normalized)


class MultiTaskBraxEnv(BaseProblem):
    """
    Evaluates a single shared network on multiple RL tasks.

    Observations are zero-padded to the max obs size across tasks.
    Actions are sliced to the native action size of each task.
    """

    jitable = True

    AGGREGATION_MODES = ("normalized_sum", "normalized_min", "normalized_product")

    def __init__(
        self,
        tasks: List[TaskSpec],
        aggregator: Optional[FitnessAggregator] = None,
        aggregation_mode: str = "normalized_sum",
    ):
        super().__init__()
        assert len(tasks) >= 2, "Need at least 2 tasks for multi-task"
        assert aggregation_mode in self.AGGREGATION_MODES, (
            f"aggregation_mode must be one of {self.AGGREGATION_MODES}, got '{aggregation_mode}'"
        )
        self.tasks = tasks
        self._max_obs = max(t.obs_size for t in tasks)
        self._max_act = max(t.act_size for t in tasks)
        self._weights = jnp.array([t.weight for t in tasks])

        if aggregator is not None:
            self.aggregator = aggregator
        elif any(t.max_reward != 1.0 for t in tasks):
            max_rewards = jnp.array([t.max_reward for t in tasks])
            if aggregation_mode == "normalized_min":
                self.aggregator = NormalizedMin(max_rewards=max_rewards)
            elif aggregation_mode == "normalized_product":
                self.aggregator = NormalizedProduct(max_rewards=max_rewards)
            else:
                self.aggregator = NormalizedWeightedSum(max_rewards=max_rewards)
        else:
            self.aggregator = WeightedSum()

    @property
    def input_shape(self):
        return (self._max_obs,)

    @property
    def output_shape(self):
        return (self._max_act,)

    def setup(self, state=State()):
        for task in self.tasks:
            state = task.env.setup(state)
        return state

    def evaluate(self, state: State, randkey, act_func: Callable, params):
        fitnesses = []
        for i, task in enumerate(self.tasks):
            key = jax.random.fold_in(randkey, i)
            adapted = self._make_adapted_act_func(
                act_func, task.obs_size, task.act_size
            )
            fitness = task.env.evaluate(state, key, adapted, params)
            fitnesses.append(fitness)
        fitnesses = jnp.array(fitnesses)
        return self.aggregator.aggregate(fitnesses, self._weights)

    def _make_adapted_act_func(self, act_func, obs_size, act_size):
        """Create an act_func that pads obs and slices actions."""
        max_obs = self._max_obs

        def adapted(state, params, obs):
            padded = jnp.concatenate([obs, jnp.zeros(max_obs - obs_size)])
            full_action = act_func(state, params, padded)
            return full_action[:act_size]

        return adapted

    def per_task_evaluate(self, state, randkey, act_func, params):
        """Return per-task fitnesses: both raw and normalized (raw / max_reward)."""
        result = {}
        for i, task in enumerate(self.tasks):
            key = jax.random.fold_in(randkey, i)
            adapted = self._make_adapted_act_func(act_func, task.obs_size, task.act_size)
            fitness = task.env.evaluate(state, key, adapted, params)
            raw = float(fitness)
            result[f"fitness/{task.env.env_name}"] = raw
            result[f"fitness_normalized/{task.env.env_name}"] = raw / task.max_reward
        return result

    def show(self, state, randkey, act_func, params, task_index=0, *args, **kwargs):
        """Visualize the network's behavior on a specific task."""
        task = self.tasks[task_index]
        adapted = self._make_adapted_act_func(act_func, task.obs_size, task.act_size)
        return task.env.show(state, randkey, adapted, params, *args, **kwargs)

    def show_details(self, state, randkey, act_func, pop_params, *args, **kwargs):
        """Print per-task fitness statistics for the population."""
        for i, task in enumerate(self.tasks):
            adapted = self._make_adapted_act_func(
                act_func, task.obs_size, task.act_size
            )

            def eval_one(key, params):
                return task.env.evaluate(state, key, adapted, params)

            keys = jax.random.split(randkey, pop_params.shape[0])
            fitnesses = jax.vmap(eval_one)(keys, pop_params)
            print(
                f"  Task {i} ({task.env.env_name}): "
                f"mean={fitnesses.mean():.1f}, "
                f"max={fitnesses.max():.1f}, "
                f"min={fitnesses.min():.1f}"
            )
