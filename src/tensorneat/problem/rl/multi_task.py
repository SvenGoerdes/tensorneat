from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp

from ..base import BaseProblem
from .rl_jit import RLEnv
from tensorneat.common import State, StatefulBaseClass


@dataclass
class TaskSpec:
    """Configuration for one sub-task in a multi-task setup."""

    env: RLEnv
    obs_size: int
    act_size: int
    weight: float = 1.0


class FitnessAggregator(StatefulBaseClass):
    """Base class for combining per-task fitnesses into a scalar."""

    def aggregate(self, fitnesses, weights):
        raise NotImplementedError


class WeightedSum(FitnessAggregator):
    """Weighted sum of per-task fitnesses."""

    def aggregate(self, fitnesses, weights):
        return jnp.dot(fitnesses, weights)


class MultiTaskBraxEnv(BaseProblem):
    """
    Evaluates a single shared network on multiple RL tasks.

    Observations are zero-padded to the max obs size across tasks.
    Actions are sliced to the native action size of each task.
    """

    jitable = True

    def __init__(
        self,
        tasks: List[TaskSpec],
        aggregator: Optional[FitnessAggregator] = None,
    ):
        super().__init__()
        assert len(tasks) >= 2, "Need at least 2 tasks for multi-task"
        self.tasks = tasks
        self.aggregator = aggregator or WeightedSum()
        self._max_obs = max(t.obs_size for t in tasks)
        self._max_act = max(t.act_size for t in tasks)
        self._weights = jnp.array([t.weight for t in tasks])

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
