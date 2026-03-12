# TensorNEAT + HA-NEAT

GPU-accelerated NEAT built on JAX, extended with **HA-NEAT** (Heterogeneous Activation NEAT) and **multi-task evaluation** for comparing NEAT vs HA-NEAT on Brax locomotion tasks.

## Project Goal

Compare standard NEAT against HA-NEAT on multi-task Brax environments (Hopper + Walker2D). A single evolved network must solve both tasks simultaneously using zero-padded observations and sliced actions.

## Build & Run

```bash
# Install (editable mode)
uv pip install -e .

# Run multi-task comparison
uv run python examples/brax/multi_task_hopper_walker.py

# Run HA-NEAT XOR example
uv run python examples/func_fit/xor_ha_neat.py

# Run tests
uv run pytest test/
```

Package source is in `src/tensorneat/`, configured via `pyproject.toml` with `setuptools`.
Python >= 3.10 required. Key deps: jax, brax, gymnax, flax, optax, networkx, sympy, mlflow.

## Architecture

Four-layer design: **Pipeline > Algorithm > Genome > Problem**

### Pipeline (`pipeline.py`)
Orchestrates evolution: `setup(state) -> step(state) -> auto_run(state)`.
Each step: `ask` (get population) -> `transform` -> `evaluate` -> `tell` (update).
Supports batched evaluation, multi-device via `pmap`, MLflow tracking, and per-task metric logging.

- `mlflow_tracking=True`: Logs per-generation metrics (fitness stats, species count, genome complexity, stagnation)
- `per_task_tracking=True`: Logs per-task fitness for multi-task problems via `per_task_evaluate()`

### Algorithm (`algorithm/`)
- **NEAT** (`neat/neat.py`): Speciation-driven evolution with `SpeciesController`
- **HyperNEAT** (`hyperneat/`): Evolves CPPN to generate substrate weights

Base class: `BaseAlgorithm` — implement `ask()`, `tell()`, `transform()`, `forward()`

### Genome (`genome/`)
- **DefaultGenome**: Feedforward networks with topological sort
- **RecurrentGenome**: Recurrent networks with configurable activation cycles
- **Gene types** (`genome/gene/`): `DefaultNode`, `BiasNode`, `DefaultConn`, `OriginConn`
- **Operations** (`genome/operations/`): `DefaultMutation`, `HANEATMutation`, `DefaultCrossover`, `DefaultDistance`

Base class: `BaseGenome` — implement `transform()`, `forward()`, `initialize()`

### Problem (`problem/`)
- **FuncFit** (`func_fit/`): XOR, custom symbolic regression
- **RL** (`rl/`): `BraxEnv`, `GymnaxEnv`, `MujocoPlayground`, `MultiTaskBraxEnv`

Base class: `BaseProblem` — implement `evaluate()`, `input_shape`, `output_shape`

## HA-NEAT (`genome/operations/mutation/ha_neat.py`)

HA-NEAT extends NEAT with per-node activation function diversity and speciation protection. Implemented entirely in `HANEATMutation`, a subclass of `DefaultMutation`.

**Three key mechanisms:**
1. **Random activation on node creation** — new nodes get a random activation (tanh, sigmoid, relu, sin, identity) instead of identity
2. **One-node-per-generation activation mutation** — with probability `activation_mutate_rate` (default 0.1), one random hidden node's activation is changed. All connections touching that node get new historical markers, pushing the mutant into a protected species
3. **Prevention of standard activation mutation** — overrides `mutate_values()` to restore activation column after parent mutation, ensuring only the controlled mechanism modifies activations

**Usage:**
```python
from tensorneat.genome.operations.mutation import HANEATMutation
from tensorneat.genome.gene import OriginConn

genome = DefaultGenome(
    node_gene=BiasNode(activation_options=[ACT.tanh, ACT.sigmoid, ACT.relu, ACT.sin, ACT.identity],
                       activation_replace_rate=0.0),
    conn_gene=OriginConn(),
    mutation=HANEATMutation(activation_mutate_rate=0.1, max_conns=50),
)
```

**Required:** `OriginConn` gene (tracks historical markers) and `activation_replace_rate=0.0` on node gene.

## Multi-Task Evaluation (`problem/rl/multi_task.py`)

`MultiTaskBraxEnv` evaluates a single shared network across multiple RL environments.

**I/O handling:** Network sized to largest task (max obs/act dimensions). Smaller tasks get zero-padded observations and sliced actions.

**Usage:**
```python
from tensorneat.problem.rl import MultiTaskBraxEnv, TaskSpec

hopper = TaskSpec(env=BraxEnv("hopper", max_step=1000), obs_size=11, act_size=3, weight=1.0)
walker = TaskSpec(env=BraxEnv("walker2d", max_step=1000), obs_size=17, act_size=6, weight=1.0)
problem = MultiTaskBraxEnv(tasks=[hopper, walker])
```

Fitness = weighted sum of per-task fitnesses (configurable via `FitnessAggregator`).

## Experimental Setup

**Comparison conditions:**
1. **Multi-Task NEAT** (baseline): Single tanh activation, standard mutation
2. **Multi-Task HA-NEAT** (treatment): 5 activation functions, historical marker reassignment, speciation protection

**Environments:** Hopper (11 obs, 3 act) + Walker2D (17 obs, 6 act)
**Network:** 17 inputs → 6 outputs (zero-padding/slicing for Hopper)
**Tracking:** MLflow per-generation metrics + per-task fitness trajectories

## JAX Patterns

### State (`common/state.py`)
Immutable functional state: `State(**kwargs)` with `.register()`, `.update()`, `.remove()`.
Registered as a JAX pytree for jit compatibility. All stateful classes inherit `StatefulClass`.

### Vectorization & Compilation
- `vmap`: Population-level evaluation and genome transformation
- `jit`: Pipeline step compilation; `lower() + compile()` in `auto_run()`
- `lax.fori_loop`: Forward pass node iteration, recurrent activation cycles, topological sort
- `lax.while_loop`: Speciation clustering, RL environment stepping, cycle detection
- `pmap`: Optional multi-device parallelization

### NaN Padding Convention
Networks have fixed `max_nodes`/`max_conns`. Inactive positions use NaN in first column.
Mask pattern: `~jnp.isnan(array[:, 0])`. Enables batching networks with different topologies.

### Tensorization
- Nodes: `[max_nodes, attrs_per_node]` — fixed attrs first, then custom attrs
- Conns: `[max_conns, attrs_per_conn]` — same layout
- `unflatten_conns()` creates `[max_nodes, max_nodes]` adjacency index for O(1) lookups
- `I_INF` (`np.iinfo(jnp.int32).max`) marks invalid indices

### Gradient Support
`genome.grad(state, nodes, conns, inputs, loss_fn)` — hybrid evolution + gradient descent.
Structural fields and NaN padding automatically zeroed in gradients.

## Extending

**New Problem**: Inherit `BaseProblem`, implement `evaluate()`, `input_shape`, `output_shape`. Set `jitable = True` for JIT.

**New Genome**: Inherit `BaseGenome`, implement `transform()`, `forward()`, `initialize()`. Provide custom gene, mutation, crossover, distance components.

**New Gene**: Inherit `BaseNode`/`BaseConn`, define `fixed_attrs`/`custom_attrs`, implement `new_random_attrs()`, `mutate()`, `distance()`, `forward()`.

**Activation/Aggregation**: `ACT.add_func("name", jnp_fn)`, `AGG.add_func("name", jnp_fn)`.

## Key Directories

```
src/tensorneat/        # Library source
  pipeline.py          # Main orchestrator (+ MLflow tracking)
  algorithm/           # NEAT, HyperNEAT
  genome/              # Network representations + operations
    operations/mutation/ha_neat.py  # HA-NEAT mutation operator
  problem/             # Evaluation environments
    rl/multi_task.py   # Multi-task evaluation
  common/              # State, utilities, sympy tools
examples/
  brax/                # Multi-task and HA-NEAT comparison experiments
  func_fit/            # XOR, xor_ha_neat
  gymnax/              # Gymnax environments
  hyperneat/           # HyperNEAT examples
test/                  # pytest tests + notebooks
tutorials/             # Learning resources
```
