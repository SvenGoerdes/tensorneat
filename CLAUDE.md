# TensorNEAT

GPU-accelerated NEAT (NeuroEvolution of Augmenting Topologies) built on JAX.

## Build & Run

```bash
# Install (editable mode)
uv pip install -e .

# Run an example
uv run python examples/func_fit/xor.py

# Run tests
uv run pytest test/
```

Package source is in `src/tensorneat/`, configured via `pyproject.toml` with `setuptools`.
Python >= 3.10 required. Key deps: jax, brax, gymnax, flax, optax, networkx, sympy.

## Architecture

Four-layer design: **Pipeline > Algorithm > Genome > Problem**

### Pipeline (`pipeline.py`)
Orchestrates evolution: `setup(state) -> step(state) -> auto_run(state)`.
Each step: `ask` (get population) -> `transform` -> `evaluate` -> `tell` (update).
Supports batched evaluation, multi-device via `pmap`.

### Algorithm (`algorithm/`)
- **NEAT** (`neat/neat.py`): Speciation-driven evolution with `SpeciesController`
- **HyperNEAT** (`hyperneat/`): Evolves CPPN to generate substrate weights. Uses `Substrate` classes (`FullSubstrate`, `MLPSubstrate`, `DefaultSubstrate`)

Base class: `BaseAlgorithm` — implement `ask()`, `tell()`, `transform()`, `forward()`

### Genome (`genome/`)
- **DefaultGenome**: Feedforward networks with topological sort
- **RecurrentGenome**: Recurrent networks with configurable activation cycles
- **Gene types** (`genome/gene/`): `DefaultNode`, `BiasNode`, `DefaultConn` — each defines `fixed_attrs` (structural) and `custom_attrs` (learnable)
- **Operations** (`genome/operations/`): `DefaultMutation`, `DefaultCrossover`, `DefaultDistance`

Base class: `BaseGenome` — implement `transform()`, `forward()`, `initialize()`

### Problem (`problem/`)
- **FuncFit** (`func_fit/`): XOR, custom symbolic regression
- **RL** (`rl/`): `BraxEnv`, `GymnaxEnv`, `MujocoPlayground`, `MultiTaskBraxEnv`

Base class: `BaseProblem` — implement `evaluate()`, `input_shape`, `output_shape`

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
  pipeline.py          # Main orchestrator
  algorithm/           # NEAT, HyperNEAT
  genome/              # Network representations + operations
  problem/             # Evaluation environments
  common/              # State, utilities, sympy tools
examples/              # Usage examples (brax/, func_fit/, gymnax/, hyperneat/)
test/                  # pytest tests + notebooks
tutorials/             # Learning resources
```
