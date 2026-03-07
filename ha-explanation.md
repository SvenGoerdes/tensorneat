# HA-NEAT Implementation — Detailed Explanation

## File 1: `src/tensorneat/genome/operations/mutation/default.py` — The base mutation class

### What the original code did

`DefaultMutation` is the standard NEAT mutation operator. Each generation, for every genome in the population, NEAT calls `mutation(state, genome, randkey, nodes, conns, new_node_key, new_conn_key)`. This does two things in sequence:

1. **`mutate_structure`** — probabilistically adds/deletes nodes and connections (topological changes)
2. **`mutate_values`** — vmaps `gene.mutate()` over every node and every connection to perturb their continuous attributes (bias, weight, activation index, etc.)

The `new_conn_key` parameter is a pre-allocated array of fresh historical marker IDs. The original code hardcoded this to exactly shape `(3,)` because at most 3 new connections can be created per mutation: 2 from node-add (splitting a connection creates 2 new ones) + 1 from connection-add.

### Change 1: `num_new_conn_markers` property (line 34-36)

```python
@property
def num_new_conn_markers(self):
    return 3
```

**Why:** The NEAT algorithm needs to know how many historical markers to pre-allocate per genome per generation. Previously this was hardcoded to `3` in `neat.py`. By putting it on the mutation object as a property, the mutation class can declare how many markers it needs. `DefaultMutation` returns `3` (unchanged behavior). `HANEATMutation` will override this to return `3 + max_conns` because it needs extra markers for reassignment when activations change.

This is the "communication channel" between the mutation operator and the NEAT algorithm — the mutation says "I need N markers per genome" and NEAT allocates them.

### Change 2: Relaxed assertion (line 44)

```python
# Before:
assert new_conn_key.shape == (3,)

# After:
assert len(new_conn_key.shape) == 1 and new_conn_key.shape[0] >= 3
```

**Why:** The old assertion demanded *exactly* 3 markers. But `HANEATMutation` needs more than 3 (it needs `3 + max_conns`). The structural mutation code only ever accesses indices `[0]`, `[1]`, `[2]` — so any array of size ≥ 3 works fine. The relaxed assertion says "must be a 1D array with at least 3 elements", which is backward-compatible (a shape-`(3,)` array still passes) and forward-compatible with larger marker arrays.

### Change 3: `_new_node_attrs` hook method (lines 55-57)

```python
def _new_node_attrs(self, state, genome, randkey):
    return genome.node_gene.new_identity_attrs(state)
```

**Why:** When `mutate_add_node` creates a new node, it needs to decide that node's attributes (bias, response, aggregation, activation). In standard NEAT, the node gets "identity" attributes — bias=0, response=1, default activation — so that inserting the node doesn't change the network's output (it's a neutral structural mutation that can later be refined by value mutation).

HA-NEAT wants new nodes to start with a *random* activation function instead. Rather than copy-pasting the entire 50-line `mutate_add_node` closure just to change one line, I extracted this decision into a small hook method that subclasses can override. `DefaultMutation` calls `new_identity_attrs` (original behavior), `HANEATMutation` overrides it to call `new_random_attrs`.

### Change 4: Random key split in `mutate_add_node` (line 69)

```python
key_, key_node_attrs = jax.random.split(key_)
```

**Why:** The `_new_node_attrs` hook in `HANEATMutation` calls `new_random_attrs`, which needs a PRNG key to sample random activation/bias/etc. The original code didn't need a key here because `new_identity_attrs` is deterministic. So I split the existing key to produce a sub-key for the hook. This is consumed at line 81:

```python
self._new_node_attrs(state, genome, key_node_attrs)
```

In `DefaultMutation`, this key is passed but ignored (identity attrs don't use randomness). In `HANEATMutation`, it's used to sample the random activation.

---

## File 2: `src/tensorneat/algorithm/neat/neat.py` — The NEAT algorithm

### What the original code did

In `_create_next_generation` (the method that creates the next population after selection), NEAT pre-allocates historical marker IDs for the entire population. It finds the current maximum marker across all genomes, then creates a `(pop_size, 3)` array where each genome gets 3 fresh, globally-unique marker IDs. These are passed to each genome's mutation call.

### Change 1: Store `num_new_conn_markers` in `__init__` (line 37)

```python
self.num_new_conn_markers = getattr(genome.mutation, 'num_new_conn_markers', 3)
```

**Why:** This reads how many markers the mutation operator needs. `getattr` with default `3` provides backward compatibility — if someone has a custom mutation class that doesn't define this property, it falls back to the original `3`.

### Change 2: Use `self.num_new_conn_markers` instead of hardcoded `3` (lines 130-138)

```python
# Before:
new_conn_markers = jnp.arange(self.pop_size * 3).reshape(self.pop_size, 3) + next_conn_markers
# ...
new_conn_markers = jnp.full((self.pop_size, 3), 0)

# After:
n = self.num_new_conn_markers
new_conn_markers = jnp.arange(self.pop_size * n).reshape(self.pop_size, n) + next_conn_markers
# ...
new_conn_markers = jnp.full((self.pop_size, self.num_new_conn_markers), 0)
```

**Why:** This is the actual allocation. Each genome now gets `n` unique markers instead of `3`. For `DefaultMutation`, `n=3` so nothing changes. For `HANEATMutation` with `max_conns=50`, `n=53`, so each genome gets 53 pre-allocated markers: 3 for structural mutations + 50 spare ones for reassigning markers on connections affected by activation changes.

The `jnp.arange(pop_size * n) + next_conn_markers` ensures every marker is globally unique across the entire population — genome 0 gets markers `[M, M+1, ..., M+52]`, genome 1 gets `[M+53, M+54, ..., M+105]`, etc. This is critical because historical markers are used by NEAT's speciation to align genomes for crossover and distance computation. If two genomes had the same marker on different connections, speciation would incorrectly treat them as homologous.

---

## File 3: `src/tensorneat/genome/operations/mutation/ha_neat.py` — The new HA-NEAT mutation operator

This is where all the HA-NEAT-specific logic lives. It extends `DefaultMutation`.

### `__init__` (lines 18-34)

Two new parameters beyond `DefaultMutation`:
- `activation_mutate_rate=0.1` — probability that a genome gets an activation mutation in a given generation
- `max_conns=100` — used to size the extra marker allocation

### `num_new_conn_markers` property (lines 36-38)

Returns `3 + self.max_conns`. The `3` covers standard structural mutations (same as `DefaultMutation`). The `+ max_conns` provides spare markers for reassignment. When an activation changes on a node, every connection touching that node needs a new marker. A node can have at most `max_conns` connections touching it (worst case), so we reserve that many.

**Why not allocate dynamically?** JAX requires all array shapes to be known at compile time (JIT). You can't say "allocate as many markers as there are affected connections" because that count varies per genome and per generation. So we pre-allocate the worst case and only use as many as needed.

### `_new_node_attrs` override (lines 40-42)

```python
def _new_node_attrs(self, state, genome, randkey):
    return genome.node_gene.new_random_attrs(state, randkey)
```

**Why:** This is HA-NEAT feature #1 — new nodes get random activation functions. In standard NEAT, a new node gets identity/default activation, so inserting it is a neutral mutation. In HA-NEAT, the new node immediately gets a random activation (sigmoid, relu, tanh, etc.), which means the structural mutation is *not* neutral — it immediately introduces functional diversity. This is the key insight of HA-NEAT: activation heterogeneity should be present from the moment of node creation, not only through later mutation.

### `__call__` override (lines 44-60)

```python
nodes, conns = self.mutate_structure(...)    # step 1: inherited
nodes, conns = self.mutate_activation(...)   # step 2: NEW - HA-NEAT activation mutation
nodes, conns = self.mutate_values(...)       # step 3: inherited but overridden
```

**Why:** The original `DefaultMutation.__call__` does 2 steps: structure, then values. HA-NEAT inserts a third step in between — `mutate_activation` — which is the controlled, one-node-per-generation activation mutation with speciation protection. The ordering matters:

1. **Structure first** — nodes/connections may be added or removed
2. **Activation mutation** — change one hidden node's activation and reassign markers on its connections
3. **Value mutation** — perturb continuous attributes (but with activation restored, see below)

### `mutate_activation` method (lines 62-113) — the core HA-NEAT logic

This is the most complex part. Here is a step-by-step walkthrough.

**Gating (line 66-67):**
```python
r = jax.random.uniform(k1)
return jax.lax.cond(r < self.activation_mutate_rate, do_mutate, no_mutate)
```
With probability `activation_mutate_rate` (default 10%), the genome undergoes activation mutation. Otherwise nothing happens. `jax.lax.cond` is used instead of Python `if` because this runs inside JIT — Python `if` on a traced value would fail.

**Node selection (lines 70-77):**
```python
node_key, node_idx = self.choose_node_key(
    k2, nodes, genome.input_idx, genome.output_idx,
    allow_input_keys=False, allow_output_keys=False,
)
```
Picks one random **hidden** node (excludes input and output nodes). This is the "one-node-per-generation" constraint — even if a genome has 10 hidden nodes, only 1 gets its activation changed. This is important because changing activation is a disruptive mutation (it changes the node's entire transfer function), so limiting it to one node prevents catastrophic destruction of learned behavior.

**Column index computation (lines 80-81):**
```python
n_fixed = len(genome.node_gene.fixed_attrs)  # = 1 (just "index")
act_col = n_fixed + genome.node_gene.custom_attrs.index("activation")  # = 1 + 3 = 4
```
Nodes are stored as flat arrays: `[index, bias, response, aggregation, activation]`. The activation is at column 4. This computes that index dynamically from the gene definition rather than hardcoding it, so it works with any node gene that has an "activation" custom attribute.

**Activation replacement (lines 84-89):**
```python
new_act = jax.random.choice(k3, jnp.array(genome.node_gene.activation_indices))
new_nodes = nodes.at[node_idx, act_col].set(
    jnp.where(node_idx != I_INF, new_act, nodes[node_idx, act_col])
)
```
Picks a random activation from the available options and sets it. The `jnp.where` guard handles the edge case where there are no hidden nodes (`node_idx == I_INF`), in which case we keep the original value (a no-op).

**Historical marker reassignment (lines 92-106) — speciation protection:**

This is the most important part for HA-NEAT's speciation behavior. Here's *why* it's needed:

In NEAT, speciation works by computing a "genetic distance" between genomes. With `OriginConn`, each connection has a historical marker that identifies *when and how* it was created. During distance computation, connections with the same marker are treated as "homologous" (same structural origin), while connections with different markers are "disjoint" or "excess" (different structural origins). Disjoint/excess connections increase genetic distance, which makes genomes more likely to be placed in different species.

When a node changes its activation function, it fundamentally changes what that node *does*. A sigmoid node and a relu node behave very differently, even if they have the same connections. Without marker reassignment, two genomes that differ only in one node's activation would be considered identical by speciation (same connections, same markers) — they'd be in the same species and could freely interbreed. But crossing over two genomes where a node has sigmoid in one and relu in the other can produce offspring that inherit a confusing mix, potentially destroying good solutions.

By reassigning markers on connections touching the mutated node, we make those connections look "new" to the speciation system. This increases genetic distance from genomes that still have the old activation, which tends to push the mutant into a separate species. That species then gets its own protected niche to explore the new activation without being overwhelmed by the old-activation majority. This is the "speciation protection" that HA-NEAT provides.

```python
active_mask = ~jnp.isnan(conns[:, 0])
affected = active_mask & (
    (conns[:, 0] == node_key) | (conns[:, 1] == node_key)
)
```
Find all active connections where the mutated node is either the source or destination.

```python
cumulative_idx = jnp.cumsum(affected) - 1
extra_markers = new_conn_key[3:]
new_markers = extra_markers[cumulative_idx]
```
The `jnp.cumsum` trick assigns consecutive indices to the affected connections: if connections 2, 5, 7 are affected, they get cumulative indices 0, 1, 2, which index into the pre-allocated marker array `new_conn_key[3:]`. The `[3:]` skips the first 3 markers reserved for structural mutations. This is a common JAX pattern for "assign N fresh values to a variable number of selected positions" without dynamic allocation.

```python
updated_markers = jnp.where(affected, new_markers, old_markers)
new_conns = conns.at[:, marker_col].set(updated_markers)
```
Replace markers only on affected connections; leave others unchanged.

The `if "historical_marker" in genome.conn_gene.fixed_attrs` check (line 92) means this only runs when using `OriginConn`. With standard `DefaultConn` (which has no historical markers), the marker reassignment is skipped — speciation protection is not possible without markers, but activation mutation still works.

### `mutate_values` override (lines 115-130)

```python
def mutate_values(self, state, genome, randkey, nodes, conns):
    original_acts = nodes[:, act_col]
    new_nodes, new_conns = super().mutate_values(state, genome, randkey, nodes, conns)
    new_nodes = new_nodes.at[:, act_col].set(original_acts)
    return new_nodes, new_conns
```

**Why this is needed:** The parent's `mutate_values` calls `genome.node_gene.mutate()` on every node via `vmap`. Inside `DefaultNode.mutate()` (line 134 of `default.py` in the node gene), there's this line:

```python
act = mutate_int(k3, act, self.activation_indices, self.activation_replace_rate)
```

This randomly replaces the activation with some probability. In standard NEAT, this is how activation mutation works — every node independently gets a chance to change activation each generation.

HA-NEAT wants to control activation mutation itself (one node per generation, with marker reassignment). If we let the standard `mutate_values` also change activations, we'd have *two* sources of activation mutation, and the standard one wouldn't have speciation protection.

**The save-and-restore pattern:** We save the activation column before calling `super().mutate_values()`, let it mutate everything (bias, response, aggregation, activation), then overwrite the activation column with the saved values. This effectively "undoes" any activation changes from the standard mutation while keeping all other value mutations intact.

**Why not just set `activation_replace_rate=0.0` on the node gene?** The example actually does this too (`activation_replace_rate=0.0`), as a belt-and-suspenders approach. But the override in `mutate_values` is the authoritative prevention — it works regardless of how the node gene is configured. Someone might forget to set `activation_replace_rate=0.0` when using `HANEATMutation`, and without this override, activations would be mutated uncontrollably.

---

## File 4: `__init__.py` exports

Just added `HANEATMutation` to the exports at both the `mutation/` and `operations/` levels so users can import it as `from tensorneat.genome import HANEATMutation`.

---

## File 5: `examples/func_fit/xor_ha_neat.py` — The example

This shows how to wire everything together:

1. **`DefaultNode` with 5 activation options and `activation_replace_rate=0.0`** — provides the activation palette and disables standard activation mutation at the gene level
2. **`OriginConn()`** — enables historical markers on connections, which is required for speciation protection via marker reassignment
3. **`HANEATMutation(activation_mutate_rate=0.1, max_conns=50)`** — the HA-NEAT mutation operator
4. **Standard `NEAT` algorithm** — no subclass needed; all HA-NEAT behavior is encapsulated in the mutation operator

The `max_conns=50` on `HANEATMutation` must match the `max_conns=50` on `DefaultGenome` because it determines how many spare markers to allocate.

---

## Summary of the design philosophy

The key design decision was: **HA-NEAT is a mutation-level concern, not an algorithm-level concern.** NEAT's ask/tell/transform/forward loop doesn't change — only *how* genomes are mutated changes. So instead of subclassing `NEAT` (which would duplicate a lot of code), all HA-NEAT logic lives in `HANEATMutation`, with two small hooks in the parent classes to support it:

1. `_new_node_attrs` hook — lets subclasses customize new node initialization without rewriting `mutate_add_node`
2. `num_new_conn_markers` property — lets the mutation class tell NEAT how many markers to allocate

Both hooks are backward-compatible: `DefaultMutation` defines them with the original behavior, and NEAT uses `getattr` with a fallback.
