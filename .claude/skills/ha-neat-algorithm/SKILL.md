---
name: ha-neat-algorithm
description: > 
 Precise algorithmic reference for NEAT (Stanley & Miikkulainen 2002) and HA-NEAT  (Hagg et al. 2017) — Heterogeneous Activation NEAT. Use this skill whenever working on HA-NEAT implementation, debugging neuroevolution code, discussing genome encoding, speciation, fitness sharing, mutation operators, or any aspect of NEAT-based algorithms. Invoke when the user mentions: NEAT, HA-NEAT, speciation, innovation numbers, genome, crossover, activation function mutation, TensorNEAT, neuroevolution, topology evolution, or asks about their thesis algorithm details.
---

# NEAT & HA-NEAT Algorithm Reference

Sven's thesis implements **HA-NEAT** — an extension of NEAT that co-evolves per-node
activation functions alongside topology and weights, evaluated on continuous control
(Hopper, Walker2D) using JAX/Brax. This skill provides precise algorithmic grounding
for both the base NEAT algorithm and the HA-NEAT extension.

---

## 1. NEAT Core (Stanley & Miikkulainen, 2002)

### 1.1 Genome Encoding
Direct encoding. Two gene types:

**Node genes:** `[id, type (input/hidden/output)]`  
**Connection genes:** `[in_node, out_node, weight, enabled_bit, innovation_number]`

- Innovation numbers are **global, monotonically increasing integers** assigned at structural mutation time
- Same structural mutation in the same generation → same innovation number (deduplication via generation-level log)
- Innovation numbers never change after assignment

### 1.2 Three Core Mechanisms

**Historical Markings (solving competing conventions)**
- Crossover aligns genes by innovation number, not position
- Matching genes (same innov #): randomly inherited from either parent
- Disjoint genes (non-matching, within range): inherited from more fit parent
- Excess genes (beyond range of shorter genome): inherited from more fit parent
- Equal fitness → disjoint/excess inherited randomly

**Speciation (protecting innovation)**
Compatibility distance δ between two genomes:
```
δ = (c1 * E / N) + (c2 * D / N) + c3 * W̄
```
Where:
- E = number of excess genes
- D = number of disjoint genes  
- W̄ = average weight difference of matching genes (incl. disabled)
- N = gene count of larger genome (set to 1 if both < 20 genes)
- c1, c2, c3 = tunable coefficients (typical: c1=1.0, c2=1.0, c3=0.4)

Each genome is placed in the first species whose representative it's compatible with (δ < δt). If none: new species created.

**Explicit Fitness Sharing**
Adjusted fitness for organism i:
```
f'_i = f_i / Σ_j sh(δ(i,j))
```
Where sh(δ) = 1 if δ < δt, else 0. Denominator reduces to species size.
Species reproduce proportionally to Σ f'_i. Lowest performers eliminated before reproduction.

**Minimal Structure / Incremental Growth**
- Initial population: zero hidden nodes (inputs connect directly to outputs)
- Only two structural mutations: **add connection** (random weight) and **add node** (splits existing connection: old disabled, two new added with weight 1.0 into node, same weight out of node)
- Growth justified by fitness → no ad-hoc complexity penalties needed
- Speciation enables starting minimal (innovations survive in their niche)

### 1.3 Mutation Operators
| Operator | What changes |
|---|---|
| Weight perturbation | Each weight: 90% perturb, 10% new random value (80% genome chance) |
| Add connection | New gene, random weight, connects previously unconnected nodes |
| Add node | Splits connection: disables old, adds 2 new (weight-in=1.0, weight-out=old weight) |
| Enable/disable | Gene enable bit toggled |

### 1.4 Key Design Choices & Justifications
- **Why minimal init?** Reduces dimensionality of weight search space throughout evolution, not just at end. Random init forces searching unnecessarily high-dimensional spaces (7x slower in ablations).
- **Why speciation?** Without it, population quickly converges on initially best topology (within ~10 generations), draining diversity.
- **Why historical markings over topological analysis?** O(1) per gene vs expensive graph matching; enables crossover across different-sized genomes.
- **Ablation results (DPV task):** Full NEAT 3,600 evals. No-Growth: 30,239 evals, 80% failure. No-Speciation: 25,600 evals, 25% failure. Random-Init: 23,033 evals, 5% failure. No-Crossover: 5,557 evals.

---

## 2. HA-NEAT Extension (Hagg, Mensing & Asteroth, 2017)

### 2.1 What HA-NEAT Adds to NEAT
Single modification: node genes are extended with an **activation function gene**:

**Extended node genes:** `[id, type, activation_function_index]`

- Input/output nodes: fixed linear activation
- Hidden nodes: activation function is part of the evolvable genome

### 2.2 Genome Structure (HA-NEAT)
```
Node genes:   [id | type | activation]
                1    input   none
                2    input   none
                3    hidden  tanh        ← activation is evolvable
                4    hidden  sigmoid     ← activation is evolvable
                5    output  lin

Connection genes: [source | target | weight | status | innovation]
                    1→3      0.7    enabled    1
                    2→4      0.3   disabled    2
                    ...
```

### 2.3 Activation Function Set
HA-NEAT uses a **discrete set** of qualitatively different functions (not parameterized):
- **Step** (discontinuous)
- **ReLU** (non-differentiable)
- **Sigmoid** (smooth, bounded)
- **Gaussian** (locally active, non-monotonic)

Rationale: Qualitatively different functions reduce nodes needed. E.g., approximating a Gaussian target takes 2 sigmoid nodes but only 1 Gaussian node. Mixed-function networks are more expressive per node.

**tanh is excluded** because its output range requires separate normalization in mixed networks — would require output scaling per neuron in heterogeneous context.

### 2.4 New/Modified Mutation Operators

**mutate_activation (new operator):**
- Randomly selects one node, assigns new random activation from set
- Max 1 node changed per genome per generation (prevents cascading disruption)
- Connection weights optimized for old activation → fitness drop expected after mutation
- Operator is **destructive but beneficial**: convergence slower initially, surpasses no-mutation variant after ~500-750 generations

**add_node (modified):**
- When creating new node: random activation function selected (vs fixed sigmoid in vanilla NEAT)

**Speciation adjustment for activation mutation:**
- When activation changes: node ID and **all incoming/outgoing connection innovation numbers are reassigned**
- This significantly increases speciation distance δ for the mutated individual
- Intentional: large qualitative changes to network function → individual placed in new niche → protected during optimization

### 2.5 Why Heterogeneous Networks Are More Parsimonious
The core claim: activation function choice determines how many nodes are needed to approximate a target function. With the right activation, fewer nodes suffice → smaller weight search space → faster convergence.

Empirical results:
- Heterogeneous nodes: 32/10/20 (cholesterol/engine/cancer datasets)
- Homogeneous nodes: 51/16/29 (same datasets)
- Heterogeneous connections: 124/35/116 vs homogeneous 225/170/197
- Accuracy: HA-NEAT matches best homogeneous networks, not just average

**Key nuance:** HA-NEAT doesn't simply find the best single activation and use it uniformly. Figure 10 in the paper shows non-trivial mixed distributions — HA-NEAT finds qualitatively different solutions than any homogeneous variant.

### 2.6 HA-NEAT vs HyperNEAT Distinction (important for thesis)
| Aspect | HA-NEAT | HyperNEAT |
|---|---|---|
| Encoding | **Direct** (explicit nodes/connections) | Indirect (CPPN generates weight matrix) |
| Network size | Parsimonious / small | Large-scale |
| Activation evolution | Per-node mutation operator | Fixed at CPPN node creation |
| Input representation | Raw data → inputs directly | Geometric coordinates of connections |
| Target domain | Regression, classification, **control** | Primarily high-dim control/vision |
| mutate_activation | ✅ (key addition) | ❌ (not used) |

HA-NEAT is a **directly encoded** minimal network approach. Sven's thesis extends it to **continuous control** (Brax/JAX), which is novel relative to the original paper's regression/classification focus.

---

## 3. Sven's HA-NEAT Thesis Extensions

Relative to the original Hagg 2017 paper, the thesis implementation includes:

- **Continuous control evaluation**: Hopper and Walker2D (Brax) instead of regression benchmarks
- **Multi-objective optimization**: NSGA-II applied to fitness objectives
- **JAX-native implementation**: via TensorNEAT fork for GPU-accelerated fitness evaluation
- **Activation function set**: may differ from original paper's Step/ReLU/Sigmoid/Gaussian — check actual implementation

When discussing results, the key comparison axes are:
1. HA-NEAT vs vanilla NEAT (homogeneous): network size, fitness, convergence
2. Performance across activation function types
3. Behavior on continuous control vs regression (generalization of parsimony hypothesis)

---

## 4. Common Precision Traps

**Don't conflate:**
- "Speciation distance" (δ, computed from excess/disjoint/weight diff) ≠ "fitness sharing" (the adjusted fitness formula)
- "Innovation number" (historical marker, never changes) ≠ "gene position in genome" (can change)
- HA-NEAT's activation mutation "reassigns innovation numbers" ≠ the original innovation number tracking being violated — it's a deliberate new mutation creating genuinely new structural genes

**Terminology consistency for thesis:**
- "Homogeneous network" = fixed single activation function for all hidden nodes
- "Heterogeneous network" = per-node activation functions co-evolved with topology/weights
- "Parsimony" = fewer nodes/connections achieving equivalent fitness (the primary HA-NEAT claim)
- "Structural innovation" = topology changes (add node / add connection mutations)
- "Speciation" protects structural innovations by giving them time to optimize in their niche