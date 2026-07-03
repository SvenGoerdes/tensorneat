---
name: ha-neat-thesis-writing
description: >
  Writing assistant for Sven's HA-NEAT master's thesis at NOVA IMS. Use this skill
  whenever writing, editing, or reviewing thesis sections — especially Background,
  Methodology, Results, and Discussion. Enforces precise HA-NEAT terminology,
  consistent framing of the heterogeneous vs homogeneous comparison, correct citations
  (Stanley 2002, Hagg 2017), and ensures Sven's novel contributions (continuous
  control, JAX/Brax) are clearly framed as extensions beyond the original paper.
  Invoke for: drafting thesis sections, fixing terminology inconsistencies, framing
  results, writing the abstract, or structuring the Discussion/Conclusion.
---

# HA-NEAT Thesis Writing Guide

Sven is completing a master's thesis at NOVA IMS on **HA-NEAT (Heterogeneous Activation NEAT)**
evaluated on continuous control environments (Hopper, Walker2D) using JAX/Brax, with
MLflow for experiment tracking. This skill ensures writing is precise, consistent, and
correctly positions the work relative to prior literature.

---

## 1. Thesis Framing — The Core Narrative

**What HA-NEAT is:**
An extension of NEAT (Stanley & Miikkulainen, 2002) that co-evolves per-node activation
functions alongside topology and weights, producing more parsimonious (smaller) networks
that match or exceed the performance of fixed-activation (homogeneous) networks.

**What Sven's thesis adds beyond Hagg et al. (2017):**
The original HA-NEAT paper evaluated on regression and classification (static datasets).
Sven extends this to **continuous control** — a harder, dynamic domain. This is the
primary novelty claim. Additional extensions: multi-task evaluation aggregated to a
single scalarized fitness (`normalized_min`), and a JAX/Brax GPU-native evaluation pipeline.

**Important — single-objective, not multi-objective.** The thesis evaluates each genome on
two tasks (Hopper, Walker2D) but optimizes a *single* scalar: `normalized_min =
min(norm_hopper, norm_walker2d)`. This is a max-min (Chebyshev) scalarization, not
multi-objective optimization. There is no Pareto front, no NSGA-II, no dominance-based
selection. Keep multi-objective language (NSGA-II, Pareto, "two objectives") out of the
entire thesis — it would promise something the single-objective Results cannot deliver.
Call it **multi-task** evaluation, never multi-objective.

**The central research question:** Does heterogeneous activation evolution retain its
parsimony and performance advantages in continuous control tasks, where the fitness
landscape is noisier and the behavioral requirements more complex?

---

## 2. Terminology — Use These Consistently

| ✅ Correct | ❌ Avoid |
|---|---|
| Homogeneous network | Fixed-activation network, uniform network |
| Heterogeneous network | Mixed network, variable-activation network |
| Activation function co-evolution | Activation learning, activation training |
| Structural innovation | Topology change, architectural mutation |
| Speciation (NEAT-specific niching) | Clustering, grouping |
| Innovation number | Gene index, marker, ID |
| Parsimony / parsimonious | Compact, small (these are fine as secondary terms) |
| Fitness sharing (explicit) | Niche fitness, adjusted scoring |
| Compatibility distance (δ) | Speciation distance, genetic distance |
| Add-node mutation | Node insertion, topology expansion |
| Add-connection mutation | Edge insertion, link addition |
| HA-NEAT | Heterogeneous Activation NEAT (spell out on first use) |
| Continuous control | Locomotion task (acceptable variant) |

**Citation style:**
- First paper: "Stanley and Miikkulainen (2002)" or "(Stanley & Miikkulainen, 2002)"
- HA-NEAT paper: "Hagg et al. (2017)" or "(Hagg, Mensing & Asteroth, 2017)"
- Don't cite "NEAT 2002" without author names in body text

---

## 3. Section-Specific Guidance

### Abstract
Must contain:
1. What NEAT is (one sentence)
2. What HA-NEAT adds (activation co-evolution → parsimony)
3. What Sven's extension is (continuous control, Brax/JAX)
4. Primary finding (does parsimony hypothesis hold? performance comparison)

### Background / Related Work
Structure:
1. Neuroevolution context → NEAT (historical markings, speciation, minimal init)
2. Topology-and-Weight evolving ANNs (TWEANNs) briefly
3. HA-NEAT as extension of NEAT (Hagg 2017) — the parsimony motivation
4. Continuous control benchmarks — why Brax/Hopper/Walker2D
5. Gap: HA-NEAT not evaluated on continuous control → Sven's contribution

**Key distinction to make explicit in Background:**
HA-NEAT uses **direct encoding** (unlike HyperNEAT's indirect CPPN-based encoding).
This matters because: direct encoding keeps networks small and interpretable, suitable
for control problems where parsimony is a virtue, not just a side effect.

### Methodology
Frame as: "We extend HA-NEAT (Hagg et al., 2017) with the following modifications..."

Key methodological choices to justify:
- **Activation set choice**: Why these functions? (qualitative diversity, bounded output ranges, parsimony motivation from Hagg 2017)
- **Feedforward only**: Hagg 2017 restricted to feedforward for regression simplicity. For locomotion, recurrent connections may be relevant — address whether this choice is intentional or a limitation
- **Fitness aggregation**: Why scalarize the two tasks via `min(norm_hopper, norm_walker2d)` instead of, e.g., a sum? (max-min rewards the *weakest* task, forcing a single network to be competent on both rather than specializing in one; this is the multi-task requirement made concrete). Justify the per-task normalization constants.
- **Brax environments**: Why Hopper and Walker2D? (standard benchmarks, progressively difficult, JAX-native)

### Results
**Always report alongside:**
- Network size (nodes, connections) — parsimony is the primary claim
- Fitness score (normalized where applicable)
- Comparison: HA-NEAT vs homogeneous NEAT variants (per activation function)

**Framing parsimony results:**
> "Heterogeneous networks achieved comparable fitness to the best-performing homogeneous
> variant while using X% fewer nodes, consistent with Hagg et al. (2017)'s parsimony
> hypothesis."

OR if results don't replicate:
> "Unlike Hagg et al. (2017)'s regression findings, the parsimony advantage was not
> observed in [Hopper/Walker2D], suggesting that continuous control tasks may impose
> different structural requirements..."

**Never write** "HA-NEAT performed better" without specifying: better on what metric,
compared to which baseline, over how many runs.

### Discussion
Core discussion axes:
1. **Does the parsimony hypothesis transfer to continuous control?** (Yes/No/Partial — with mechanistic explanation)
2. **Which activation functions are selected?** (Does the distribution differ from regression tasks? What does this suggest about the control problem's structure?)
3. **Multi-task tradeoff**: Does the max-min aggregation produce networks balanced across both tasks, or do runs collapse onto one task at the other's expense? What does the per-task fitness breakdown show?
4. **Limitations**: Feedforward-only, activation set choice, evaluation budget, single random seed vs multiple runs

### Conclusion
Should answer the thesis question directly, then: what would be needed to go further
(recurrent networks, larger activation set, other Brax environments, larger population).

---

## 4. Common Writing Mistakes to Avoid

**Overclaiming novelty:**
❌ "HA-NEAT is a novel algorithm"
✅ "We apply HA-NEAT (Hagg et al., 2017) to continuous control, evaluating whether its parsimony advantages transfer to this domain"

**Conflating mechanisms:**
❌ "Speciation ensures smaller networks"
✅ "Speciation protects structural innovations by isolating them in niches, allowing newly added nodes/connections time to optimize before competing with the full population"

**Vague performance claims:**
❌ "HA-NEAT converged faster"
✅ "HA-NEAT reached threshold fitness in X fewer evaluations than homogeneous NEAT with sigmoid activation (the best-performing homogeneous variant)"

**Underselling the evaluation pipeline:**
The JAX/Brax + MLflow setup is infrastructure contribution — mention it briefly in Methodology as enabling reproducible, GPU-accelerated evaluation at scale.

---

## 5. Structural Template — Suggested Chapter Flow

```
1. Introduction
   - Motivation (parsimony in neural networks, NE advantages)
   - Research question
   - Thesis structure

2. Background
   - Neuroevolution & NEAT
   - HA-NEAT (Hagg 2017)
   - Continuous control benchmarks
   - Related work gap

3. Methodology
   - HA-NEAT implementation details
   - Extensions (multi-task fitness aggregation, activation set, Brax envs)
   - Experimental design

4. Results
   - Fitness comparison (HA-NEAT vs homogeneous variants)
   - Network size / parsimony analysis
   - Activation function distributions

5. Discussion
   - Parsimony hypothesis in continuous control
   - Activation function selection behavior
   - Limitations

6. Conclusion
   - Answers to research questions
   - Future work
```

---

## 6. MLflow / Experiment Integrity Notes (for Results section)

When citing MLflow results:
- Always specify: run ID or experiment name, number of seeds, evaluation episodes
- If max_fitness and per-environment normalized scores differ, report both and explain normalization
- Distinguish: "best run" vs "median over N runs" — use median for claims

---

## 7. Quick Checklist Before Submitting Any Section

- [ ] All NEAT mechanisms explained precisely (historical markings, speciation, minimal init)
- [ ] HA-NEAT clearly positioned as extension of Hagg 2017, not a new algorithm
- [ ] Sven's contribution (continuous control, multi-task scalarized fitness, JAX/Brax) explicitly stated
- [ ] No multi-objective / NSGA-II / Pareto language anywhere (the work is single-objective via `normalized_min`)
- [ ] Parsimony claim tied to concrete numbers (node/connection counts)
- [ ] Homogeneous baselines specified (which activation function(s))
- [ ] No vague performance claims without metric + baseline
- [ ] Direct encoding vs indirect (HyperNEAT) distinction made where relevant