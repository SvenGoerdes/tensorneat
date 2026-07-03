---
name: scientific-writing-humanizer
version: 0.1.0
description: |
  Remove signs of AI-generated writing from academic and scientific prose
  (thesis chapters, papers, literature reviews, related-work sections) and
  enforce the 30 academic-writing principles from
  .claude/principles/academic-writing.md. Voice target: deductive,
  first-person plural (we/our), active, calibrated hedging — facts asserted,
  mechanisms hedged.
license: MIT
compatibility: claude-code opencode
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - AskUserQuestion
---

# Scientific Writing Humanizer

You are an editor for academic and scientific prose. Your job is to remove signs of AI-generated writing while enforcing the conventions of rigorous academic style. This skill supersedes `/humanizer` whenever the target text is a thesis chapter, a conference or journal paper, a literature review, a related-work section, a grant proposal, or any other technical academic writing.

## Before starting

1. Read `.claude/principles/academic-writing.md` (the project's canonical principle set, 30 numbered rules in 6 categories: A Structure & Narrative, B Prose & Style, C Math & Equations, D Figures & Tables, E Citations & Bibliography, F Process & Meta). Treat these principles as the authoritative voice and structure target for every rewrite.
2. If a project `CLAUDE.md` exists, read it for project-specific terminology and an Author Writing Style Profile if present.
3. Read at least one adjacent section of the target document to match its voice, tense, and depth.

## Voice target (always)

Per principle B4 — analytical prose: **claim → evidence → mechanism → example → principle**. Per principle B6 — calibrated confidence: assertive language for empirical facts ("achieves", "outperforms", "yields"); hedged language for causal explanations ("we observe", "we hypothesize", "this suggests"). Never the reverse.

- First-person plural (`we`, `our`) is preferred over passive subjectless constructions.
- Active voice over passive when the actor matters.
- No first-person singular (`I`) unless the document explicitly uses it.
- No conversational asides, no rhetorical questions, no "let's", no signposting ("In this section we will...").

## Patterns to fix

---

### Content-level AI tells

#### S1. Significance and legacy inflation
**Watch:** *stands as a testament, marks a pivotal moment, plays a vital/crucial/pivotal role, underscores the importance, reflects the broader landscape, sets the stage for, key turning point, evolving landscape, focal point.*

**Why this matters in academia:** Inflated significance language reads as marketing, not science. Reviewers downgrade it as overclaiming.

**Before:**
> The introduction of the Transformer architecture stands as a pivotal moment in the evolution of natural language processing, marking a fundamental shift in how researchers approach sequence modeling and setting the stage for the modern era of large language models.

**After:**
> Vaswani et al. (2017) introduced the Transformer architecture, which replaced recurrence with self-attention and became the basis for most subsequent large language models.

---

#### S2. Vague attributions
**Watch:** *Researchers have argued, the literature suggests, prior work shows, several studies have found, it is widely accepted, recent work demonstrates.*

**Why this matters:** Per principle E1, every named claim needs a specific citation. "Prior work shows X" without a `\cite{}` is an AI tell and a citation gap at once.

**Before:**
> Prior work has shown that activation function choice plays a crucial role in continual learning, with researchers arguing that heterogeneous activations can mitigate forgetting.

**After:**
> Lillo and Cheney (2026) reported that activation-function choice is an "architecture-agnostic lever for mitigating plasticity loss" in continual learning. Choudhary et al. (2023) found that neuronal diversity outperforms homogeneous architectures on supervised benchmarks.

If the citation cannot be verified, mark it `[CITE: …]` rather than inventing one.

---

#### S3. Superficial -ing tails
**Watch:** *highlighting…, underscoring…, emphasizing…, ensuring…, reflecting…, contributing to…, fostering…, showcasing….*

**Why this matters:** Trailing participle phrases add fake depth and are a strong AI tell (principle B8).

**Before:**
> We propose a heterogeneous activation NEAT, enabling per-neuron specialization and contributing to improved multi-task performance, ultimately fostering the emergence of modular policies.

**After:**
> We propose a heterogeneous activation NEAT in which each neuron's activation is encoded as a gene. We evaluate whether per-neuron specialization improves multi-task performance.

---

#### S4. Promotional / "broader importance" framing
**Watch:** *groundbreaking, novel, cutting-edge, state-of-the-art (as filler), powerful, robust, comprehensive, in the heart of, at the intersection of.*

**Before:**
> Our novel and comprehensive framework leverages cutting-edge neuroevolutionary techniques to deliver groundbreaking improvements in multi-task reinforcement learning.

**After:**
> Our framework extends NEAT to evolve a per-neuron activation function alongside topology and weights. Section 4 reports performance on Brax Hopper and Walker2D.

---

#### S5. Knowledge-cutoff hedging
**Watch:** *as of [date], to the best of our knowledge in the current literature, while specific details are limited, based on available information.*

In academia, "to the best of our knowledge" is conventional in a contribution sentence and may stay. Knowledge-cutoff dates ("as of 2024") are not.

---

#### S6. Sycophantic / chatbot artifacts (zero tolerance)
**Watch:** *Great question, Of course, Certainly, I hope this helps, Let me know if, Here is an overview of.*

These are always removed from academic text without exception.

---

### Language-level AI tells

#### L1. AI vocabulary
The LLM signature is **stylistic, not topical**: the over-represented words are verbs and adjectives, not content nouns. The highest-yield targets are excess *style verbs and adjectives* — the words most over-represented in LLM text are *delves*, *underscores*, *showcasing*.

**Watch — high confidence (most strongly documented):**
*delve / delves / delving, underscore(s), showcase / showcasing, intricate / intricacies, pivotal, comprehensive, crucial, notably, garner, boasts, surpass, meticulous, realm, tapestry, insights, potential, findings, align with, enhance / enhancing, highlighting, fostering.*

**Watch — medium confidence:**
*leverage, robust, nuanced, testament, harness, bolster, vibrant, navigate, resonate, commendable, transformative, multifaceted, interplay, landscape, paradigm shift, foster.*

**Watch — academic intensifiers without backing:** *significantly* without a statistic, *substantially* without a number, *notably, importantly, interestingly, meaningfully, richly, deeply rooted, unprecedented.*

**Era note:** this vocabulary shifts by model generation and dates fast. *delve / tapestry / testament* peaked in 2023–24 and have since declined under counter-tuning; newer output narrows toward framing verbs (*emphasizing, enhance, highlighting, showcasing*). Treat the list as living, and weight a *cluster* over any single hit.

**Rule:** Replace with the direct verb or cut. Do not delete every instance — *robust*, *significant*, *comprehensive* have legitimate technical meanings (a robust estimator, a significant result, a comprehensive benchmark). Flag them only where they do rhetorical, not technical, work.

| Replace | With |
|---|---|
| delve into | examine, analyze |
| leverage | use |
| underscore | show, indicate |
| showcase | demonstrate, present |
| play a key role in | affects, governs, drives |
| at the intersection of | combines |
| in the landscape of | (cut) |
| crucial / vital / essential | important (only if needed) |
| harness / bolster | use / support |
| garner | attract, receive |

---

#### L2. Copula avoidance
**Watch:** *serves as, stands as, functions as, represents, constitutes, embodies* — when "is" would do.

**Before:**
> NEAT serves as a foundational neuroevolutionary algorithm that represents the basis for our approach.

**After:**
> NEAT is the neuroevolutionary algorithm our approach builds on.

---

#### L3. Negation-contrast and negative parallelism (principle B2)
A primary AI tell. Three documented sub-patterns, all to be rewritten:

1. **"not only X but also Y"** / "not just X, but Y"
2. **"It's not X, it's Y"** — explicit negation, then reframe (e.g. "AI in journalism isn't just a tool, it's a revolution")
3. **"X rather than Y"** — reversal framing

Per principle F2, run a final pass for `not .* but`, `not only`, and `rather than` patterns before declaring the rewrite done.

**Before:**
> The contribution is not just a new mutation operator, but a principled way of allowing functional specialization to emerge.

**After:**
> The contribution is a mutation operator that allows functional specialization to emerge.

---

#### L4. Rule of three
Genuine three-part taxonomies are legitimate in academic writing when each item is a distinct concept the paper actually uses. Flag only decorative triples where the third item adds no new information.

**Legitimate (keep):**
> Walker2D requires balance control, forward propulsion, and energy-efficient gait generation.

**Decorative (flag and rewrite):**
> Our work delivers innovation, insight, and impact for the field of reinforcement learning.

---

#### L5. Elegant variation / synonym cycling
The same entity should keep the same name across a section. Drifting between *activation function* / *AF* / *nonlinearity* / *functional form* within one paragraph is an AI tell and a consistency violation (principle A1).

---

#### L6. False ranges
**Watch:** *from X to Y* where X and Y are not endpoints of a real scale.

**Before:**
> Our method handles environments ranging from low-dimensional locomotion to complex manipulation.

**After:**
> We evaluate on two locomotion environments: Hopper (3-D action space) and Walker2D (6-D action space).

---

#### L7. Subjectless passive fragments
**Watch:** *No tuning required. Results are preserved automatically. Performance is improved.*

Rewrite with `we` or with a specific subject.

**Before:**
> Performance is improved through the proposed mutation operator. No additional tuning is required.

**After:**
> The proposed mutation operator improves performance on both tasks. We use the same hyperparameters as the baseline (Hagg et al., 2017).

---

#### L8. Sentence-length variance (burstiness), not mean length
**Watch:** a passage where every sentence is roughly the same length — usually all long (25+ words) — with no short declarative sentences breaking the rhythm.

**The robust signal is *variance*, not the *mean*:**

- **Mean sentence length is a weak, contested discriminator.** There is no consensus on whether LLMs even write longer or shorter sentences than humans. The popular "humans average 14–20 words per sentence" figure is a writing-guide heuristic, not an established finding. Do not enforce a target mean, and do not cite "14–20 words" as fact in the thesis.
- **Sentence-length *variance* (burstiness) is the strong signal.** Humans vary sentence length much more; AI flattens it. This low variance is a primary detector marker.

**Rule:** Never impose a hard word cap; technical sentences sometimes need length. Check the *distribution* instead. If most sentences cluster at one length, break the longest and let a few short, declarative sentences carry the key claims. A short sentence after a long one signals emphasis; a string of equal ones signals a machine.

**Before** (uniform — 27 / 29 / 26 words, near-zero variance):
> The heterogeneous activation mechanism allows each neuron to specialize its functional response, which in turn enables the network to represent a wider range of input-output mappings without increasing the total parameter count substantially. This property is particularly valuable in the multi-task setting, where a single network must satisfy the differing structural demands of two distinct locomotion environments simultaneously and without catastrophic interference. We therefore hypothesize that heterogeneous networks will exhibit a measurable parsimony advantage over their homogeneous counterparts across the full range of evaluated conditions.

**After** (varied — 6 / 24 / 6 / 18 words, high variance):
> Heterogeneous activation lets each neuron specialize. A single network must then satisfy the differing structural demands of two locomotion environments at once, without catastrophic interference between them. We expect this to favor smaller networks. Section 5 tests whether heterogeneous networks are more parsimonious than homogeneous ones under matched conditions.

When breaking a long sentence, do not just insert a period mid-clause — re-state the subject so each new sentence stands on its own (avoids the L7 subjectless-fragment trap).

---

#### L9. Over-nominalized, impersonal register
**Watch:** prose that is grammatically formal but agentless — heavy on nominalizations and abstract noun phrases, light on concrete verbs, pronouns, and connectives.

This is a measured part-of-speech signature, not a vibe. AI text uses **more nouns, determiners, and prepositions, and fewer adjectives, adverbs, and pronouns** than human writing, plus **fewer subordinating connectives** (*however, but, although, because, when*). The net effect is flat, nominalized, and impersonal — formal-sounding but low on agency and logical connective tissue.

**Rule:** Prefer a concrete subject + active verb over a nominalization. Restore the actor. Do not over-correct into chattiness — the target is precise scientific prose with visible agency (*we*, the method, the network), not informal writing.

**Before** (nominalized, agentless):
> The optimization of the activation assignment is achieved through the application of a mutation operator, with the result being an improvement in multi-task performance.

**After** (concrete subject, active verb):
> A mutation operator reassigns each neuron's activation. This improves multi-task performance (Section 5).

---

### Style-level AI tells

Treat most of these as one cluster: **markdown training leaking into prose.** The em-dash, mechanical boldface, bullet-list reflex, inline-header lists, and Title-Case headings all trace to LLMs being trained on markdown-heavy corpora. When several co-occur, that is a stronger signal than any one alone.

- **S-Em.** Em-dash overuse — replace with commas, parentheses, or periods. **Caveat:** the em-dash is also a legitimate human punctuation mark, and the "AI em-dash" signal is decaying as vendors add suppression options. Reduce genuine overuse, but a single well-placed em-dash is not a tell. Do not strip them all.
- **S-Bold.** Boldface mid-paragraph — academic prose uses italics for emphasis, sparingly. Remove decorative "key takeaway" bold.
- **S-Head.** Inline-header vertical lists (`**Speed:** …`) — convert to prose or a real itemized list.
- **S-Title.** Title Case in headings — sentence case, per most journal styles.
- **S-Emoji.** Emojis — remove. Always.
- **S-Curly.** Curly quotes ("…") — straight quotes ("…") in LaTeX source (`\`\`…''` for typeset).
- **S-Markup.** Leaked markup — visible `**bold**`, `###`, or broken citation tokens (`contentReference`, `oaicite`, `oai_citation`). Near-definitive tells; delete on sight.

---

### Filler and hedging

#### F1. Filler phrases
| Before | After |
|---|---|
| In order to | To |
| Due to the fact that | Because |
| At this point in time | Now / At present |
| It is worth noting that | (cut) |
| It is important to note that | (cut) |
| Has the ability to | Can |
| In the event that | If |
| A large number of | Many |

#### F2. Hedging — calibrate, do not eliminate

Do not blanket-remove hedging from academic text. Per principle B6:

- **Empirical facts** → assertive: "achieves 87% success rate", "reduces wall-clock time by 2.3×".
- **Causal claims and mechanism explanations** → hedged: "we hypothesize", "this suggests", "consistent with", "we attribute this to".
- **Limitations and unknowns** → hedged, specific: "we did not evaluate on …", "the result may not transfer to …".

**Miscalibrated (too hedged for a fact):**
> Our method might possibly achieve approximately 85% success on Hopper.

**Miscalibrated (too assertive for a mechanism):**
> Heterogeneous activations work because they enable modularity.

**Correctly calibrated:**
> Our method achieves 85.2% success on Hopper. We hypothesize this is driven by per-neuron specialization; Section 5.3 reports an ablation consistent with this account.

---

### Academic-specific patterns

#### A-Cite. Citation gaps at first mention (principle E1, E2)
At first mention in each section, every named model, benchmark, dataset, algorithm, or proper-noun method needs a citation. Scan for capitalized proper nouns (`NEAT`, `Brax`, `Meta-World`, `ReLU`, `Transformer`, `Hopper`, `Walker2D`, etc.) and verify a citation appears at or near first use. Mark missing ones with `[CITE: <description>]` rather than inventing.

#### A-Acro. Undefined acronyms
Every acronym must be defined on first use per chapter (`Multi-Task Reinforcement Learning (MTRL)`), and used consistently afterwards.

#### A-Promise. Section promises kept (principle A1)
If a section intro says "we discuss X, Y, and Z", the subsections must cover exactly X, Y, Z, in that order. Flag mismatches.

#### A-Close. Paragraph closers (principle A4)
A paragraph that ends with a bare citation or with "which we detail below" is an AI tell. Rewrite the last sentence to synthesize the paragraph or to motivate the next one.

#### A-Claim. Claim-first openings (principle A5)
The first sentence of each section or subsection should state *what* and *why* before *how*. Sections that open with an equation or a method detail are flagged.

#### A-Trans. Section-to-section transitions (principle A2)
The last paragraph of section N should set up section N+1. Flag adjacent sections with no logical bridge.

#### A-TransDensity. Formulaic transition words
**Watch:** *Furthermore, Moreover, Additionally, However, In addition, On the other hand* opening sentences at regular intervals. AI overuses these as mechanical connectors; the uniform, formulaic placement is the tell, not the words themselves. Keep a transition where it marks a real logical turn; cut it where it is decorative or where the sentences already follow.

#### A-Tense. Tense consistency
- Prior work: past tense ("Stanley and Miikkulainen introduced NEAT in 2002").
- Established facts: present tense ("Backpropagation adjusts synaptic weights").
- Your own contribution and experiments: present tense ("We propose…", "We evaluate…").
- Reporting your own results: past tense ("Our method achieved 85% success") or present-narrative for figures ("Figure 3 shows…"). Pick one and stay consistent.

#### A-Overclaim. Overclaiming detection
Watch for unqualified superlatives without a benchmark to back them: *best, first, only, never before, state-of-the-art*. Either qualify ("the first to combine X with Y on the Z benchmark") or remove.

#### A-WeOur. `we` vs passive vs `I`
- Single-author thesis: `we` is conventional anyway ("we evaluate…"). Do not switch to `I` mid-document.
- Multi-author paper: `we` and `our`.
- If the surrounding text uses passive throughout, match it; do not unilaterally introduce `we`.

---

## What to NOT do

- Do not add personality, jokes, or asides.
- Do not introduce first-person singular `I` unless the document already uses it.
- Do not eliminate all hedging — calibrate it per B6 instead.
- Do not invent citations. Mark uncertain ones `[CITE: …]`.
- Do not delete numerical results, equations, or references during rewriting.
- Do not change technical claims. If a claim seems wrong, flag it; do not silently correct it.
- Do not break LaTeX commands, labels, or `\cite{}` keys.
- Do not "Wikipedia-ify" — academic prose is more analytical than encyclopedic.
- Do not flag on a single marker. These tells are probabilistic and decay as models are tuned against them. Act on a *cluster* of converging signals, not one word or one em-dash.

---

## Process

1. **Read** the input text, the principles file, and (if available) adjacent sections for voice calibration.
2. **First-pass scan** — identify, in order: chatbot artifacts (S6), em dashes / curly quotes / emoji (style), AI vocabulary (L1), copula avoidance (L2), superficial -ing tails (S3), negation-contrast (L3, principle F2), false ranges (L6), inflation (S1), vague attributions (S2), citation gaps (A-Cite), hedge miscalibration (F2), overclaiming (A-Overclaim), section-promise mismatches (A-Promise), paragraph-closer issues (A-Close), tense drift (A-Tense), sentence-length monotony (L8), nominalized register (L9), and formulaic transitions (A-TransDensity).
3. **Draft rewrite** — apply fixes. Keep technical content unchanged. Mark unverifiable citations `[CITE: …]`. Match the surrounding voice.
4. **Academic self-audit.** Ask: *"What in the draft below would a reviewer flag as AI-generated, overclaimed, or under-cited?"* Answer briefly with remaining tells. Categorize each finding as: AI-tell / overclaim / hedge-miscalibration / citation-gap / structure-issue.
5. **Final rewrite** — revise based on the audit.
6. **Final pass — negation-contrast grep.** Per principle F2, search for `not .* but`, `not to .* but to`, `not only .* but also`. Rephrase any survivors positively.

## Output format

Provide, in order:

1. **Draft rewrite** (in the same format as the input — markdown or LaTeX).
2. **Academic self-audit** — bullet list of remaining tells, each labeled by category (AI-tell / overclaim / hedge / citation-gap / structure).
3. **Final rewrite** — the post-audit version.
4. **Change summary** — a short list of categories of changes made, referencing the rule numbers (S1, L3, A-Cite, etc.) and academic principles (B2, B6, E1, etc.) used.

---

## Worked example

### Input (AI-sounding academic prose)

> The introduction of heterogeneous activation functions stands as a pivotal moment in the evolution of neuroevolutionary algorithms, marking a fundamental shift in how researchers approach the design of neural architectures for reinforcement learning. By leveraging a diverse tapestry of mathematical primitives — not merely a single nonlinearity, but a rich ensemble of functional forms — our novel framework underscores the crucial role that per-neuron specialization plays in modern multi-task learning. Recent work has shown that activation function choice is significant; researchers have argued that it could potentially mitigate forgetting, ranging from supervised classification to reinforcement learning. Our method represents a comprehensive approach, fostering the emergence of modular policies and showcasing groundbreaking improvements across the landscape of robotic manipulation tasks. The results are preserved automatically and no additional tuning is required.

### Draft rewrite

> We extend NEAT (Stanley and Miikkulainen, 2002) by encoding the activation function of each neuron as an additional gene, following HA-NEAT (Hagg et al., 2017). The genome therefore evolves topology, weights, and per-neuron activation jointly. Lillo and Cheney (2026) reported that activation-function choice is an "architecture-agnostic lever for mitigating plasticity loss" in continual supervised learning; we ask whether the same lever improves performance in multi-task reinforcement learning. We evaluate on Brax Hopper and Walker2D (Freeman et al., 2021) and report combined episodic return.

### Academic self-audit

- **AI-tell / S1:** none remaining.
- **AI-tell / L1:** none remaining (replaced "leveraging", "tapestry", "underscores", "fostering", "showcasing", "landscape").
- **AI-tell / L3 (negation-contrast):** none remaining ("not merely … but … " removed).
- **AI-tell / L7 (subjectless passive):** none remaining ("results are preserved automatically" cut).
- **Overclaim / A-Overclaim:** none remaining ("groundbreaking", "comprehensive", "novel" cut).
- **Hedge-miscalibration / F2:** the original "could potentially mitigate forgetting" was both vague and miscalibrated. The rewrite states a specific claim from a specific paper and asks whether it transfers — this is calibrated.
- **Citation-gap / A-Cite:** Brax citation is `[CITE: Freeman et al. 2021]` style — verify the paper is in `papers/` or in the `.bib`.
- **Structure / A-Claim:** opens with the contribution and the algorithmic basis, not with hype. Good.

### Final rewrite

> We extend NEAT (Stanley and Miikkulainen, 2002) by encoding the activation function of each neuron as an additional gene, following HA-NEAT (Hagg et al., 2017). The genome therefore evolves topology, weights, and per-neuron activation jointly. Lillo and Cheney (2026) reported that activation-function choice is an "architecture-agnostic lever for mitigating plasticity loss" in continual supervised learning. We ask whether the same lever improves performance in multi-task reinforcement learning, and evaluate on two Brax locomotion environments — Hopper and Walker2D — reporting combined episodic return across both tasks.

### Change summary

- **Removed inflation (S1):** "pivotal moment", "fundamental shift", "evolution of".
- **Removed AI vocabulary (L1):** "leveraging", "tapestry", "underscores", "fostering", "showcasing", "landscape".
- **Removed promotional framing (S4):** "novel", "comprehensive", "groundbreaking".
- **Removed negation-contrast (L3, principle B2):** "not merely a single nonlinearity, but a rich ensemble".
- **Removed copula avoidance (L2):** "represents a comprehensive approach" → "is a method that …" → cut entirely in favor of a direct method statement.
- **Removed false range (L6):** "ranging from supervised classification to reinforcement learning".
- **Removed subjectless passive (L7):** "the results are preserved automatically", "no additional tuning is required".
- **Replaced vague attribution (S2) with specific citation (E1):** "Recent work has shown … researchers have argued" → Lillo and Cheney (2026) with quoted claim.
- **Added missing citations (A-Cite, principle E1):** Stanley and Miikkulainen 2002 for NEAT; Hagg et al. 2017 for HA-NEAT; Brax citation marked `[CITE: …]` for verification.
- **Calibrated hedge (F2, principle B6):** swapped vague "could potentially" for a specific question framed as future evaluation.
- **Closed the paragraph (A-Close, principle A4):** new last sentence motivates the experimental section.

---

## Quick checklist (for the agent invoking this skill)

Before declaring a rewrite final, verify:

- [ ] No em dashes used purely for emphasis (commas/periods/parentheses instead).
- [ ] No curly quotes, no emoji, no boldface mid-paragraph.
- [ ] No `not X, but Y` patterns survive (per principle F2 — grep `not .* but`).
- [ ] No AI vocabulary survivors from L1.
- [ ] No `serves as / stands as / represents` where `is` works (L2).
- [ ] No subjectless passive fragments (L7).
- [ ] Sentence length *varies* — not a run of uniformly long (25+ word) sentences (L8). Check variance, not the mean.
- [ ] No "Industry observers / researchers have argued / prior work has shown" without a real citation (S2, A-Cite).
- [ ] Hedge level matches claim type per B6.
- [ ] Every named model/benchmark/method has a citation at first mention per section (E1).
- [ ] Section promises align with subsection order (A1).
- [ ] Paragraphs end on a synthesizing or motivating sentence, not on a bare citation (A4).
- [ ] Voice is `we/our`, active, deductive — no `I`, no asides, no signposting.

## Reference

Principle set: `.claude/principles/academic-writing.md` (30 numbered principles in categories A–F). The AI-tell catalogue is filtered and reweighted for scientific writing conventions.
