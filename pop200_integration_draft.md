# pop200 integration draft — three exploratory findings

Ready-to-paste LaTeX for integrating the pop200 post-hoc analyses (network complexity,
Hopper plateau / multi-task pressure, activation usage) into
`/Users/svengoerdes/Projects/Thesisv2/text/main.tex`.

**Framing constraint honored throughout:** all three findings are labelled explicitly as
*exploratory / post-hoc*, run only after the pre-registered fitness comparison returned a null
result (Kruskal–Wallis p = 0.476). No causal claim; the marker-reassignment account is flagged
as an untested mechanism. No new numbers appear in the Discussion blocks.

---

## 0. Important structural note before you paste anything

The pop200 experiment (pop 200, 750 generations, 30 seeds × 3 conditions, and the headline
`p = 0.476` null result) is **not yet written into `main.tex`**. The current
`\section{Results and Discussion}` (starts line 952) still describes the *older* experiment line
(pop 1024–4096, 500–2000 generations, 2–5 seeds, `\subsection{Multi-task performance comparison}`
at line 1069 ends with `[PLACEHOLDER]` cells for the "final" experiment). There is also no
standalone `\subsection{Discussion}` — Results and Discussion are fused, and interpretation is
currently mixed into the Results prose.

Consequence for integration: the three new subsections below **depend on a pop200 setup paragraph
and the `p = 0.476` fitness result existing first.** Two options:

- **(Recommended)** Add a short pop200 anchor subsection (setup + the fitness null result) right
  after `\subsection{Experimental overview}` (line 954–964), then append the three Results
  subsections below it, and add one consolidated `\subsection{Discussion}` at the end (before line
  1103 `%\section{Conclusions and Future Work}`). A stub for the anchor is provided in §1.0.
- If you already have a pop200 fitness subsection drafted elsewhere, drop the three Results
  subsections directly after it and merge the Discussion block into your existing discussion.

Every `\ref{}` below assumes the labels defined in this file. The `p = 0.476` figure is
referenced as `\ref{...}`-free prose; wire it to your real label once the anchor exists.

---

## 1. Integration map — where each finding goes

| # | Finding | Target anchor in `main.tex` | Action |
|---|---|---|---|
| 0 | pop200 setup + fitness null (dependency) | After `\subsection{Experimental overview}` — anchor line 964 (`Random seeds were drawn from ...`) | **New** `\subsection{The pop200 study}` (stub in §1.0) |
| 1 | Network complexity of champions | Immediately after the pop200 fitness result; conceptually extends `\subsection{Multi-task performance comparison}` (line 1069) and the complexity remit named in `\subsection{Experimental overview}` line 962 (`Section 5.5 covers species dynamics and genome complexity`) | **New** `\subsection{Genome complexity and generational dynamics}` (§2.1) |
| 2 | Hopper plateau / multi-task pressure | After finding 1; extends the per-task remit named line 962 (`Section 5.3 breaks it down per task`) and the caveat at line 1096 (`whether both tasks improved or whether one task carried the other`) | **New** `\subsection{Per-task trade-off and multi-task pressure}` (§2.2) |
| 3 | Activation usage within HA-NEAT | After finding 2; directly tests the mechanism defined line 967 (`draws per-node activations from a set of five functions`) | **New** `\subsection{Activation-function usage within HA-NEAT}` (§2.3) |
| D | Interpretation of 1–3 + limitations + future work | End of section, before line 1103 `%\section{Conclusions and Future Work}` | **New** `\subsection{Discussion of the pop200 analyses}` (§3) |

Quoted anchor lines:

- Line 962: `The chapter works through the results in layers ... species and complexity dynamics, and the ablation study on activation diversity.`
- Line 964: `Random seeds were drawn from $\{42, 123, 75, 7, 21\}$; preliminary results use subsets of 2--3 seeds, and the final experiment uses all 5.`  ← insert §1.0 anchor after this line.
- Line 1069: `\subsection{Multi-task performance comparison}` ← the complexity subsection is the natural continuation.
- Line 1096: `The aggregate number alone does not tell us whether both tasks improved or whether one task carried the other.` ← finding 2 answers exactly this.
- Line 1103: `%\section{Conclusions and Future Work}` ← Discussion block goes just above this.

---

## 1.0 Dependency stub — pop200 setup + fitness null (paste first, adapt to your real wording)

> Only paste this if a pop200 setup + fitness subsection does not already exist. If it does,
> skip this and just reuse its `\label`s in the blocks below.

```latex
\subsection{The pop200 study}
\label{sec:pop200}
The experiments in Sections~\ref{sec:pop200} onward use a fixed, larger-sample design:
population 200, 750 generations, and 30 independent random seeds per condition. Three conditions
were compared: standard NEAT (tanh only), HA-NEAT (activation set tanh, sigmoid, relu, sin,
identity), and an ablation that runs HA-NEAT's full machinery restricted to tanh. All runs share
the multi-task Hopper$+$Walker2D problem and the normalized-minimum objective
$\mathrm{fit} = \min(r_{\text{hopper}}/3000,\, r_{\text{walker2d}}/5000)$.

Across the three conditions, the best-found normalized-minimum fitness showed no significant
difference (Kruskal--Wallis $H = [\,\ldots\,]$, $p = 0.476$). The analyses that follow were
therefore run \emph{after} this null result, to ask a different question: whether the conditions
that reach equal fitness do so with equivalent internal structure and task balance. These are
exploratory, post-hoc analyses and are framed as such throughout.
\label{res:pop200-fitness}
```

---

## 2. Results subsections (ready to paste)

### 2.1 Genome complexity and generational dynamics (Finding 1)

```latex
\subsection{Genome complexity and generational dynamics}
\label{sec:pop200-complexity}

Because the three conditions reached statistically indistinguishable fitness
(Section~\ref{res:pop200-fitness}), we examined whether they arrived there with equivalent genome
structure, testing the possibility that activation diversity substitutes for topological
complexity. For each of the 90 champion genomes (30 per condition) we counted hidden nodes and
active connections from the raw genome arrays; the 17 input and 6 output nodes are fixed by the
task and excluded from the hidden-node count.

The number of hidden nodes in the final champions differed across conditions
(Figure~\ref{fig:pop200-hidden}). NEAT champions carried a median of 2.0 hidden nodes
(IQR $[1.0, 3.0]$), HA-NEAT 1.0 ($[0.0, 2.0]$), and the ablation 0.0 ($[0.0, 2.0]$); a
Kruskal--Wallis test rejected equality of the three distributions ($H = 7.45$, $p = 0.024$,
$\varepsilon^2 = 0.063$). Holm-corrected pairwise Mann--Whitney tests localized the effect to a
single contrast: NEAT versus the ablation ($p = 0.030$, Cliff's $\delta = 0.38$, medium),
while NEAT versus HA-NEAT ($p = 0.10$) and HA-NEAT versus the ablation ($p = 0.52$) were not
significant. Connection counts did not differ across conditions (medians 48.5 / 51.5 / 47.0;
$H = 2.82$, $p = 0.244$; Figure~\ref{fig:pop200-conns}).

The per-generation trajectories from the training logs locate when this gap opens
(Figure~\ref{fig:pop200-complexity-gens}). Best-genome node counts overlapped across all three
conditions until roughly generation 300, after which NEAT rose above both HA-NEAT variants,
reaching a final-window mean of 26.6 total nodes (including the 23 fixed input/output nodes)
against 24.7 (HA-NEAT) and 24.0 (ablation). Connection counts pruned in lockstep across all three
conditions, falling from roughly 100 to 40--44 by the final window. Species counts saturated near
the same ceiling in every condition (final-window means 9.97 / 9.97 / 9.97; Kruskal--Wallis
$H = 0.11$, $p = 0.945$; Figure~\ref{fig:pop200-species}), consistent with the configured target of
$\approx 10$ species. Taken together, the conditions converged to equal fitness but not to
equal topology, with the ablation and HA-NEAT settling on more compact networks than NEAT.
```

Figure environments for finding 1:

```latex
\begin{figure}[h]
\centering
\begin{minipage}[t]{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{Figures/complexity_hidden_nodes.png}
\subcaption{Hidden nodes per champion.}
\label{fig:pop200-hidden}
\end{minipage}\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{Figures/complexity_connections.png}
\subcaption{Active connections per champion.}
\label{fig:pop200-conns}
\end{minipage}
\caption{Final-champion complexity in the pop200 study ($n=30$ per condition). Hidden-node
distributions differ across conditions (Kruskal--Wallis $p = 0.024$), driven by the NEAT vs.\
ablation contrast; connection counts do not ($p = 0.244$).}
\label{fig:pop200-complexity}
\end{figure}

\begin{figure}[h]
\centering
\begin{minipage}[t]{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{Figures/complexity_over_generations.png}
\subcaption{Best-genome node and connection counts over generations.}
\label{fig:pop200-complexity-gens}
\end{minipage}\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{Figures/species_over_generations.png}
\subcaption{Species count over generations.}
\label{fig:pop200-species}
\end{minipage}
\caption{Generational dynamics in the pop200 study. Lines are across-seed means; bands are the
inter-quartile range. Node counts diverge only after $\approx$ generation 300; connection counts
prune in lockstep; species counts saturate near the configured ceiling in all conditions.}
\label{fig:pop200-generational}
\FloatBarrier
\end{figure}
```

> Note: `\subcaption` requires `\usepackage{subcaption}`. The existing thesis uses a
> `\begin{minipage}...\textit{(a) ...}` pattern instead (see lines 989–998). If you prefer to
> avoid a new package, replace each `\subcaption{...}\label{...}` with the italic caption style
> and move the sub-labels into the main `\caption`; see §4 checklist item 3.

---

### 2.2 Per-task trade-off and multi-task pressure (Finding 2)

```latex
\subsection{Per-task trade-off and multi-task pressure}
\label{sec:pop200-tradeoff}

The aggregate fitness leaves open whether both tasks improved together or one task set the score
throughout; to resolve this we plotted each champion's normalized Hopper score against its
normalized Walker2D score and compared the per-task balance across conditions. Per-task scores are
the normalized 20-episode returns from re-evaluation; $\min$ of the two is the training objective.

Champions in all three conditions clustered in a narrow vertical band at normalized Hopper
$\approx 0.33$--$0.35$, while Walker2D spread across a wider range and set the minimum for the
large majority of genomes (Figure~\ref{fig:pop200-tradeoff}). Walker2D was the binding task for
97\% of NEAT champions, 83\% of HA-NEAT, and 90\% of the ablation. The per-genome imbalance
$|r_{\text{hopper}} - r_{\text{walker2d}}|$ did not differ across conditions (medians 0.159 /
0.117 / 0.142; Kruskal--Wallis $H = 3.39$, $p = 0.184$). The Hopper band sits at
$0.34 \times 3000 \approx 1000$ raw reward, which for this environment corresponds to roughly
1000 timesteps of the per-step healthy/alive bonus. Across every condition, then, champions solved
the multi-task problem with the same lopsided profile: a saturated Hopper score and a Walker2D
score that carried the aggregate.
```

Figure environment for finding 2:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.75\textwidth]{Figures/tradeoff_frontier.png}
\caption{Per-task trade-off frontier in the pop200 study ($n=30$ per condition). Each point is a
champion's normalized Hopper score (x) against its normalized Walker2D score (y). Champions cluster
in a vertical band at Hopper $\approx 0.33$--$0.35$ in all conditions; Walker2D is the binding task
for 83--97\% of champions. Per-genome imbalance does not differ across conditions
(Kruskal--Wallis $p = 0.184$).}
\label{fig:pop200-tradeoff}
\FloatBarrier
\end{figure}
```

---

### 2.3 Activation-function usage within HA-NEAT (Finding 3)

```latex
\subsection{Activation-function usage within HA-NEAT}
\label{sec:pop200-activation}

If activation diversity is the ingredient that HA-NEAT contributes, its champions should both use
the non-tanh activations and be rewarded for doing so; we examined the activation composition of
HA-NEAT's hidden nodes and its relationship to fitness. Of the 30 HA-NEAT champions, 17 evolved at
least one hidden node; the remaining 13 had none, leaving 55 hidden nodes across the informative
seeds.

The evolved activations were diverse and rarely defaulted to tanh
(Figure~\ref{fig:pop200-activation-dist}). Among the 55 hidden nodes, relu accounted for 27\%,
sigmoid 24\%, identity 22\%, sin 16\%, and tanh 11\%. Neither summary of diversity predicted
fitness across the 17 informative champions (Figure~\ref{fig:pop200-activation-fit}): activation
entropy showed no monotonic relationship with fitness (Spearman $\rho = -0.05$, $p = 0.84$), nor
did the fraction of non-tanh hidden nodes ($\rho = 0.27$, $p = 0.30$). With only 17 informative
genomes these correlation tests are underpowered, so this is an absence of evidence rather than
evidence of absence.
```

Figure environments for finding 3:

```latex
\begin{figure}[h]
\centering
\begin{minipage}[t]{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{Figures/haneat_activation_distribution.png}
\subcaption{Activation composition of the 55 HA-NEAT hidden nodes.}
\label{fig:pop200-activation-dist}
\end{minipage}\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{Figures/activation_entropy_vs_fitness.png}
\subcaption{Activation entropy vs.\ fitness ($n=17$).}
\label{fig:pop200-activation-fit}
\end{minipage}
\caption{Activation usage within HA-NEAT champions in the pop200 study. Diversity is genuinely
exploited (tanh is the least common activation at 11\%), but neither entropy nor non-tanh fraction
predicts fitness across the 17 champions with hidden nodes.}
\label{fig:pop200-activation}
\FloatBarrier
\end{figure}
```

---

## 3. Discussion block (ready to paste)

> No new numbers. Opens each subpoint with the conclusion. Labels everything exploratory.
> Flags the marker-reassignment mechanism as untested. Contains ≥3 limitations and concrete
> future work.

```latex
\subsection{Discussion of the pop200 analyses}
\label{sec:pop200-discussion}

Activation diversity was expected to raise multi-task fitness; it did not, but the HA-NEAT
machinery left a structural fingerprint that the fitness comparison alone would have missed. The
three analyses in this section were exploratory, run only after the pre-registered fitness
comparison returned a null result, and are interpreted here as hypothesis-generating rather than
confirmatory.

The clearest structural signal is that NEAT champions were topologically larger than the
ablation's, even though the two reached the same fitness. This is the opposite of a simple
"NEAT with one activation" expectation, and it points to the ablation not being standard NEAT at
all. The ablation runs the HA-NEAT mutation operator, and even with a tanh-only activation set the
activation-mutation event still fires and still reassigns the historical markers on every
connection of the chosen hidden node. The activation change is a functional no-op (tanh to tanh),
but the marker side effect persists: reassigned markers register as disjoint genes in the
compatibility distance and degrade crossover alignment for any genome carrying hidden nodes. A
plausible reading is that this suppresses the spread of hidden-node structures, which would explain
why the ablation patterns with HA-NEAT rather than with NEAT in the node counts, and would locate
the compactness effect in the HA-NEAT machinery rather than in activation diversity itself. We
stress that this mechanism is a hypothesis: it is consistent with the code path and with the
observed ordering, but it was not tested directly, and the observed effect is small and rests on a
single significant pairwise contrast.

Within HA-NEAT, the evolved networks did use the full activation set rather than collapsing back to
tanh, so the diversity mechanism is genuinely exercised. But no relationship between activation
diversity and fitness was detectable, which is consistent with the headline null: on this task the
lever HA-NEAT provides is pulled, and pulling it does not move fitness. The per-task analysis
suggests why the task may be insensitive to it. In every condition the champions saturated Hopper
at a low plateau and were bottlenecked by Walker2D, and the normalized-minimum objective makes
Hopper improvements beyond the plateau selectively invisible once Hopper already exceeds Walker2D.
The effective optimization problem degenerates toward "improve Walker2D subject to keeping Hopper
alive," which exerts far less genuine multi-task pressure than the two-task framing implies, and
leaves little room for any mechanism to distinguish itself.

These analyses have clear limitations. First, the hidden-node comparison rests on very small,
heavily tied counts (medians of 0--2 hidden nodes), which weakens the rank test and inflates the
influence of a few large genomes; the significant NEAT-versus-ablation contrast should be read as
suggestive. Second, the species-count comparison is uninformative because all conditions saturate
the configured species ceiling, so this design cannot test whether HA-NEAT's speciation protection
sustains more diversity. Third, the Hopper plateau confounds the multi-task claim: because Walker2D
is almost always the binding task, the study measures single-task Walker2D optimization more than
true multi-task pressure. Fourth, the activation-fitness correlations rest on only 17 informative
genomes and are underpowered. Fifth, all of these results come from a single population size (200),
so the structural effects may not persist at other scales.

Three concrete follow-ups would address these limitations directly. The marker-reassignment
hypothesis is testable with a targeted ablation that sets the activation-mutation rate to zero
while keeping the rest of the HA-NEAT machinery: if the compactness effect is caused by marker
reassignment, it should disappear. Restoring genuine multi-task pressure calls for either a harder
first task or a different aggregation that does not clip improvements on the leading task, such as
the normalized product, so that both tasks remain under selection throughout. Finally, repeating
the complexity and activation analyses at larger population sizes would test whether the structural
differences observed here are scale-dependent. None of these fit the remaining compute budget and
are left to future work.
```

---

## 4. Manual steps checklist (for the user)

1. **Copy the 7 figures** from
   `/Users/svengoerdes/Projects/Thesisv2/tensorneat/analysis/outputs/pop200/` into
   `/Users/svengoerdes/Projects/Thesisv2/text/Figures/`:
   `complexity_hidden_nodes.png`, `complexity_connections.png`,
   `complexity_over_generations.png`, `species_over_generations.png`,
   `tradeoff_frontier.png`, `haneat_activation_distribution.png`,
   `activation_entropy_vs_fitness.png`.
   (e.g. `cp analysis/outputs/pop200/{complexity_hidden_nodes,complexity_connections,complexity_over_generations,species_over_generations,tradeoff_frontier,haneat_activation_distribution,activation_entropy_vs_fitness}.png ../text/Figures/`)

2. **Add the pop200 anchor** (§1.0) after line 964, or wire the blocks' `\ref{res:pop200-fitness}`
   / `\ref{sec:pop200}` to your existing pop200 fitness subsection. Fill the Kruskal--Wallis
   $H = [\,\ldots\,]$ placeholder with the real statistic. This is the only hard dependency.

3. **Decide the subfigure style.** The blocks use `\subcaption` (needs
   `\usepackage{subcaption}` in the preamble). If you'd rather match the existing thesis pattern
   (`\textit{(a) ...}` inside minipages, lines 989–998), replace each
   `\subcaption{...}\label{...}` accordingly and move sub-labels into the main caption.

4. **No new citations are strictly required.** Optional: cite Brax (`\cite{freeman_brax_2021}`,
   already in the bib) at the first mention of Hopper's alive bonus in §2.2 if you want a source
   for the reward structure. The marker-reassignment mechanism is internal to this work and needs
   no citation.

5. **Cross-references and build.** After pasting, update the section roadmap in
   `\subsection{Experimental overview}` (lines 960–962) so the "Section 5.x" list points at the
   new subsections, then run a full `latexmk`/`biber` pass and check for undefined `\ref`s (the
   `res:pop200-fitness` / `sec:pop200` labels must resolve) and any duplicate `\label` clashes.
```
