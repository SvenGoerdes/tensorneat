# Chapter Review: Methodology, Experimental Settings, Results & Discussion

Review date: 2026-07-09. Source: `thesis/main.tex` (Methodology lines 716–936, Experimental Settings 938–1118, Results & Discussion 1237–1630). Feedback only — no edits applied.

---

## How each chapter type should work

**Methodology** explains *what the method is and why it is designed that way*. It owns the conceptual and technical contribution: problem formulation, the algorithm, each design decision with its rationale, and the implementation approach to the extent it affects the science. It should be parameter-free where possible (specific values belong in Experimental Settings, referenced by table) and justified throughout — every design decision defended, every deviation from the standard algorithm flagged. It should *not* contain: results, experiment logistics (logging inventories, hardware), or hyperparameter values that exist only to configure one study.

**Experimental Settings** is the reproducibility contract. Purely factual: what ran (environments, conditions, hyperparameters, seeds), how outcomes were measured, how they were analyzed, on what hardware. A reader should be able to re-run the study from this chapter plus the Methodology. The *why* belongs upstream; the *what happened* downstream. Pilot/calibration results that justify design choices are the one legitimate place results-like content appears here.

**Results and Discussion** has two jobs in sequence: report findings objectively (numbers, effect sizes, uncertainty, minimal interpretation), then interpret (mechanism, relation to prior work and the research question, limitations, future work). Every analysis promised in the Evaluation Protocol must appear; exploratory analyses must be labeled as such; the discussion must connect back to the literature and the thesis's research question.

Measured against these templates, all three chapters are structurally sound. The core architecture is right: the methodology defers NEAT background to the Literature Review and flags its deviations from the original algorithm; the settings chapter is factual with an honest calibration subsection; the primary result leads the Results chapter, exploratory analyses are labeled post-hoc, the mechanism account is hedged, and limitations are specific. The findings below are gaps, misplacements, and duplication — not restructuring.

---

## 1. Methodology (§3, lines 716–936)

> **Status 2026-07-10: all items resolved.** M1–M7 and M17 fixed by the author (restructure, MLflow table and backend story commented out, transfer sentence cut); M8–M16 applied in session, with the humanizer pass run over the new prose.

### Substantive

**M1. Factual error: new-node activation claim (line 798) — verified against code.**
> "In standard NEAT, a newly added node receives an identity activation function, making the structural mutation initially neutral."

This is wrong on both counts. In the codebase, `DefaultMutation` calls `DefaultNode.new_identity_attrs`, which assigns bias 0, response 1, and the **default activation** — the first of the configured options, i.e. **tanh** in the NEAT baseline, not identity (`src/tensorneat/genome/gene/node/default.py:89-95`). The method name refers to the identity-like bias/response, and a stale code comment ("activation=-1 means ACT.identity") appears to be the source of the confusion. In the original NEAT paper, approximate neutrality of node insertion comes from the connection weights (incoming 1.0, outgoing = old weight), not from the activation. The real contrast with HA-NEAT survives — HA-NEAT's `_new_node_attrs` override draws a **random** activation (`ha_neat.py:40-42`) — but the sentence needs correcting: standard NEAT assigns the default activation with neutral bias/response; HA-NEAT assigns a random one.

**M2. Content that doesn't fit: the MLflow metrics table (`tab:mlflow_metrics`, lines 912–936).**
A 14-row inventory of logging metric names is experiment logistics, not methodology, and it duplicates the per-generation tracking list in §4.5 (Evaluation Protocol, level 1) almost item for item. Move the table to Experimental Settings or an appendix; keep one sentence in Implementation Stack saying MLflow logs per-generation population and best-genome metrics.

**M3. Content placement: the `generalized`-backend floor exploit (lines 902–904).**
"Early runs used the generalized backend, but it permitted a physics exploit… All reported experiments therefore use MJX." This is a pilot-run finding used to justify a design decision — the same epistemic category as the aggregation and budget calibrations, which live in §4.4. Its placement inside "Implementation Stack" is inconsistent. Either move it to §4 (Environment Details or Calibration) with a cross-reference, or keep it here but explicitly frame it as a preliminary-run observation. Unifying where pilot evidence lives makes the thesis easier to defend.

**M4. Overclaim in Problem Formulation: "We pair the two tasks to test for cross-task transfer" (line 734).**
The design has no single-task baselines, so transfer is not measurable — there is nothing to compare the multi-task runs against. The study tests activation heterogeneity *under multi-task pressure*, which the very next sentences state correctly. Reword the transfer sentence toward "to create genuine multi-task pressure" / "a setting where transfer is possible in principle," or cut it. (A single-task-vs-multi-task condition existed in an early commented-out draft and was dropped; this sentence is a residue of that design.)

**M5. Broken promise: padded-dimension interpretability (line 833).**
"The interpretability of the padded dimensions remains an open question warranting examination in the analysis." No such analysis exists in the Results chapter. Cut the sentence or move the idea to future work in §5.6.

**M6. Duplication: the functional-role hypothesis is stated twice in full.**
The argument "periodic functions for gait timing, saturating functions for balance, linear/proportional for joint control; a homogeneous tanh network must approximate all of these, potentially at greater structural cost" appears in §3.2.1 Conceptual Motivation (lines 744–749) and again as the closing paragraph of §3.4 Network Architecture (line ~885). Say it once — Conceptual Motivation is the natural home — and have §3.4 reference it.

**M7. Structural: architecture content is split across two homes, and §3.4 arrives too late.**
The multi-task padding scheme and the array-level genome storage live in §3.2.4 (inside "Algorithm Design: HA-NEAT"), while initial topology, output activation, and the activation palette live in §3.4 — *after* Fitness Evaluation, which already talks about networks controlling environments. Two issues: (a) the padding scheme is a problem-interface decision, not an HA-NEAT algorithm detail — it applies identically to the NEAT baseline; (b) the NaN/array storage detail (lines 836–838) is implementation-stack material. Cleanest order: Problem Formulation → Network Architecture (incl. padding) → Algorithm Design → Fitness Evaluation → Implementation Stack. Minimal fix: move the array-storage paragraph to §3.5 and swap §3.3/§3.4.

### Minor

- **M8.** Hyperparameter value leakage: `$p_{\text{act}} = 0.1$` is hardcoded in §3.2.3 (line 800), while the chapter's own convention (used for θ, c_d/c_w, survival threshold) is to reference `tab:hyperparameters`. Write "with probability $p_{\text{act}}$ (Table~\ref{tab:hyperparameters})".
- **M9.** Line 726: typo "environnment" and a missing sentence break — "…the observation and action spaces of each environnment\nHopper exposes…" runs two sentences together.
- **M10.** Line 813: "architectural induction bias" → "inductive bias".
- **M11.** §3.4 activation list: stray closing parenthesis — "`\textbf{$\tanh$) and sigmoid ($\sigma$)}`".
- **M12.** Line 831: "The zero-padded input positions carry no reward signal from Hopper" — inputs don't carry reward; the intended meaning is that no selection pressure from Hopper acts through those inputs. Reword.
- **M13.** Line 834: "A rough representation of the network can be seen in Figure…" — hedgy phrasing, and the figure caption ("Network representation used in the multi-task setup.") is a single uninformative line. Captions should say what to look at (17 padded inputs, 6 sliced outputs, zero-block for Hopper).
- **M14.** Tense wobble in §3.3: the intro sentence is past ("every genome … was evaluated") while the pipeline items are present tense. Pick one.
- **M15.** Display math uses plain-TeX `$$…$$` throughout the chapter; `\[…\]` or `equation` is the LaTeX-correct form and matters if equation numbering is ever needed.
- **M16.** The marker pre-allocation paragraph (line 803, "$3 + C_{\max}$ markers…static shapes") is deep JAX implementation detail inside the algorithm description. Defensible because it explains a real design constraint, but it could be compressed to one sentence with the details in §3.5.
- **M17.** Commented-out legacy NEAT-foundation draft (lines 776–788) still in the file — Phase 6 cleanup.

### What works well
NEAT Foundation restates only what HA-NEAT builds on and defers the rest to the Literature Review; the distance formula documents TensorNEAT's merged disjoint/excess term as a deviation from original NEAT; the three HA-NEAT modifications are crisply enumerated with rationale each; the padding-over-modularity decision is explicitly justified; and the minimum-aggregation rationale is argued from principle with a forward pointer to the empirical calibration.

---

## 2. Experimental Settings (§4, lines 938–1118)

> **Status 2026-07-10: all items E1–E7 resolved.** E1 (Python 3.11 / JAX 0.9.1) fixed by the author; E2 (seed statement + [repository URL] placeholder), E3 (Conditions moved before Hyperparameters), E4 (DV list / protocol split), E5 (table parenthetical removed), E6 (48 GB), E7 (constants named in Environment Details) applied in session. The [repository URL] placeholder in the Evaluation Protocol still needs the real link before submission.

### Substantive

**E1. Internal contradiction in the compute table (should fix).**
The table lists "JAX 0.9.1" and "Python 3.10+". JAX 0.9.1 requires Python ≥ 3.11, so both rows cannot be true. Either the server ran ≥ 3.11 (write "Python 3.11", keep JAX 0.9.1) or it ran 3.10 (then JAX was 0.6.2 per the training-commit lockfile). "3.10+" is too vague for a reproducibility chapter regardless — pin one version. **Open: needs user confirmation of the server's Python version.**

**E2. Missing reproducibility anchors.**
The chapter never states the seed values (only the seed-5213 footnote hints they are arbitrary integers), and there is no code/data availability statement. One sentence — "the 30 seeds per condition, full configuration, and analysis code are available at [repository]" — is standard and closes the gap.

**E3. Ordering: Conditions after Hyperparameters is backwards.**
§4.2 parameterizes the experiment before §4.3 defines what is being compared; "identical across all 90 runs" claims can't be judged before the conditions exist. Conventional flow: environments → conditions/design → hyperparameters → calibration → protocol → compute. Swapping §4.2 and §4.3 is a clean fix.

**E4. Redundancy between §4.3's dependent-variable list and §4.5's measurement levels.**
Both answer "what do we measure"; per-generation tracking appears in both. Keep §4.3 as the conceptual DV list and make §4.5 purely about how/when measurement happens.

### Minor

- **E5.** "(early stopping never triggered)" appears in the hyperparameter table *and* the prose below it — result-flavored content sits better in prose only.
- **E6.** A6000 VRAM: spec is 48 GB (49 is nvidia-smi rounding); 48 reads more standard.
- **E7.** The Environment Details paragraph on Walker2D's larger reward scale could carry a `\ref` to the 3000/5000 constants in §3.3.

### What works well
The calibration subsection earns its place (pilot evidence justifying design decisions, single-seed caveat stated); the hyperparameter table is now ground-truthed against MLflow with the empirical θ framing; the statistical plan matches what the Results chapter actually does (KW → Holm-MWU → HL+CI → Cliff's δ → TOST, seed-level unit).

---

## 3. Results and Discussion (§5, lines 1237–1630)

### Substantive

**R1. A promised analysis never appears (the biggest gap in all three chapters).**
§4.5 commits to three measurement levels; level 3 is "Learning dynamics: fitness-over-generation curves, averaged across runs with 95% confidence bands." The Results chapter contains no such figure. §5.3 shows node/species counts over generations; the only per-condition fitness curves anywhere are panel (b) of the budget-calibration figure in §4.4, framed as calibration evidence. This is both a broken promise and a genuinely expected figure — a reader of a null result wants to see whether the three conditions' training curves overlap throughout or diverge and reconverge. Preferred fix: add the training-curve figure to §5 (data is in `mlflow.db`; plotting pattern exists in `text/figures/plot_training_curves.py`). It actively strengthens the null. Alternative: weaken the §4.5 promise.

**R2. The Discussion never returns to the literature or the research question.**
§5.6 is entirely internal — mechanism, limitations, follow-ups. It cites no prior work (except the Brax reward reference in §5.4), never revisits the §1.8 research gap, and never explicitly answers the thesis's research question. For a thesis this is the most conventional expectation of a discussion. Concretely: the original HA-NEAT work reported benefits on control tasks — the fitness null plus the compactness signal is a direct data point refining that claim in a multi-task setting; the activation-diversity literature in §1 made predictions this study tested. One or two connecting paragraphs would complete the chapter. (Comparison-with-prior-work belongs here even if the research-question verdict is repeated in the Conclusion.)

**R3. "Pre-registered" is an overclaim (line 1598).**
"The pre-registered fitness comparison" — unless a protocol was actually registered, this term has a specific meaning examiners know. "Pre-specified" or "primary" says what is meant.

**R4. Overview duplication with §4.**
The wall-clock sentence ("median of roughly 3.4 h (mean 3.8 h)… about 340 GPU-hours") appears verbatim in §4.6 and §5.1, and §5.1's conditions/budget recap restates §4.3 fairly fully. Orientation recap is good practice; verbatim duplication is not. Halve the §5.1 recap and cut the wall-clock sentence from §5.1 (it is settings, not results).

**R5. Confusing sentence in the overview (line 1251).**
"Smaller-scale preliminary runs (populations of a few thousand at two to three seeds)" — the populations were *larger* than the main study's 200; only seed counts were smaller. "Smaller-scale" contradicts its own parenthetical. Reword around seed count.

**R6. The Hopper interpretation could be one step more explicit.**
§5.4 establishes Hopper band ≈ 1000 raw reward ≈ the alive bonus alone; §5.6 builds "keep Hopper alive" on it. The unstated implication is that champions may barely locomote on Hopper — essentially surviving in place. If true, it is a vivid, honest characterization and checkable by rendering a few episodes. Either state it (hedged) or verify it; it would also sharpen the degeneration argument.

### Minor / mechanical (mostly queued Phase 6 work)

- **R7.** `\label{res:main-fitness}` sits mid-paragraph with no counter to attach to; `\ref` resolves to the subsection number — works, but fragile.
- **R8.** Minipage figures place `\label{fig:main-hidden}` etc. after `\textit` pseudo-captions with no real `\caption`, so those refs resolve to the wrong counter. `subcaption` is the fix.
- **R9.** ~170 lines of commented-out legacy results (1259–1430) still in the file.
- **R10.** Subsection title "Discussion of the main-study analyses" carried its qualifier from when the pop-1024 sections coexisted; plain "Discussion" is cleaner now.
- **R11.** `tab:main-fitness` reports normalized values only; a raw-reward column or one sentence of raw-scale context would help readers calibrated to Brax reward magnitudes.
- **R12.** Five limitations, three follow-ups: the follow-ups sentence claims they "address these limitations directly" but they map onto limitations 1, 3, and 5 only. A lighter claim would be exact.

### What works well
Primary result leads and is fully quantified (medians/IQR, HL shifts, Cliff's δ, all with CIs); the MDE and two-bound TOST framing is honest about what the null does and does not rule out; exploratory analyses are explicitly labeled post-hoc with a multiplicity caveat; the marker-reassignment mechanism is presented as a testable hypothesis with a concrete falsifying experiment; limitations are specific rather than boilerplate.

---

## Cross-chapter issues

- **X1.** Pilot evidence lives in two places: backend exploit in §3.5, aggregation + budget calibrations in §4.4. Pick one home (§4.4) — see M3.
- **X2.** "What we measure" is described three times: §3.5 MLflow table, §4.3 DV list, §4.5 measurement levels. One conceptual list (§4.3) + one protocol description (§4.5) suffices — see M2, E4.
- **X3.** §3.1's transfer framing (M4) is quietly corrected by §5.6's finding that the problem degenerates to Walker2D optimization — aligning the Problem Formulation's promise with what the study measures avoids an examiner catching the mismatch.
- **X4.** Display-math style (`$$…$$`), `\cite` vs `\parencite`, and commented-legacy cleanup are consistent chapter-wide issues — Phase 6.

## Priority summary

Fix-first (examiner-margin items):
1. **R1** — missing training-curves figure (promised in §4.5).
2. **R2** — Discussion has no literature connection or research-question verdict.
3. **M1** — factually wrong new-node activation claim (verified against code).
4. **E1** — JAX 0.9.1 / Python 3.10+ contradiction (needs user confirmation of server Python).
5. **M4** — "test for cross-task transfer" overclaim in Problem Formulation.

Second tier: M2/X2 (MLflow table placement), M7 (architecture-content split), R3 ("pre-registered"), R4/R5 (overview duplication and wording), E2 (seeds + code availability), E3 (section order).

Everything else is polish, and several items (R7–R10, M17, X4) are already queued for Phase 6.
