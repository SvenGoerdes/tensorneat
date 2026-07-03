---
name: wiki
description: Personal knowledge base for the thesis. Use /wiki to ingest raw thoughts (findings, hypotheses, paper notes, conversation excerpts) into structured pages at wiki/. Use /wiki query to ask what we know about a topic. Use /wiki lint to health-check for contradictions, orphans, and stale claims. The wiki compounds knowledge across sessions — it is the thinking layer between raw thought and finalized thesis prose.
---

# Wiki

## Overview

The wiki at `wiki/` is a persistent, interlinked collection of markdown files that Claude maintains. Sven provides raw thoughts; Claude classifies, structures, cross-references, and files them. Nothing is re-derived on every query — the synthesis is already there.

**Wiki structure:**
```
wiki/
├── README.md          # Human-facing intro
├── index.md           # Content catalog (read this first when querying)
├── log.md             # Append-only operation log
├── findings/          # Empirical observations from experiments
├── hypotheses/        # Conjectures and open questions
├── concepts/          # Reusable concept pages
├── papers/            # Literature notes (one per paper)
└── excerpts/          # Conversation snippets worth preserving
```

**Page conventions:**
- Filenames: `kebab-case.md`
- Links: `[[page-name]]` wikilink style (Obsidian-compatible)
- Every page starts with `# Title` then a one-line summary paragraph
- Pages stay < 200 lines; split when they grow
- No YAML frontmatter required (add it when a specific page benefits)
- The skill never modifies `text/`, `docs/`, or any other directory

---

## When to Use

| Trigger | Operation |
|---|---|
| Sven pastes a finding, hypothesis, paper note, or conversation excerpt | **ingest** |
| Sven asks "what do we know about X?" or "what have we found about Y?" | **query** |
| Sven says `/wiki lint` or requests a health-check | **lint** |

Sven may also say "file this to the wiki" or "add this to the wiki" — treat as ingest.

---

## Operation 1: Ingest

**Goal:** Take raw text Sven provides and integrate it into the wiki — updating or creating pages, linking them together, and keeping the index and log current.

### Steps

1. **Classify** the content:
   - `finding` — empirical observation from an experiment (data-backed)
   - `hypothesis` — conjecture, open question, untested idea
   - `concept` — a reusable term, mechanism, or design decision
   - `paper` — literature note for a specific paper
   - `excerpt` — conversation or meeting snippet

2. **Extract** 1–3 atomic claims or ideas from the raw input. Discard noise; keep the essential insight.

3. **Check `wiki/index.md`** — are there existing pages on this topic? Read them before writing.

4. **Update or create** page(s):
   - Prefer updating an existing page over creating a new one when the idea belongs there
   - Create a new page only when no existing page fits, or the new topic is meaningfully distinct
   - A single input may touch multiple pages (e.g. a finding about compat_threshold touches `findings/haneat-compat-threshold.md` and `concepts/compat-threshold.md`)

5. **Add/update `[[wikilinks]]`** between related pages — cross-references are what make the wiki valuable.

6. **Update `wiki/index.md`**:
   - Add new page entries under the correct section header with a one-line summary
   - Update existing entries if the page content changed significantly

7. **Append to `wiki/log.md`**:
   ```
   ## [YYYY-MM-DD] ingest | <short title>
   - touched: wiki/findings/foo.md, wiki/concepts/bar.md
   - source: <conversation | paper note | experiment observation>
   - summary: one sentence describing what was filed
   ```

8. **Report to Sven**: list what pages were created/updated and what cross-references were added. Keep it concise — one sentence per page.

### Page templates

**Finding page:**
```markdown
# <Finding title>

<One-line summary of what was observed.>

## Observation

<What was measured or seen. Be specific — cite experiment names, conditions, seed numbers, metrics.>

## Conditions

- Experiment: `<experiment name>`
- Algorithm: NEAT | HA-NEAT
- compat_threshold: 0.3 | 0.5
- Seeds: <list>

## Implications

<What this finding suggests for the thesis argument. Link to related hypotheses and concepts.>

## Related

- [[hypothesis-name]]
- [[concept-name]]
```

**Hypothesis page:**
```markdown
# <Hypothesis title>

<One-line statement of the conjecture.>

## Claim

<The full hypothesis in plain language.>

## Status

open | supported | refuted | inconclusive

## Evidence

- [[finding-name]] — supports / refutes / neutral
- <Add entries as evidence accumulates>

## Open questions

- <What would confirm or refute this hypothesis?>

## Related

- [[concept-name]]
- [[finding-name]]
```

**Concept page:**
```markdown
# <Concept name>

<One-line definition.>

## Description

<Explanation of the concept — mechanism, design decision, or definition. Keep it precise.>

## Role in the thesis

<How this concept connects to the NEAT vs HA-NEAT comparison or the multi-task setup.>

## Related

- [[finding-name]]
- [[hypothesis-name]]
```

**Paper page:**
```markdown
# <Author(s), Year — Short Title>

<One-line summary of the paper's core contribution.>

## Citation

<Full citation in your preferred format.>

## Core contribution

<What the paper does and why it matters.>

## Relevance to thesis

<How it connects to NEAT, HA-NEAT, multi-task learning, or your experimental setup.>

## Key points

- <Bullet 1>
- <Bullet 2>

## Quotes / figures worth citing

> "<Exact quote>" (p. X)
```

**Excerpt page:**
```markdown
# <YYYY-MM-DD — Short topic>

<One-line summary of the insight.>

## Context

<Who/what conversation. Date. Why it matters.>

## Excerpt

<The relevant text, slightly cleaned up if needed. Keep it close to verbatim.>

## Key insight

<The one thing worth remembering from this excerpt.>

## Related

- [[concept-name]]
- [[hypothesis-name]]
```

---

## Operation 2: Query

**Goal:** Answer Sven's question using what's already in the wiki.

### Steps

1. Read `wiki/index.md` first to identify relevant pages.
2. Read those pages in full.
3. Synthesize an answer with citations — reference pages as `wiki/findings/foo.md` or link-style `[[foo]]`.
4. If the synthesized answer adds new value not already in a page, offer to file it back into the wiki as a new page (a comparison, an analysis, a connection).
5. Append to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] query | <topic>
   - pages read: wiki/findings/foo.md, wiki/concepts/bar.md
   - summary: what was asked and what was found
   ```

---

## Operation 3: Lint

**Goal:** Health-check the wiki. Report issues; do not auto-fix.

### Steps

1. Read `wiki/index.md` and all pages it references.
2. Check for:
   - **Contradictions** — two pages making conflicting claims
   - **Orphan pages** — pages not linked from any other page (check inbound links)
   - **Stale claims** — findings that newer pages supersede or contradict
   - **Missing cross-references** — pages that mention a concept/finding but don't link to it
   - **Missing concept pages** — terms appearing in multiple pages but lacking their own page
   - **Data gaps** — important questions that have no finding or hypothesis page yet
3. Report findings as a structured list:
   ```
   ## Lint report — YYYY-MM-DD

   ### Contradictions
   - wiki/findings/foo.md claims X; wiki/findings/bar.md claims not-X

   ### Orphan pages
   - wiki/hypotheses/baz.md — no inbound links

   ### Missing concept pages
   - "historical markers" mentioned in 3 pages but no wiki/concepts/historical-markers.md

   ### Suggested investigations
   - "Why does HA-NEAT produce fewer species at compat=0.5?" has no hypothesis page yet
   ```
4. Append to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] lint | <N issues found>
   - summary: brief description of main issues
   ```

---

## Example: Full Ingest Round-Trip

**Sven says:**
> "I noticed that HA-NEAT runs with compat_threshold=0.3 consistently produce more species than compat_threshold=0.5 across all 5 seeds. Like 12 vs 6 on average. Could be that 0.3 overcounts topological noise as separate species."

**Claude does:**

1. Classify: this has both a **finding** (more species at 0.3) and a **hypothesis** (0.3 overcounts topological noise).
2. Extract: (a) HA-NEAT compat=0.3 → ~12 species avg; compat=0.5 → ~6 species avg, across 5 seeds. (b) Hypothesis: lower threshold treats topological noise as speciation signal.
3. Check index: no existing pages on this. Create two.
4. Create `wiki/findings/haneat-compat-threshold-species-count.md` with the observed numbers.
5. Create `wiki/hypotheses/compat-threshold-speciation-noise.md` linking back to the finding.
6. Check if `wiki/concepts/compat-threshold.md` exists. If not, create it with a short definition.
7. Update `wiki/index.md` with all three new entries.
8. Append to `wiki/log.md`:
   ```
   ## [2026-04-20] ingest | HA-NEAT compat_threshold species count
   - touched: wiki/findings/haneat-compat-threshold-species-count.md, wiki/hypotheses/compat-threshold-speciation-noise.md, wiki/concepts/compat-threshold.md
   - source: conversation observation
   - summary: HA-NEAT produces ~2x more species at compat=0.3 vs 0.5; hypothesis filed that lower threshold overcounts topological noise.
   ```
9. Report: "Filed 3 pages: finding (species counts), hypothesis (noise overcount), concept (compat_threshold). All three link to each other."
