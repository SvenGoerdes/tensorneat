# TensorNEAT + HA-NEAT

GPU-accelerated NEAT built on JAX, extended with **HA-NEAT** (Heterogeneous Activation NEAT) and **multi-task evaluation** for comparing NEAT vs HA-NEAT on Brax locomotion tasks.

## Project Goal

Compare standard NEAT against HA-NEAT on multi-task Brax environments (Hopper + Walker2D). A single evolved network must solve both tasks simultaneously using zero-padded observations and sliced actions.

## Build & Run

```bash
uv pip install -e .
uv run python main.py --config config.yaml
uv run pytest test/
```

Python >= 3.10. Key deps: jax, brax, gymnax, flax, optax, networkx, sympy, mlflow.
Use `mjx` backend (default). `generalized` is faster but allows Walker2D floor-penetration exploits.

## Architecture

Four-layer design: **Pipeline > Algorithm > Genome > Problem**

- **Pipeline** (`pipeline.py`): `setup → step → auto_run`. MLflow tracking + per-task fitness breakdown.
- **Algorithm** (`algorithm/neat/neat.py`): Speciation-driven NEAT with `SpeciesController`.
- **Genome** (`genome/`): `DefaultGenome` with topological sort. Gene types: `DefaultNode`, `BiasNode`, `DefaultConn`, `OriginConn`. Mutations: `DefaultMutation`, `HANEATMutation`.
- **Problem** (`problem/rl/`): `BraxEnv`, `MultiTaskBraxEnv`. Fitness aggregation: `normalized_min` = `min(norm_hopper, norm_walker2d)`.

See `REFERENCE.md` for internals (HA-NEAT mechanism, pipeline details, aggregation logic).

## Experimental Setup

| Condition | Algorithm | Activations | Notes |
|---|---|---|---|
| Baseline | NEAT | tanh only | Standard mutation |
| Treatment | HA-NEAT | tanh, sigmoid, relu, sin, identity | Historical marker reassignment, speciation protection |
| Ablation | HA-NEAT (tanh only) | tanh only | Tests if activation diversity is the active ingredient |

**Environments:** Hopper (11 obs, 3 act) + Walker2D (17 obs, 6 act)
**Network:** 17 inputs → 6 outputs (zero-padding/slicing for Hopper)

## Experiment History

| Experiment | Key config | Outcome |
|---|---|---|
| `multi_task_neat_vs_haneat` | generalized backend, no normalization | Walker2D exploit via floor penetration |
| `multi_task_neat_vs_haneat_mjx_backend` | mjx, normalized_sum/min/product, pop 512-2048, 100-200 gen | Baseline MJX runs |
| `multi_task_neat_vs_haneat_mjx_backend_v2` | mjx, normalized_min, pop 1024+2048, 500 gen, disjoint sweep [0.5,1.0] | Best run: neat pop2048 disjoint1.0 seed123, fitness=0.43, 11 species |
| `multi_task_neat_vs_haneat_v3` | mjx, normalized_min, pop 2048, 500 gen, compat_threshold=0.5 | 4 runs (neat+ha_neat × seed42+123) |
| `multi_task_neat_vs_haneat_v4` | mjx, normalized_min, pop 4096, 1000 gen, compat_threshold=0.5 | 4 runs (neat+ha_neat × seed42+123) |
| `multi_task_neat_vs_haneat_ablation` | mjx, normalized_min, pop 4096, 1000 gen, compat_threshold=0.5, ha_neat tanh only | 6 runs. No stat. sig. difference (permutation p=0.70, Cohen's d small) |
| `multi_task_neat_vs_haneat_final` | mjx, normalized_min, pop 4096, 2000 gen, compat_threshold=[0.3,0.5], 5 seeds | 10 NEAT + 4 HA-NEAT (server exp 1) |
| `multi_task_haneat_finalv2` | same config, ha_neat only, second GPU | 10 HA-NEAT (server exp 2). Combined with above for full comparison. |

**Final experiment genome counts (server):**
- NEAT: 10 seeds (5 × compat0.3 + 5 × compat0.5), all FINISHED
- HA-NEAT: 14 seeds total across both experiments (dedup needed for compat0.3 duplicates)

## Server access (no longer available)

Training previously ran on a Nova GPU server (`ssh 20240503@10.10.80.3`). **That access has been permanently lost — the SSH command no longer works and there is no way to reach the server.** All experiment work now must run locally; do not suggest SSH, rsync from the server, or `git pull`-to-sync-from-server as a next step.

Data availability (checked locally on 2026-07-02):
- `mlflow.db` (~847 MB) — **present locally** in the repo root, already synced.
- `results/<experiment>/*.npz` — **present locally**, 221 genome files across all experiments (gitignored, not on GitHub).
- `eval/outputs/*.json` — **present locally** (re-evaluation results).
- `analysis/outputs/` — **present locally**, populated with prior analysis runs.
- `output*.log`, `logs/*.log` — **not present locally**. These training stdout logs only ever lived on the server and are now unrecoverable.

Since the local `mlflow.db` and `results/*.npz` already cover everything needed for analysis, this is not expected to block ongoing work — but any server-only artifact not already synced (raw stdout logs, anything not listed above as present) is permanently gone.

## Zotero

PDFs can be added to the user's Zotero library via `tools/add_to_zotero.py` (Web API, pyzotero via PEP 723 inline deps — no project dependency):

```bash
uv run tools/add_to_zotero.py paper.pdf --doi 10.48550/arXiv.2106.13281 --collection "Master Thesis"
uv run tools/add_to_zotero.py paper.pdf --title "Some Paper" --item-type preprint
```

- **Never hand-edit `master_thesis.bib`.** It is auto-exported from Zotero by Better BibTeX. To add a reference, add the item to Zotero (via the tool above), then let Better BibTeX regenerate the `.bib` on sync/auto-export. Manually adding a bib entry will be overwritten and drifts from the library.
- **Verify metadata before adding.** Confirm the DOI resolves on CrossRef (`curl https://api.crossref.org/works/<doi>`) rather than trusting a remembered DOI; a 404 means the DOI is wrong.
- **Check for duplicates first.** Query the library (`items?q=<title>&itemType=-attachment`) before uploading — many papers are already present. If a duplicate is created, delete the new item by key via the Web API (`DELETE items/<key>` with `If-Unmodified-Since-Version`).
- **Collections** (exact names, case-sensitive): default to `Master Thesis` for thesis references. Others: `NEAT Paper`, `LitReview_Priority`, `ActivationFunctions`, `Neuroplasicity`, `Interpretability`, `One Pager`. The `--collection` value must match one of these or the tool aborts before writing.
- After adding an item, get its Better BibTeX citekey via the JSON-RPC `item.search` on port 23119 and use that exact key in `\cite{}`.
- Credentials live in `~/.config/zotero/credentials.json` (`user_id` + `api_key`, key needs personal-library write access). Never print or commit the key.
- Items go to the cloud library and appear in the desktop app on next sync.
- The local Zotero HTTP API (port 23119) is read-only — useless for adding items. Better BibTeX is installed (JSON-RPC on the same port) for citekey lookups.

## Key Directories

```
main.py                      # Config-driven experiment grid runner
config.yaml                  # Current experiment config
eval/
  evaluate_genomes.py        # Batch re-evaluate .npz genomes → JSON
  outputs/                   # JSON re-evaluation results
results/                     # Best genomes saved as .npz per run
thesis/                      # Master thesis LaTeX source (subtree-merged from the former
  main.tex                   #   github.com/SvenGoerdes/master-thesis repo — canonical thesis text)
  master_thesis.bib          # Bibliography
  Figures/                   # Thesis figures
text/                        # Older Markdown drafts (superseded by thesis/main.tex)
  results_and_discussion.md  # Thesis Results & Discussion draft (Markdown, legacy)
  figures/                   # Plotting + stats scripts
    plot_training_curves.py       # Training curves from MLflow SQLite
    plot_evaluation_results.py    # Bar plots from re-evaluation JSON
    statistical_test.py           # Mann-Whitney U, permutation test, Cohen's d
tools/
  add_to_zotero.py           # Upload PDF + metadata to Zotero (see Zotero section)
src/tensorneat/              # Library source
docs/FEATURE_LOG.md          # Running log of all implemented features
REFERENCE.md                 # Detailed implementation notes
```

