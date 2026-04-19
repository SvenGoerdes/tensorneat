"""Merge and deduplicate HA-NEAT genomes from two experiments."""

import os
import re
import glob
from dataclasses import dataclass


@dataclass
class GenomeEntry:
    path: str
    fname: str
    algorithm: str  # 'neat' | 'ha_neat' | 'ha_neat_ablation'
    seed: int | None
    compat: float | None
    experiment: str


def _parse_fname(fname: str, experiment: str) -> GenomeEntry:
    seed_m = re.search(r"seed(\d+)", fname)
    compat_m = re.search(r"compat([\d.]+)", fname)
    seed = int(seed_m.group(1)) if seed_m else None
    compat = float(compat_m.group(1)) if compat_m else None

    if fname.startswith("ha_neat_ablation"):
        algo = "ha_neat_ablation"
    elif fname.startswith("ha_neat"):
        algo = "ha_neat"
    else:
        algo = "neat"

    return GenomeEntry(path="", fname=fname, algorithm=algo, seed=seed, compat=compat, experiment=experiment)


def collect_genomes(results_root: str, experiments: list[str]) -> list[GenomeEntry]:
    """Collect all .npz genome entries from both experiments."""
    entries: list[GenomeEntry] = []
    for exp in experiments:
        exp_dir = os.path.join(results_root, exp)
        for path in sorted(glob.glob(os.path.join(exp_dir, "*.npz"))):
            fname = os.path.basename(path)
            entry = _parse_fname(fname, exp)
            entry.path = path
            entries.append(entry)
    return entries


def dedup_haneat(entries: list[GenomeEntry]) -> list[GenomeEntry]:
    """
    Deduplicate HA-NEAT genomes across experiments.
    If same (seed, compat) appears in both, keep the one with the larger file
    (proxy for more training completed — larger npz means more data saved).
    """
    ha_entries = [e for e in entries if e.algorithm == "ha_neat"]
    neat_entries = [e for e in entries if e.algorithm != "ha_neat"]

    seen: dict[tuple, GenomeEntry] = {}
    for e in ha_entries:
        key = (e.seed, e.compat)
        if key not in seen:
            seen[key] = e
        else:
            # prefer larger file (more training)
            if os.path.getsize(e.path) > os.path.getsize(seen[key].path):
                seen[key] = e

    deduped_ha = list(seen.values())
    return neat_entries + deduped_ha


def split_by_algorithm(entries: list[GenomeEntry]) -> dict[str, list[GenomeEntry]]:
    groups: dict[str, list[GenomeEntry]] = {}
    for e in entries:
        groups.setdefault(e.algorithm, []).append(e)
    return groups
