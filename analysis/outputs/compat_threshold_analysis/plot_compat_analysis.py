import sqlite3
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DB = "/Users/svengoerdes/Projects/Thesisv2/tensorneat/mlflow.db"
OUT = "/Users/svengoerdes/Projects/Thesisv2/tensorneat/analysis/outputs/compat_threshold_analysis"

conn = sqlite3.connect(DB)

# NEAT from experiment 9 (multi_task_neat_vs_haneat_final)
# HA-NEAT from experiment 11 (multi_task_haneat_finalv2) — authoritative parallel run
rows = conn.execute("""
    SELECT r.run_uuid, t.value as run_name, r.experiment_id
    FROM runs r
    JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'mlflow.runName'
    WHERE (r.experiment_id = 9 AND t.value LIKE 'neat_%')
       OR  r.experiment_id = 11
    ORDER BY run_name
""").fetchall()

runs = {}
for uuid, name, exp_id in rows:
    m = re.match(r'(neat|ha_neat)_.*_compat(\d\.\d)_seed(\d+)', name)
    if not m:
        continue
    key = (m.group(1), m.group(2), m.group(3))
    runs[key] = (uuid, name)

# ── Plot 1: Bar chart with individual seed dots ──────────────────────────────

groups = [
    ("NEAT", "0.3"),
    ("NEAT", "0.5"),
    ("HA-NEAT", "0.3"),
    ("HA-NEAT", "0.5"),
]
algo_map = {"NEAT": "neat", "HA-NEAT": "ha_neat"}

group_vals = []
for algo, compat in groups:
    key_prefix = algo_map[algo]
    seed_vals = []
    for (a, c, s), (uuid, name) in runs.items():
        if a == key_prefix and c == compat:
            row = conn.execute(
                "SELECT MAX(value) FROM metrics WHERE run_uuid=? AND key='best_fitness'",
                (uuid,)
            ).fetchone()
            if row and row[0] is not None:
                seed_vals.append(row[0])
    group_vals.append(seed_vals)

means = [np.mean(v) for v in group_vals]
labels = [f"{a}\ncompat {c}" for a, c in groups]
colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
alphas = [0.9, 0.45, 0.9, 0.45]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(groups))
bars = ax.bar(x, means, width=0.5, color=colors,
              alpha=1.0, zorder=2)

for i, (bar, vals, a) in enumerate(zip(bars, group_vals, alphas)):
    bar.set_alpha(a)
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
    ax.scatter(x[i] + jitter, vals, color="black", s=28, zorder=3, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Best fitness (normalized_min)", fontsize=10)
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.axhline(0, color="black", linewidth=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# annotate means
for i, m in enumerate(means):
    ax.text(x[i], m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", alpha=0.9, label="NEAT"),
    Patch(facecolor="#DD8452", alpha=0.9, label="HA-NEAT"),
    Patch(facecolor="#4C72B0", alpha=0.45, label="compat 0.5 (lighter)"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper right", framealpha=0.7)

ax.set_title("Mean best fitness by algorithm and compat threshold\n(5 seeds, 2000 gen, pop 4096)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/plot1_compat_bar.png", dpi=150)
plt.close()
print("Saved plot1_compat_bar.png")

# ── Plot 2: Saturation curves — NEAT compat 0.3 all seeds ───────────────────

neat_03_runs = {s: (uuid, name) for (a, c, s), (uuid, name) in runs.items()
                if a == "neat" and c == "0.3"}

fig, ax = plt.subplots(figsize=(8, 4.5))

palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, (seed, (uuid, name)) in enumerate(sorted(neat_03_runs.items())):
    curve = conn.execute(
        "SELECT step, value FROM metrics WHERE run_uuid=? AND key='fitness/max' ORDER BY step",
        (uuid,)
    ).fetchall()
    if not curve:
        continue
    steps = [r[0] for r in curve]
    vals = [r[1] for r in curve]
    ax.plot(steps, vals, color=palette[i % len(palette)],
            linewidth=1.4, alpha=0.85, label=f"seed {seed}")

ax.axvline(750, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="750 gen cutoff")
ax.axvspan(0, 300, alpha=0.06, color="green", label="phase 1 (rapid rise)")
ax.axvspan(300, 700, alpha=0.06, color="orange", label="plateau")
ax.axvspan(700, 1200, alpha=0.06, color="blue", label="phase 2 (breakthrough)")

ax.set_xlabel("Generation", fontsize=10)
ax.set_ylabel("Best fitness (fitness/max)", fontsize=10)
ax.set_title("Saturation curves — NEAT compat 0.3, all seeds", fontsize=10)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 0.65)
ax.grid(linestyle="--", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

fig.tight_layout()
fig.savefig(f"{OUT}/plot2_saturation_curves.png", dpi=150)
plt.close()
print("Saved plot2_saturation_curves.png")

conn.close()
