"""Rebuild the 20-shot budget curves from the accompanying run-level results."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

FIGURES = Path(__file__).resolve().parent
ANALYSIS = FIGURES.parent / "results" / "analysis"
BUDGETS = (25, 50, 75, 100)
TASKS = {
    "boolean_expressions": ("Boolean Expressions", "#087F78", "o", "-"),
    "date_understanding": ("Date Understanding", "#BB542E", "s", "--"),
}


def read_series():
    with (ANALYSIS / "resource_profiles.csv").open(newline="") as stream:
        resources = {row["id"]: row for row in csv.DictReader(stream)}
    by_condition = defaultdict(dict)
    with (ANALYSIS / "absolute_by_run.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["stage"], row["shots"], row["param_mode"], row["updates"]) != (
                "controls", "20", "both", "16"
            ):
                continue
            budget = round(float(row["budget"]) * 100)
            expected_method = "full" if budget == 100 else "answer_attention"
            if row["method"] != expected_method:
                raise ValueError(f"Unexpected selection method for {row['id']}")
            profile = resources[row["id"]]
            seeds = by_condition[row["task"], budget]
            seed = int(row["demo_seed"])
            if seed in seeds:
                raise ValueError(f"Duplicate demonstration seed for {row['id']}")
            seeds[seed] = {
                "accuracy": 100 * int(row["n_correct"]) / int(row["n_questions"]),
                "time": float(profile["adaptation_seconds"]),
                "memory": float(profile["adaptation_peak_allocated_bytes"]) / 2**30,
            }
    result = {}
    for task in TASKS:
        for budget in BUDGETS:
            block = by_condition[task, budget]
            if set(block) != {0, 1, 2}:
                raise ValueError(f"Expected three paired seeds for {task} at {budget}%")
            result[task, budget] = {
                metric: mean(sample[metric] for sample in block.values())
                for metric in ("accuracy", "time", "memory")
            }
    return result


def main():
    data = read_series()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 5.4))
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.235, top=0.68, wspace=0.30)
    ink, muted = "#193442", "#48606B"
    fig.text(0.055, 0.91, "How KV retention changes accuracy and cost",
             fontsize=22, fontweight="bold", color=ink)
    fig.text(0.055, 0.846, "20 demonstrations  ·  16 joint KV updates  ·  mean of three demonstration seeds",
             fontsize=12, color=muted)
    panels = [
        ("accuracy", "Test accuracy (%)", (48, 90), [50, 60, 70, 80, 90]),
        ("time", "Adaptation time (s)", (11.6, 22.5), [12, 15, 18, 21]),
        ("memory", "Peak GPU allocation (GiB)", (14, 20.7), [14, 16, 18, 20]),
    ]
    for ax, (metric, title, limits, ticks) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=ink, pad=12)
        for task, (label, color, marker, line) in TASKS.items():
            values = [data[task, budget][metric] for budget in BUDGETS]
            ax.plot(BUDGETS, values, label=label, color=color, marker=marker,
                    linestyle=line, linewidth=2.4, markersize=6)
            for x, y in zip(BUDGETS, values):
                upper = task == ("boolean_expressions" if metric == "accuracy" else "date_understanding")
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 9 if upper else -11),
                            textcoords="offset points", ha="center", va="center",
                            fontsize=9, fontweight="medium", color=color)
        ax.set_xlim(19, 106)
        ax.set_ylim(*limits)
        ax.set_xticks(BUDGETS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.set_yticks(ticks)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.8, color="#DCE5E7")
        ax.tick_params(axis="both", colors=muted, length=0, pad=6)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#A5B6BD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.052, 0.82),
               frameon=False, ncol=2, handlelength=2.8, columnspacing=1.8, borderaxespad=0)
    fig.text(0.515, 0.116, "Demonstration KV retained", ha="center",
             fontsize=11, fontweight="semibold", color=ink)
    fig.text(0.055, 0.046, "25–75% uses answer attention. 100% is the full cache. "
             "All settings retain the instruction prefix.", color=muted, fontsize=10)
    fig.savefig(FIGURES / "kv_retention_cost.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
