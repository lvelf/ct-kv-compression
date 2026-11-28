"""Rebuild the original 10-shot update comparisons from the published CSVs."""
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results" / "analysis"
TEAL, RUST, INK, GRAY = "#087F78", "#BB542E", "#1F3340", "#637682"


def read_csv(name):
    with (DATA / name).open(newline="") as fp:
        return list(csv.DictReader(fp))


def paired_mean(rows, column, **fields):
    subset = [r for r in rows if all(r[k] == str(v) for k, v in fields.items())]
    if len(subset) != 3 or {r["demo_seed"] for r in subset} != {"0", "1", "2"}:
        raise ValueError(f"Expected one observation for each demonstration seed: {fields}")
    return mean(float(r[column]) for r in subset)


def test_scores(task, runs, historical):
    common = dict(task=task, shots=10, method="full", param_mode="both", optimizer_seed=0)
    # Select one historical zero-update ICL reference per seed.
    scores = [paired_mean(historical, "old_accuracy_pp", stage="controls", updates=16,
                          reference="prior Standard ICL", **common)]
    for horizon in (4, 8, 16, 32):
        stage = "controls" if horizon == 16 else "steps"
        scores.append(paired_mean(runs, "accuracy_pp", stage=stage,
                                  updates=horizon, **common))
    return scores


def build():
    runs = read_csv("absolute_by_run.csv")
    historical = read_csv("historical_checks.csv")
    train_rows = read_csv("training_loss.csv")
    val_rows = read_csv("validation_checkpoints.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=.075, right=.96, top=.74, bottom=.21, wspace=.27)
    fig.text(.025, .951, "What changes with more KV updates?", fontsize=22,
             weight="bold", va="top")
    fig.text(.025, .875, "Original 10-shot demonstrations   ·   Full cache   ·   Mean across three demonstration seeds",
             fontsize=11.5, color=GRAY)
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DCE4E8", lw=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#ABB9BF")
        ax.tick_params(length=0, pad=5)

    left.set_title("Test accuracy by training length", loc="left", pad=11,
                   weight="bold", fontsize=14)
    horizons = [0, 4, 8, 16, 32]
    for task, name, color, marker, displacement in (
        ("boolean_expressions", "Boolean", TEAL, "o", 8),
        ("date_understanding", "Date", RUST, "s", -15),
    ):
        scores = test_scores(task, runs, historical)
        left.plot(horizons[:2], scores[:2], ls=":", lw=1.8, color=color)
        left.plot(horizons[1:], scores[1:], marker=marker, lw=2.4, ms=6, color=color)
        left.plot(0, scores[0], marker=marker, linestyle="none", markerfacecolor="white",
                  markeredgewidth=1.8, ms=7, color=color)
        for x, y in zip(horizons, scores):
            left.annotate(f"{y:.1f}", (x, y), (0, displacement),
                          textcoords="offset points", ha="center", color=color, fontsize=9)
        left.annotate(name, (32, scores[-1]), (6, 0), ha="left", va="center",
                      textcoords="offset points", color=color, weight="bold", fontsize=10)
    left.set(xlim=(-1.4, 37.6), ylim=(43, 93), xticks=horizons,
             ylabel="Exact-match accuracy (%)", xlabel="Total updates in each independent run",
             yticks=[50, 60, 70, 80, 90])

    right.set_title("Boolean loss within a 16-update run", loc="left", pad=11,
                    weight="bold", fontsize=14)
    setup = dict(stage="controls", task="boolean_expressions", shots=10, method="full",
                 param_mode="both", optimizer_seed=0, updates=16)
    checkpoints = [0, 2, 4, 8, 16]
    validation = [paired_mean(val_rows, "nll", step=s, **setup) for s in checkpoints]
    updates = range(1, 17)
    training = [paired_mean(train_rows, "training_nll", step=s, **setup) for s in updates]
    right.plot(checkpoints, validation, color=INK, lw=2.6, marker="o", ms=5.5,
               label="Validation")
    right.plot(updates, training, color=GRAY, lw=1.9, ls="--", marker="D", ms=3.4,
               label="Training")
    right.legend(loc="upper left", frameon=False, ncol=2, fontsize=10,
                 borderaxespad=0, handlelength=2., columnspacing=1.2)
    for value, color in ((validation[-1], INK), (training[-1], GRAY)):
        right.annotate(f"{value:.2f}", (16, value), (-1, 8), ha="right",
                       textcoords="offset points", color=color, fontsize=10)
    right.set(xlim=(-.5, 16.7), ylim=(-.12, 2.4), xticks=checkpoints,
              yticks=[0, .5, 1, 1.5, 2], xlabel="Update step in the 16-update schedule",
              ylabel="Mean negative log-likelihood per answer token")
    fig.text(.025, .045, "Open markers show historical ICL accuracy on the same 200 test questions per task. "
             "Validation uses 10 separate questions per run.", fontsize=9.3, color=GRAY)
    fig.savefig(HERE / "kv_update_steps.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
