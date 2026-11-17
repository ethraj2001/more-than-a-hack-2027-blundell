"""docs/approaches.gif and docs/approaches.png: the approaches we tried, in the order we tried them,
by held-out spellF1. Aggregate scores only -- nothing from the provided match.

    python docs/make_approaches.py

Every score is a game predicted by a model that never saw it: the seven DFL games each held out in
turn, and Metrica's three sample games by models trained on DFL games alone. Sources: possession/
README.md ("How it got here") and reports/claude/runs (sbp, msb, svm, cpu/base/flow, feat).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Params
STEPS = (  # (name, note under it, DFL, Metrica, caption)
    ("Published baseline", "nearest player, smoothed", 0.312, 0.360,
     "The floor: the nearest tracked player to the ball, majority vote over ±50 frames."),
    ("Control: kernel SVM", "same features, one model", 0.690, 0.692,
     "Our control: a kernel SVM on the same features. No experts, no gate, no flow model."),
    ("Mixture of experts", "five experts, gate, decoder", 0.733, 0.704,
     "Five boosted-tree experts, a gate that reads them over ±10 s, and a decoder."),
    ("Mixture + flow model", "blended half and half", 0.745, 0.737,
     "The mixture blended half and half with an LSTM flow-matching model."),
    ("+ lanes, keepers, turns", "submitted", 0.748, 0.741,
     "Submitted. Three feature families added: +0.004 on both sets, intervals cross zero."),
)
SERIES = (("Seven DFL games, each held out", "#2a78d6"),
          ("Metrica's three games, never trained on", "#eb6834"))
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FPS, GROW, HOLD, LAST_HOLD = 12, 7, 26, 54   # frames: a bar growing, a step held, the last held
BAR, GAP = 0.30, 0.04
X_MAX = 0.86


def frames():
    """(step, share of that step's bars drawn) per frame."""
    out = []
    for step in range(len(STEPS)):
        out += [(step, (i + 1) / GROW) for i in range(GROW)]
        out += [(step, 1.0)] * (LAST_HOLD if step == len(STEPS) - 1 else HOLD)
    return out


def draw(ax, caption, step, share):
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(len(STEPS) - 0.45, -0.55)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color=MUTED, linewidth=1)
    for row, (name, note, dfl, met, _) in enumerate(STEPS[:step + 1]):
        grown = share if row == step else 1.0
        ax.text(-0.012, row - 0.07, name, ha="right", va="center", fontsize=11.5, color=INK,
                fontweight="bold" if row == step else "normal")
        ax.text(-0.012, row + 0.19, note, ha="right", va="center", fontsize=9, color=MUTED)
        for k, value in enumerate((dfl, met)):
            y = row + (k - 0.5) * (BAR + GAP)
            ax.barh(y, value * grown, height=BAR, color=SERIES[k][1], linewidth=0)
            if grown == 1.0:
                ax.text(value + 0.008, y, "{:.3f}".format(value), ha="left", va="center",
                        fontsize=10, color=INK)
    caption.set_text(STEPS[step][4])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figure = plt.figure(figsize=(9.6, 5.6), dpi=100, facecolor=SURFACE)
    ax = figure.add_axes([0.27, 0.17, 0.70, 0.62])
    figure.text(0.03, 0.945, "What each approach scores on games it never saw", fontsize=15,
                color=INK, fontweight="bold")
    figure.text(0.03, 0.895, "Mean spellF1, higher is better. Versions were chosen on these two "
                             "sets, never on the provided match.", fontsize=10, color=MUTED)
    for k, (label, colour) in enumerate(SERIES):
        figure.patches.append(plt.Rectangle((0.03 + 0.31 * k, 0.832), 0.016, 0.024, color=colour,
                                            transform=figure.transFigure))
        figure.text(0.052 + 0.31 * k, 0.844, label, fontsize=10, color=INK, va="center")
    caption = figure.text(0.03, 0.05, "", fontsize=11, color=INK)
    order = frames()
    animation = FuncAnimation(figure, lambda i: draw(ax, caption, *order[i]), frames=len(order))
    animation.save(os.path.join(here, "approaches.gif"), writer=PillowWriter(fps=FPS))
    draw(ax, caption, len(STEPS) - 1, 1.0)
    figure.savefig(os.path.join(here, "approaches.png"), dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
