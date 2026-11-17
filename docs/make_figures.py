"""The figures of TECHNICAL.md other than the animation: docs/pipeline.png, docs/validation.png,
docs/changes.png and docs/agreement.png. Aggregate scores only -- nothing from the provided match
frame by frame.

    python docs/make_figures.py

Numbers are the held-out measurements quoted in TECHNICAL.md (paired 90% bootstrap intervals).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Params
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TINT = {"data": "#f1f1ef", "model": "#cde2fb", "rule": "#fbe0d4", "out": "#d8efe3"}
HERE = os.path.dirname(os.path.abspath(__file__))
# (label, DFL (point, low, high), Metrica (point, low, high), in the submission?)
CHANGES = (
    ("Lanes, keepers, turns in the mixture", (0.0055, -0.0004, 0.0121), (0.0061, -0.0040, 0.0168), True),
    ("Dilated second stage", (0.0041, 0.0011, 0.0073), (-0.0006, -0.0037, 0.0024), True),
    ("Duel and gap rules", (-0.0005, -0.0021, 0.0011), (-0.0022, -0.0061, 0.0016), True),
    ("Minimum possession 4 frames, not 10", (-0.0035, -0.0051, -0.0016), (-0.0031, -0.0046, -0.0014), True),
    ("Kernel SVM blended in at 0.15", (-0.003, -0.005, -0.000), (0.001, -0.005, 0.007), False),
    ("Out of play beyond 0.5 m, forced", (-0.0016, -0.0029, -0.0004), (-0.0052, -0.0076, -0.0032), False),
    ("Second stage's own change probabilities", (-0.0319, -0.0377, -0.0259), (-0.0267, -0.0367, -0.0171), False),
    ("Takeover needs a run of 2 s", (-0.0515, -0.0606, -0.0417), (-0.0316, -0.0474, -0.0175), False),
)
# Possession-level agreement between the mixture and the flow model (one scored against the other).
AGREE_DFL = (0.852, 0.918, 0.886, 0.885, 0.866, 0.898, 0.898)
AGREE_METRICA = (0.898, 0.836, 0.848)
AGREE_PROVIDED = 0.727


def canvas(width, height):
    figure = plt.figure(figsize=(width, height), dpi=100, facecolor=SURFACE)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return figure, ax


def box(ax, x, y, w, h, title, body="", kind="model"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=TINT[kind], edgecolor=MUTED, linewidth=1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, color=INK,
            fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=8.6, color=MUTED,
                linespacing=1.35)


def arrow(ax, start, end, label="", bend=0.0, at=None):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle="-|>", color=MUTED, linewidth=1.3, shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad={}".format(bend)))
    if label and at is None:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, ha="center",
                va="bottom", fontsize=8.2, color=MUTED)
    elif label:
        ax.text(at[0], at[1], label, ha="left", va="center", fontsize=8.2, color=MUTED)


GREY = "#9a9994"
SIDE = {"H": BLUE, "A": ORANGE, "D": GREY}


def icon_pitch(ax, x, y, w=0.62, h=0.4, ball=True, lost=False):
    """A small pitch with the two teams on it."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#e9f5ec", edgecolor=MUTED, linewidth=0.8))
    ax.plot([x + w / 2] * 2, [y, y + h], color=MUTED, linewidth=0.6)
    ax.add_patch(plt.Circle((x + w / 2, y + h / 2), h * 0.17, fill=False, edgecolor=MUTED,
                            linewidth=0.6))
    for dx, dy, colour in ((0.12, 0.3, BLUE), (0.25, 0.7, BLUE), (0.38, 0.4, BLUE),
                           (0.62, 0.62, ORANGE), (0.76, 0.3, ORANGE), (0.88, 0.7, ORANGE)):
        ax.plot([x + dx * w], [y + dy * h], "o", color=colour, markersize=3.2)
    if lost:
        ax.text(x + 0.5 * w, y + 0.5 * h, "?", ha="center", va="center", fontsize=9, color=INK,
                fontweight="bold")
    elif ball:
        ax.plot([x + 0.46 * w], [y + 0.52 * h], "o", color=INK, markersize=2.6)


def icon_strip(ax, x, y, w, pattern, h=0.16, letters=False):
    """A timeline of possession: pattern is [(label, share)]."""
    at = x
    for label, share in pattern:
        ax.add_patch(plt.Rectangle((at, y), w * share, h, facecolor=SIDE[label], edgecolor=SURFACE,
                                   linewidth=0.6))
        if letters and share > 0.12:
            ax.text(at + w * share / 2, y + h / 2, label, ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold")
        at += w * share


def icon_stats(ax, x, y):
    for dx, colour in ((0.0, BLUE), (0.55, ORANGE)):
        ax.add_patch(matplotlib.patches.Ellipse((x + dx + 0.2, y + 0.2), 0.42, 0.26, fill=False,
                                                edgecolor=colour, linewidth=1.3))
        ax.plot([x + dx + 0.2], [y + 0.2], "+", color=colour, markersize=7, markeredgewidth=1.5)


def icon_trees(ax, x, y, n=5):
    for k in range(n):
        cx = x + k * 0.3
        for dx in (-0.08, 0.08):
            ax.plot([cx, cx + dx], [y + 0.36, y + 0.2], color=MUTED, linewidth=0.9)
            for ddx in (-0.04, 0.04):
                ax.plot([cx + dx, cx + dx + ddx], [y + 0.2, y + 0.04], color=MUTED, linewidth=0.9)
        ax.plot([cx], [y + 0.36], "o", color=BLUE, markersize=4)


def icon_gate(ax, x, y):
    for k in range(5):
        ax.plot([x + k * 0.22, x + 0.44], [y + 0.4, y + 0.1], color=MUTED, linewidth=0.9)
        ax.plot([x + k * 0.22], [y + 0.4], "o", color=BLUE, markersize=3.5)
    ax.plot([x + 0.44], [y + 0.1], "o", color=INK, markersize=6)


def icon_dilated(ax, x, y):
    centre = 8
    for k in range(17):
        tap = abs(k - centre) in (0, 1, 2, 4, 8)
        ax.plot([x + k * 0.075], [y + 0.05], "o", color=BLUE if tap else GRID, markersize=3.2)
        if tap:
            ax.plot([x + k * 0.075, x + centre * 0.075], [y + 0.05, y + 0.4], color=MUTED,
                    linewidth=0.7)
    ax.plot([x + centre * 0.075], [y + 0.4], "o", color=INK, markersize=5.5)


def icon_lstm(ax, x, y):
    for k in range(4):
        ax.add_patch(plt.Rectangle((x + k * 0.33, y + 0.1), 0.2, 0.2, facecolor=SURFACE,
                                   edgecolor=BLUE, linewidth=1.2))
    for k in range(3):
        ax.annotate("", xy=(x + k * 0.33 + 0.33, y + 0.25), xytext=(x + k * 0.33 + 0.2, y + 0.25),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8, mutation_scale=6))
        ax.annotate("", xy=(x + k * 0.33 + 0.2, y + 0.15), xytext=(x + k * 0.33 + 0.33, y + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8, mutation_scale=6))


def icon_blend(ax, x, y):
    ax.add_patch(matplotlib.patches.Wedge((x, y + 0.2), 0.2, 90, 270, facecolor=BLUE, edgecolor=SURFACE))
    ax.add_patch(matplotlib.patches.Wedge((x, y + 0.2), 0.2, 270, 90, facecolor="#6250d6", edgecolor=SURFACE))
    ax.text(x + 0.3, y + 0.2, "½ + ½", ha="left", va="center", fontsize=9, color=INK)


def icon_rules(ax, x, y):
    for k in range(3):
        ax.add_patch(plt.Rectangle((x, y + k * 0.15), 0.1, 0.1, facecolor=SURFACE, edgecolor=ORANGE,
                                   linewidth=1.1))
        ax.plot([x + 0.02, x + 0.045, x + 0.085], [y + k * 0.15 + 0.05, y + k * 0.15 + 0.02,
                                                    y + k * 0.15 + 0.085], color=ORANGE, linewidth=1.2)
        ax.plot([x + 0.18, x + 0.75 - k * 0.12], [y + k * 0.15 + 0.05] * 2, color=MUTED, linewidth=1.4)


def icon_bars(ax, x, y):
    for k, height in enumerate((0.12, 0.3, 0.4, 0.22, 0.1, 0.06)):
        ax.add_patch(plt.Rectangle((x + k * 0.12, y), 0.1, height, facecolor=BLUE, edgecolor=SURFACE))


def icon_folds(ax, x, y, n=7, held=2):
    for k in range(n):
        ax.add_patch(plt.Rectangle((x + k * 0.2, y + 0.08), 0.16, 0.24,
                                   facecolor=ORANGE if k == held else BLUE, edgecolor=SURFACE))
    ax.text(x + held * 0.2 + 0.08, y - 0.02, "held out", ha="center", va="top", fontsize=7,
            color=MUTED)


def icon_interval(ax, x, y):
    ax.plot([x, x + 1.3], [y + 0.05] * 2, color=MUTED, linewidth=0.8)
    ax.plot([x + 0.45] * 2, [y, y + 0.42], color=INK, linewidth=1)
    ax.text(x + 0.45, y - 0.03, "0", ha="center", va="top", fontsize=7.5, color=MUTED)
    for row, (low, high, colour) in enumerate(((0.6, 1.1, BLUE), (0.3, 0.8, GREY))):
        ax.plot([x + low, x + high], [y + 0.34 - row * 0.17] * 2, color=colour, linewidth=2)
        ax.plot([x + (low + high) / 2], [y + 0.34 - row * 0.17], "o", color=colour, markersize=5)
    ax.text(x + 1.16, y + 0.34, "keep", ha="left", va="center", fontsize=7.5, color=INK)
    ax.text(x + 0.86, y + 0.17, "drop", ha="left", va="center", fontsize=7.5, color=MUTED)


def icon_check(ax, x, y):
    ax.add_patch(plt.Circle((x + 0.2, y + 0.2), 0.2, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.4))
    ax.plot([x + 0.2, x + 0.2, x + 0.3], [y + 0.32, y + 0.2, y + 0.14], color=ORANGE, linewidth=1.4)
    ax.plot([x + 0.6, x + 0.7, x + 0.9], [y + 0.2, y + 0.1, y + 0.34], color=INK, linewidth=2)


NOISY = [("H", .1), ("A", .03), ("H", .12), ("D", .04), ("H", .05), ("A", .14), ("H", .03),
         ("A", .1), ("D", .12), ("A", .04), ("D", .05), ("H", .18)]
CLEAN = [("H", .30), ("A", .30), ("D", .22), ("H", .18)]


def icon(ax, draw, cx, cy, width, extent, *args, **kwargs):
    """draw(inset, 0, 0, ...) scaled to width (data units) and centred on (cx, cy). extent is the
    drawing's own (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = extent
    height = width * (y1 - y0) / float(x1 - x0)
    inset = ax.inset_axes([cx - width / 2.0, cy - height / 2.0, width, height],
                          transform=ax.transData)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.axis("off")
    draw(inset, 0, 0, *args, **kwargs)


def station(ax, x, y, title, caption):
    ax.text(x, y, title, ha="center", va="top", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(x, y - 0.27, caption, ha="center", va="top", fontsize=8.6, color=MUTED,
            linespacing=1.3)


def pitches(ax, x, y, n=1, lost=()):
    for k in range(n):
        icon_pitch(ax, x + k * 0.7, y, lost=k in lost)


def two_strips(ax, x, y):
    icon_strip(ax, x, y + 0.26, 1.5, NOISY, h=0.16)
    icon_strip(ax, x, y, 1.5, CLEAN, h=0.16)


def one_strip(ax, x, y):
    icon_strip(ax, x, y, 1.5, CLEAN, h=0.3, letters=True)


def half(ax, x, y):
    ax.add_patch(matplotlib.patches.Wedge((x + 0.2, y + 0.2), 0.2, 90, 270, facecolor=BLUE,
                                          edgecolor=SURFACE))
    ax.add_patch(matplotlib.patches.Wedge((x + 0.2, y + 0.2), 0.2, 270, 90, facecolor="#6250d6",
                                          edgecolor=SURFACE))


PITCH, STRIPS = (-0.02, -0.02, 0.64, 0.42), (0, -0.02, 1.5, 0.44)


def pipeline():
    figure, ax = canvas(12.0, 4.5)
    step, first, y = 1.3, 0.85, 3.0
    at = [first + k * step for k in range(9)]
    parts = (
        (icon_pitch, PITCH, "Tracking", "players, ball,\ntags"),
        (icon_stats, (-0.05, 0.02, 1.0, 0.38), "Features", "team statistics,\nno identities"),
        (icon_trees, (-0.14, 0.0, 1.34, 0.4), "Experts", "five tree\nmodels"),
        (icon_gate, (-0.05, 0.04, 0.93, 0.46), "Gate", "weighs them\nover ±10 s"),
        (icon_dilated, (-0.04, 0.0, 1.24, 0.45), "Dilated stage", "taps out to\n±128 frames"),
        (half, (-0.02, -0.02, 0.42, 0.42), "Blend", "half mixture,\nhalf flow"),
        (icon_rules, (-0.03, -0.03, 0.8, 0.43), "Rules", "duel, gap,\nkick-off"),
        (two_strips, STRIPS, "Decoder", "whole\npossessions"),
        (one_strip, (0, -0.06, 1.5, 0.36), "Submission", "H, A or D\nper frame"),
    )
    for x, (draw, extent, title, caption) in zip(at, parts):
        icon(ax, draw, x, y, 0.5 if draw is half else 0.92, extent)
        station(ax, x, y - 0.5, title, caption)
    for x in at[:-1]:
        arrow(ax, (x + 0.52, y), (x + step - 0.52, y))
    icon(ax, icon_lstm, at[5], 0.95, 0.92, (-0.03, 0.06, 1.22, 0.34))
    station(ax, at[5], 0.6, "LSTM flow model", "fills in masked labels")
    arrow(ax, (at[5], 1.2), (at[5], 1.75))
    arrow(ax, (at[1] + 0.25, 1.8), (at[5] - 0.6, 0.95), bend=0.12)
    arrow(ax, (at[3], y + 0.42), (at[7], y + 0.42), "P(change)", bend=-0.22,
          at=((at[3] + at[7]) / 2 - 0.3, y + 1.12))
    figure.savefig(os.path.join(HERE, "pipeline.png"), dpi=160, facecolor=SURFACE)
    plt.close(figure)


def validation():
    figure, ax = canvas(12.0, 5.6)
    columns, rows = (2.6, 5.3, 8.0, 10.7), (4.6, 2.85, 1.1)
    for y, lane in zip(rows, ("Choose", "Check on\nreal tracking", "Predict")):
        ax.text(0.2, y, lane, ha="left", va="center", fontsize=10.5, color=MUTED, fontweight="bold")
    for y in (3.75, 2.0):
        ax.plot([0.2, 11.8], [y, y], color=GRID, linewidth=1)
    cells = {
        (0, 0): (pitches, (-0.02, -0.02, 2.06, 0.42), 1.5, "Seven DFL games", "tracking-derived labels", dict(n=3)),
        (0, 1): (pitches, PITCH, 0.8, "Noisy copies", "ball lost, paths smoothed", dict(lost=(0,))),
        (0, 2): (icon_folds, (-0.02, -0.2, 1.38, 0.36), 1.3, "Leave one game out", "train on six, score the seventh", {}),
        (0, 3): (icon_interval, (-0.02, -0.12, 1.6, 0.46), 1.5, "Paired bootstrap", "keep only if the 90% interval\nexcludes zero", {}),
        (1, 0): (pitches, (-0.02, -0.02, 2.06, 0.42), 1.5, "Three Metrica games", "real tracker, labels from events", dict(n=3, lost=(1,))),
        (1, 2): (icon_folds, (-0.02, -0.2, 0.58, 0.36), 0.6, "Never trained on", "scored by DFL-trained models", dict(n=3, held=1)),
        (2, 0): (pitches, PITCH, 0.8, "Provided match", "no labels of any kind", dict(lost=(0,))),
        (2, 1): (icon_bars, (-0.02, -0.02, 0.74, 0.44), 0.75, "Noise profile", "counts only", {}),
        (2, 2): (one_strip, (0, -0.06, 1.5, 0.36), 1.3, "Final model predicts", "made twice, bytes compared", {}),
        (2, 3): (icon_check, (-0.03, -0.03, 0.95, 0.43), 0.9, "Public 5 minutes", "alignment check only", {}),
    }
    for (r, c), (draw, extent, width, title, caption, options) in cells.items():
        icon(ax, draw, columns[c], rows[r] + 0.28, width, extent, **options)
        station(ax, columns[c], rows[r] - 0.12, title, caption)
    gap = 0.95
    for r, pairs in ((0, ((0, 1), (1, 2), (2, 3))), (2, ((0, 1), (1, 2), (2, 3)))):
        for a, b in pairs:
            arrow(ax, (columns[a] + gap, rows[r] + 0.28), (columns[b] - gap, rows[r] + 0.28))
    arrow(ax, (columns[0] + gap, rows[1] + 0.28), (columns[2] - gap, rows[1] + 0.28))
    arrow(ax, (columns[2] + gap, rows[1] + 0.28), (columns[3] - 0.2, rows[0] - 0.75), bend=0.2)
    arrow(ax, (columns[1], rows[2] + 0.75), (columns[1], rows[0] - 0.75), "sets the noise",
          at=(columns[1] + 0.1, rows[1] - 0.45))
    figure.savefig(os.path.join(HERE, "validation.png"), dpi=160, facecolor=SURFACE)
    plt.close(figure)


def changes():
    figure = plt.figure(figsize=(10.4, 5.6), dpi=100, facecolor=SURFACE)
    ax = figure.add_axes([0.36, 0.13, 0.44, 0.66])
    figure.text(0.03, 0.94, "What each late change does to held-out spellF1", fontsize=14,
                color=INK, fontweight="bold")
    figure.text(0.03, 0.885, "Paired difference with its 90% interval. An interval that crosses "
                             "zero is not a result.", fontsize=10, color=MUTED)
    for k, (label, colour) in enumerate((("Seven DFL games", BLUE), ("Metrica's three games", ORANGE))):
        figure.text(0.05 + 0.2 * k, 0.83, "●", color=colour, fontsize=11, va="center")
        figure.text(0.068 + 0.2 * k, 0.83, label, color=INK, fontsize=10, va="center")
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=MUTED, linewidth=1)
    for row, (name, dfl, metrica, kept) in enumerate(CHANGES):
        figure_y = row
        ax.text(-0.071, figure_y, name, ha="right", va="center", fontsize=10, color=INK)
        for offset, (point, low, high), colour in ((-0.16, dfl, BLUE), (0.16, metrica, ORANGE)):
            ax.plot([low, high], [figure_y + offset] * 2, color=colour, linewidth=2,
                    solid_capstyle="round")
            ax.plot([point], [figure_y + offset], "o", color=colour, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(0.024, figure_y, "in the submission" if kept else "left out", ha="left",
                va="center", fontsize=9.5, color=INK if kept else MUTED,
                fontweight="bold" if kept else "normal")
    ax.set_xlim(-0.068, 0.022)
    ax.set_ylim(len(CHANGES) - 0.5, -0.6)
    ax.set_yticks([])
    ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02])
    ax.set_xticklabels(["−0.06", "−0.04", "−0.02", "0", "+0.02"])
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("change in mean spellF1", fontsize=10, color=MUTED)
    figure.savefig(os.path.join(HERE, "changes.png"), dpi=160, facecolor=SURFACE)
    plt.close(figure)


def agreement():
    figure = plt.figure(figsize=(10.4, 3.3), dpi=100, facecolor=SURFACE)
    ax = figure.add_axes([0.24, 0.24, 0.72, 0.42])
    figure.text(0.03, 0.9, "The models agree with each other less on the provided match",
                fontsize=14, color=INK, fontweight="bold")
    figure.text(0.03, 0.8, "Possessions of the mixture scored against the flow model's, no labels "
                           "needed. One dot per game.", fontsize=10, color=MUTED)
    ax.set_facecolor(SURFACE)
    for row, (name, values, colour) in enumerate((("Seven DFL games, held out", AGREE_DFL, BLUE),
                                                  ("Metrica's three games", AGREE_METRICA, ORANGE),
                                                  ("Provided match", (AGREE_PROVIDED,), INK))):
        ax.plot(values, [row] * len(values), "o", color=colour, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(0.693, row, name, ha="right", va="center", fontsize=10, color=INK)
    ax.text(AGREE_PROVIDED, 2.42, "{:.2f}".format(AGREE_PROVIDED), ha="center", va="top",
            fontsize=10, color=INK)
    ax.set_xlim(0.70, 0.94)
    ax.set_ylim(2.75, -0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    figure.savefig(os.path.join(HERE, "agreement.png"), dpi=160, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    pipeline()
    validation()
    changes()
    agreement()
