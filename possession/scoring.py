"""The official metric code, imported rather than re-implemented.

Every score this package reports -- held-out DFL games, synthetic tests, the decoder search -- goes
through evaluation/possession_metrics.py, the file that produces the official result. A private
copy of spellF1 could drift from the real one; an import cannot.
"""

import os
import sys

import numpy as np

from possession import config

sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (needs the path above)

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
LETTERS = np.array(pm.LABELS)


def score(gt_codes, pred_codes, bootstrap_n=0):
    """spellF1, macroF1 and possession counts of one prediction, exactly as the scorer computes."""
    result = pm.evaluate(np.asarray(gt_codes, np.int8), np.asarray(pred_codes, np.int8),
                         bootstrap_n=bootstrap_n)
    keys = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells", "alive_dead_f1",
            "spell_bootstrap_low", "spell_bootstrap_high")
    return {key: result[key] for key in keys}


def spell_f1(gt_codes, pred_codes):
    """The ranking metric alone, without the rest of evaluate(): what the decoder search needs."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(np.asarray(pred_codes, np.int8), runs)
    return pm.spell_scores(gt_codes, filled, runs)["mean_f1"]


def paired_spell_bootstrap(games, n_draws=1000, seed=0):
    """Mean spellF1 difference (B - A) over games, with a paired bootstrap interval.

    games: [(gt, pred_a, pred_b)]. Each draw resamples every game's ground-truth possessions and
    scores both versions on the same draw, so the interval is about the difference itself -- far
    tighter than comparing two separate intervals. Matching per possession is done once, as in
    the official pm.bootstrap_spell_f1.
    """
    prepared = []
    for gt, pred_a, pred_b in games:
        gt = np.asarray(gt, np.int8)
        runs = pm.valid_runs(gt)
        gt_spells = pm.possession_spells(gt, runs)
        sides = []
        for pred in (pred_a, pred_b):
            filled = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
            spells = pm.possession_spells(filled, runs)
            overlaps = pm.spell_overlaps(gt_spells, spells)
            flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
                               for threshold in pm.SPELL_IOUS])
            sides.append((flags, len(spells)))
        prepared.append((len(gt_spells), sides))

    def mean_f1(flags, n_pred, picks):
        f1s = []
        for row in flags:
            tp = int(row[picks].sum())
            fp, fn = max(0, n_pred - tp), len(picks) - tp
            f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return np.mean(f1s)

    def difference(draw_picks):
        return np.mean([mean_f1(*b, picks) - mean_f1(*a, picks)
                        for (n, (a, b)), picks in zip(prepared, draw_picks)])

    point = difference([np.arange(n) for n, _ in prepared])
    rng = np.random.RandomState(seed)
    draws = [difference([rng.randint(0, n, n) for n, _ in prepared]) for _ in range(n_draws)]
    low, high = np.percentile(draws, pm.BOOTSTRAP_PERCENTILES)
    return float(point), float(low), float(high)


def write_labels(path, codes):
    """A frame,label CSV in the submission format. Byte-identical for identical codes."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    letters = LETTERS[np.asarray(codes, np.int64)]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(letters))
    return path


def validation_error(path, n_expected):
    """None if the file passes the official validation, else the scorer's one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_expected)
