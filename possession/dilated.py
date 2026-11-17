"""A second stage with a temporal convolutional network's receptive field, at the speed of trees.

A TCN sees far in time through dilated convolutions: taps 1, 2, 4, ... frames apart. Here a
gradient-boosted tree model reads the gate's P(H, A, D) and P(change to H / A / D) at those same
offsets -- 0, +-1, +-2, +-4, ... +-128 frames, about 5 s either way -- and gives P(H, A, D) again.
It is fitted on the gate's out-of-fold output (train's decoder cases), so it learns from output of
the quality it will see. The decoder keeps the gate's own P(change): the second stage's was
measured and is worse.

Held-out, blended with the flow model as `predict` does (config.DILATED_STAGE): seven DFL games
+0.004 (90% interval +0.001 to +0.007), better on 5 of 7; Metrica's three -0.001 (-0.004 to
+0.002), better on 1 of 3. Its own P(change) as well: -0.032 and -0.027, 0 of 10.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession.experts import SWAP_STATE
from possession.scoring import A, D, H

# Params
OFFSETS = (0,) + tuple(sign * step for step in (1, 2, 4, 8, 16, 32, 64, 128) for sign in (-1, 1))
STRIDE = 8        # training rows: one frame in eight
TREES = 100
SWAP_COLUMNS = (1, 0, 2, 4, 3, 5)   # [p_H, p_A, p_D, b_H, b_A, b_D] with the teams swapped


def taps(probs, boundary, period):
    """(T, 6 * len(OFFSETS)) float32: the gate's output at each offset, NaN outside play and
    where the offset crosses into another period."""
    x = np.concatenate([probs, boundary], axis=1).astype(np.float32)
    x[period <= 0] = np.nan
    columns = []
    for offset in OFFSETS:
        y = np.full_like(x, np.nan)
        if offset > 0:
            y[offset:] = x[:-offset]
        elif offset < 0:
            y[:offset] = x[-offset:]
        else:
            y[:] = x
        y[np.roll(period, offset) != period] = np.nan
        columns.append(y)
    return np.concatenate(columns, axis=1)


def swapped(x):
    """taps() of the same frames with the two teams in the other order."""
    return x.reshape(len(x), len(OFFSETS), 6)[:, :, SWAP_COLUMNS].reshape(len(x), -1)


def fit(cases, seed=0):
    """The second stage from [(probs, boundary, period, labels)], each a game's output from a
    gate that never saw it. Every row is used in both team orders."""
    rows, labels = [], []
    for probs, boundary, period, y in cases:
        index = np.flatnonzero((period > 0) & np.isin(y, (H, A, D)))[::STRIDE]
        x = taps(probs, boundary, period)[index]
        rows += [x, swapped(x)]
        labels += [y[index], SWAP_STATE[y[index]]]
    model = HistGradientBoostingClassifier(max_iter=TREES, learning_rate=0.1, random_state=seed)
    model.fit(np.concatenate(rows), np.concatenate(labels))
    if list(model.classes_) != [H, A, D]:
        raise ValueError("the second stage needs all three labels in its training rows")
    return {"model": model, "offsets": OFFSETS}


def refine(stage, probs, boundary, period):
    """probs (T, 3) again from the second stage; frames outside play keep the gate's."""
    if tuple(stage["offsets"]) != OFFSETS:
        raise ValueError("the second stage was fitted with other offsets -- train again")
    out = np.array(probs, dtype=np.float64)
    play = np.flatnonzero(period > 0)
    if len(play):
        out[play] = stage["model"].predict_proba(taps(probs, boundary, period)[play])
    return out
