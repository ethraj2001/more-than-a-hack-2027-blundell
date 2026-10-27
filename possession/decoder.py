"""From per-frame probabilities to possessions: a Viterbi decode with minimum durations.

spellF1 scores possessions, not frames, so the last step must produce a realistic number of them
with sharp edges. Per-frame argmax flickers into thousands of fake possessions; a majority filter
blurs every edge by its window. The decoder does neither: it finds the single label sequence that
best balances

    alpha * log P(label | frame)                    per-frame evidence, from the gate
    beta  * log P(change to k here) / log P(none)   from the transition expert
    log pi(from -> to) - switch_penalty             how changes usually go, and a cost per change

subject to every possession lasting at least min_possession frames and every stoppage at least
min_dead. Minimum durations are enforced exactly by expanding each label into a chain of
sub-states that must be walked through before the label may change.

Decoding runs per period, never across halftime.
"""

import numpy as np

from possession import config
from possession.scoring import A, D, H, X, spell_f1
from possession.tracking import period_blocks

# Params
EPS = 1e-6
LABELS = (H, A, D)


def transition_prior(label_arrays, period_arrays):
    """log pi[from, to] for label changes in training data, diagonal -inf. Add-one smoothed."""
    counts = np.ones((3, 3))
    for labels, period in zip(label_arrays, period_arrays):
        for start, stop, _ in period_blocks(period):
            window = labels[start:stop]
            window = window[window != X]
            change = np.flatnonzero(window[1:] != window[:-1])
            np.add.at(counts, (window[change], window[change + 1]), 1)
    np.fill_diagonal(counts, 0)
    with np.errstate(divide="ignore"):
        return np.log(counts / counts.sum(axis=1, keepdims=True))


def decode(probs, boundary, period, params, log_pi):
    """(T,) label codes: the Viterbi path inside each period, X outside play."""
    out = np.full(len(period), X, dtype=np.int8)
    for start, stop, _ in period_blocks(period):
        out[start:stop] = viterbi(probs[start:stop], boundary[start:stop], params, log_pi)
    return out


def viterbi(probs, boundary, params, log_pi):
    n = len(probs)
    lengths = [max(2, int(params["min_possession"]))] * 2 + [max(2, int(params["min_dead"]))]
    first = np.cumsum([0] + lengths[:-1])
    last = first + np.array(lengths) - 1
    n_states = int(sum(lengths))
    label_of = np.repeat(np.arange(3), lengths)
    chained = np.setdiff1d(np.arange(n_states), first)    # states reached from their predecessor
    last_pos = np.searchsorted(chained, last)            # where each label's last state sits

    emit = params["alpha"] * np.log(np.clip(probs, EPS, 1.0))
    # boundary is (n,) P(any change) or (n, 3) P(change to H / A / D); either way move[t, to].
    b = np.clip(boundary, EPS, 1 - EPS)
    if b.ndim == 1:  # P(any change): the same evidence for every destination
        stay = params["beta"] * np.log1p(-b)
        move = np.repeat(params["beta"] * np.log(b)[:, None], 3, axis=1)
    else:
        stay = params["beta"] * np.log1p(-np.clip(b.sum(axis=1), EPS, 1 - EPS))
        move = params["beta"] * np.log(b)
    move = move - params["switch_penalty"]
    enter = log_pi  # [from, to]

    delta = emit[0, label_of].copy()
    back = np.empty((n, n_states), dtype=np.int16)
    back[0] = np.arange(n_states)
    new = np.empty(n_states)
    for t in range(1, n):
        # Walk along a chain, or stay put in a label's last (self-looping) state.
        step = delta[chained - 1]
        step_from = chained - 1
        hold = delta[last]
        better = hold > step[last_pos]
        step[last_pos] = np.where(better, hold, step[last_pos])
        step_from[last_pos] = np.where(better, last, step_from[last_pos])
        new[chained] = step + stay[t]
        back[t, chained] = step_from
        # Enter a new label from another label's last state.
        scores = delta[last][:, None] + enter + move[t]
        source = np.argmax(scores, axis=0)
        new[first] = scores[source, np.arange(3)]
        back[t, first] = last[source]
        delta = new + emit[t, label_of]

    state = int(np.argmax(delta))
    path = np.empty(n, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return np.asarray(LABELS, dtype=np.int8)[label_of[path]]


def tune(cases, log_pi, grid=None, start=None, passes=2):
    """Coordinate ascent over the grid, maximising mean spellF1 over the cases.

    cases: [(probs, boundary, period, labels)] -- out-of-fold DFL output, never the provided game.
    Returns (best params, best mean spellF1, every (params, score) tried).
    """
    grid = grid or config.DECODER_GRID
    best = dict(start or config.DECODER_DEFAULT)
    tried = {}

    def evaluate(params):
        key = tuple(sorted(params.items()))
        if key not in tried:
            scores = [spell_f1(labels, decode(p, b, period, params, log_pi))
                      for p, b, period, labels in cases]
            tried[key] = float(np.mean(scores))
        return tried[key]

    best_score = evaluate(best)
    for _ in range(passes):
        improved = False
        for name, values in sorted(grid.items()):
            for value in values:
                candidate = dict(best, **{name: value})
                score = evaluate(candidate)
                if score > best_score + 1e-9:
                    best, best_score, improved = candidate, score, True
        if not improved:
            break
    history = [(dict(key), score) for key, score in tried.items()]
    return best, best_score, history
