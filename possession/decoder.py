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
sub-states that must be walked through before the label may change -- conceptually; the code
collapses each chain to a running sum, which gives the same path an order of magnitude faster.

Decoding runs per period, never across halftime.
"""

import numpy as np

from possession import config
from possession.scoring import A, D, H, X, macro_f1, spell_f1
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


def absorb_flicks(codes, period, frames):
    """codes with each possession shorter than frames that interrupts the other team's -- between
    two of its possessions, or between one and a stoppage -- given to that team: a touch without
    control, or a deflection, is not a change of possession (CHALLENGE.md)."""
    out = codes.copy()
    if frames <= 0:
        return out
    for start, stop, _ in period_blocks(period):
        block = out[start:stop]
        edges = np.flatnonzero(np.diff(block)) + 1
        bounds = np.concatenate(([0], edges, [len(block)]))
        for i in range(1, len(bounds) - 2):
            a, b = bounds[i], bounds[i + 1]
            before, label, after = block[bounds[i - 1]], block[a], block[b]
            if (label in (H, A) and before in (H, A) and before != label and b - a < frames
                    and after in (before, D)):
                block[a:b] = before
    return out


def viterbi(probs, boundary, params, log_pi):
    n = len(probs)
    lengths = [max(2, int(params["min_possession"]))] * 2 + [max(2, int(params["min_dead"]))]
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

    # A chain has no choices inside it: a label entered at s is still in it at t, and has scored
    # enter + the stay-and-emit gains of frames s+1..t. So the recursion only tracks, per label,
    # the score of entering it at each frame (E) and of its last, self-looping state (l), and
    # reads a whole chain off a running sum (T). Plain Python floats over three labels are
    # about 12x faster than numpy over every chain state, and give the same path.
    gain = stay[:, None] + emit
    total = np.cumsum(gain, axis=0)
    arrive = move + emit
    (p01, p02), (p10, p12), (p20, p21) = [[float(log_pi[j, k]) for k in range(3) if k != j]
                                          for j in range(3)]
    G0, G1, G2 = (gain[:, k].tolist() for k in range(3))
    T0, T1, T2 = (total[:, k].tolist() for k in range(3))
    R0, R1, R2 = (arrive[:, k].tolist() for k in range(3))
    E0, E1, E2 = [0.0] * n, [0.0] * n, [0.0] * n
    E0[0], E1[0], E2[0] = (float(v) for v in emit[0])  # frame 0 may sit anywhere in a chain
    l0, l1, l2 = E0[0], E1[0], E2[0]
    hold = bytearray(3 * n)    # 1: label k's last state at t was held rather than arrived at
    source = bytearray(3 * n)  # the label left when entering k at t
    L0, L1, L2 = lengths
    for t in range(1, n):
        # Enter a label from another label's last state; ties go to the lower label.
        a, c = l1 + p10, l2 + p20
        if c > a:
            E0[t] = c + R0[t]
            source[3 * t] = 2
        else:
            E0[t] = a + R0[t]
            source[3 * t] = 1
        a, c = l0 + p01, l2 + p21
        if c > a:
            E1[t] = c + R1[t]
            source[3 * t + 1] = 2
        else:
            E1[t] = a + R1[t]
        a, c = l0 + p02, l1 + p12
        if c > a:
            E2[t] = c + R2[t]
            source[3 * t + 2] = 1
        else:
            E2[t] = a + R2[t]
        # Hold the last state, or reach it from the chain entered L - 1 frames ago.
        s = max(t - L0 + 1, 0)
        chain = E0[s] + (T0[t - 1] - T0[s])
        if l0 > chain:
            hold[3 * t] = 1
            l0 = l0 + G0[t]
        else:
            l0 = chain + G0[t]
        s = max(t - L1 + 1, 0)
        chain = E1[s] + (T1[t - 1] - T1[s])
        if l1 > chain:
            hold[3 * t + 1] = 1
            l1 = l1 + G1[t]
        else:
            l1 = chain + G1[t]
        s = max(t - L2 + 1, 0)
        chain = E2[s] + (T2[t - 1] - T2[s])
        if l2 > chain:
            hold[3 * t + 2] = 1
            l2 = l2 + G2[t]
        else:
            l2 = chain + G2[t]
    E, T, last = (E0, E1, E2), (T0, T1, T2), (l0, l1, l2)

    # The best final state, which may be part-way along a chain: the match can end mid-possession.
    best, state = -np.inf, None
    for k, length in enumerate(lengths):
        for i in range(length - 1):
            s = max(n - 1 - i, 0)
            value = E[k][s] + (T[k][n - 1] - T[k][s])
            if value > best:
                best, state = value, (k, i)
        if last[k] > best:
            best, state = last[k], (k, None)

    path = np.empty(n, dtype=np.int64)
    (k, i), t = state, n - 1
    while True:
        if i is None:  # in k's last state at t
            if t == 0:
                path[0] = k
                break
            if hold[3 * t + k]:
                path[t] = k
                t -= 1
                continue
            s = t - lengths[k] + 1
        else:          # i frames into k's chain at t
            s = t - i
        path[max(s, 0):t + 1] = k
        if s <= 0:
            break
        k, i, t = source[3 * s + k], None, s - 1
    return np.asarray(LABELS, dtype=np.int8)[path]


def tune(cases, log_pi, grid=None, start=None, passes=2):
    """Coordinate ascent over the grid, maximising mean spellF1 over the cases, with macroF1 at
    weight DECODER_MACRO_WEIGHT to break near-ties.

    cases: [(probs, boundary, period, labels)] -- out-of-fold DFL output, never the provided game.
    Returns (best params, best mean objective, every (params, objective) tried).
    """
    grid = grid or config.DECODER_GRID
    best = dict(start or config.DECODER_DEFAULT)
    tried = {}

    def evaluate(params):
        key = tuple(sorted(params.items()))
        if key not in tried:
            scores = []
            for p, b, period, labels in cases:
                decoded = decode(p, b, period, params, log_pi)
                scores.append(spell_f1(labels, decoded)
                              + config.DECODER_MACRO_WEIGHT * macro_f1(labels, decoded))
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
