"""The gate: learns how far to trust each expert, reading their outputs over about +-10 s.

It is a stacked model -- tree ensembles over the experts' outputs plus a handful of context
features that say which situation the frame is in (is the ball seen, how long since it was, how
many players are tracked). Trees split on context and weigh experts inside each split, which is a
learned, context-dependent mixture without the training instability of an end-to-end mixture of
experts.

It is also the sequence model. Besides each expert's output at the frame itself it sees the same
outputs shifted up to two seconds either way, and averaged over windows of up to ten seconds, so
it can learn patterns no single frame shows: the ball expert flipping a few frames ahead of the
shape expert, a pass that is a pass because the other team never gets near it. It makes two
predictions from those inputs:

    state     P(H, A, D) per frame, for the decoder's per-frame evidence
    boundary  P(a change to H / A / D is near), for the decoder's change cost

It is trained only on OUT-OF-FOLD expert outputs: predictions each expert made on a game it was
not trained on. Trained on in-sample outputs, the gate would learn to trust whichever expert
overfits most.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import config
from possession.experts import SWAP_BOUNDARY, SWAP_STATE
from possession.features import rolling_mean
from possession.tracking import period_blocks

# Params
CONTEXT = ("S.ball_vis.0", "S.ball_vis.w25", "S.since_seen.0", "S.until_seen.0", "S.gap_len.0",
           "S.n_vis.0", "A.cnt.0", "B.cnt.0")
EPS = 1e-4
# Frames. Shifts reach two seconds each way; windows average over up to ten seconds.
OFFSETS = (-50, -25, -12, -5, -2, 2, 5, 12, 25, 50)
WINDOWS = (12, 50, 125)
SIDE = 25                      # past-only and future-only means, for before/after contrasts
BOUNDARY_OFFSETS = (-5, -2, 2, 5)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _shift(values, k):
    out = np.full_like(values, np.nan)
    if k > 0:
        out[k:] = values[:-k]
    elif k < 0:
        out[:k] = values[-k:]
    else:
        out[:] = values
    return out


class Gate(object):
    def __init__(self, output_names, swap_index, params=None, temporal=True):
        self.output_names = list(output_names)
        self.swap_index = np.asarray(swap_index)
        self.params = params or config.GATE_HGB
        self.temporal = temporal
        self.boundary_columns = [i for i, n in enumerate(self.output_names)
                                 if n.startswith("transition.")]
        self.state_columns = [i for i in range(len(self.output_names))
                              if i not in self.boundary_columns]
        self.state_model = self.boundary_model = None

    # -- inputs ------------------------------------------------------------------------------

    def _inputs(self, item, order, rows):
        """Gate inputs for one game in one team order, at the given frames."""
        n = len(item["period"])
        expert = item["E"] if order == (0, 1) else item["E"][:, self.swap_index]
        full = np.full((n, expert.shape[1]), np.nan)
        full[item["rows"]] = logit(expert)
        context = np.full((n, len(CONTEXT)), np.nan, dtype=np.float32)
        context[item["rows"]] = item["ctx_ab"] if order == (0, 1) else item["ctx_ba"]
        parts = [full[rows], context[rows]]
        if self.temporal:
            parts.append(self._temporal(full, item["period"])[rows])
        return np.hstack(parts).astype(np.float32)

    def _temporal(self, full, period):
        """Shifted and window-averaged expert outputs, never reaching across a period edge."""
        n = len(full)
        state, boundary = self.state_columns, self.boundary_columns
        width = (len(state) * (len(OFFSETS) + len(WINDOWS) + 2)
                 + len(boundary) * (len(BOUNDARY_OFFSETS) + 1))
        out = np.full((n, width), np.nan, dtype=np.float32)
        for start, stop, _ in period_blocks(period):
            block = full[start:stop]
            columns = []
            for j in state:
                values = block[:, j]
                columns.extend(_shift(values, k) for k in OFFSETS)
                columns.extend(rolling_mean(values, half, half) for half in WINDOWS)
                columns.append(rolling_mean(values, SIDE, -1))
                columns.append(rolling_mean(values, -1, SIDE))
            for j in boundary:
                values = block[:, j]
                columns.extend(_shift(values, k) for k in BOUNDARY_OFFSETS)
                columns.append(rolling_mean(values, 5, 5))
            out[start:stop] = np.column_stack(columns)
        return out

    # -- fit / predict ---------------------------------------------------------------------

    def fit(self, items, stride, training_rows):
        """items: out-of-fold expert outputs per game, with labels. training_rows: the pipeline's
        row sampler, so the gate samples frames exactly as the experts do."""
        X_state, y_state, X_bound, y_bound = [], [], [], []
        for item in items:
            labels, period = item["labels"], item["period"]
            rows, is_stride, boundary = training_rows(labels, period, stride)
            rows_state = rows[is_stride]
            for order, swap_y, swap_b in (((0, 1), np.arange(3), np.arange(4)),
                                          ((1, 0), SWAP_STATE, SWAP_BOUNDARY)):
                X_state.append(self._inputs(item, order, rows_state))
                y_state.append(swap_y[labels[rows_state]])
                X_bound.append(self._inputs(item, order, rows))
                y_bound.append(swap_b[boundary[rows]])
        self.state_model = HistGradientBoostingClassifier(**self.params).fit(
            np.vstack(X_state), np.concatenate(y_state))
        self.boundary_model = HistGradientBoostingClassifier(**self.params).fit(
            np.vstack(X_bound), np.concatenate(y_bound))
        return self

    def predict(self, item):
        """((n, 3) P(H, A, D), (n, 3) P(change to H / A / D near)), symmetric in the teams.
        Frames outside play get uniform values; the decoder never looks at them."""
        n = len(item["period"])
        rows = item["rows"]
        probs, bound = np.full((n, 3), 1.0 / 3), np.full((n, 3), 0.1)
        ab, ba = self._inputs(item, (0, 1), rows), self._inputs(item, (1, 0), rows)
        probs[rows] = 0.5 * (_proba(self.state_model, ab, 0)
                             + _proba(self.state_model, ba, 0)[:, SWAP_STATE])
        bound[rows] = 0.5 * (_proba(self.boundary_model, ab, 1)
                             + _proba(self.boundary_model, ba, 1)[:, SWAP_STATE])
        return probs, bound


def _proba(model, X_rows, shift):
    """(n, 3) class probabilities; shift 1 drops the boundary model's "no change" class."""
    proba = model.predict_proba(X_rows)
    out = np.zeros((len(X_rows), 3))
    for j, cls in enumerate(model.classes_):
        if int(cls) >= shift:
            out[:, int(cls) - shift] = proba[:, j]
    return out

