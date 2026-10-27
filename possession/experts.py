"""The four specialists, and the bank that trains and runs them as one.

Two are specialised by what they may look at, two by what they are asked:

    ball        H/A/D from ball-relative features only -- strong when the ball is seen
    shape       H/A/D from team shape and movement only -- the ball never enters, so it keeps
                working through the 41% of frames where the tracking has lost it
    dead        alive-or-dead from everything
    transition  "a change TO H / A / D is within BOUNDARY_TOLERANCE frames", from local features

Each is a gradient-boosted tree ensemble. Trees take NaN natively, so a missing ball is missing
rather than zero-filled.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import config
from possession.scoring import A, D, H

# Params
STATE, DEAD, BOUNDARY = "state", "dead", "boundary"
BALL_BASES = {"d_ball", "d_ball2", "n_r2", "n_r5", "near", "bx_att", "ball_vis", "ball_avail",
              "since_seen", "until_seen", "gap_len", "ball_speed", "ball_dv", "ball_dspd",
              "ball_turn", "margin", "margin_goal", "margin_touch", "car_max", "car_sum",
              "car_top"}
SHAPE_BASES = {"cnt", "cx_att", "cy_abs", "sx", "sy", "q10", "q90", "spd_mean", "still",
               "fwd_vel", "all_spd_mean", "all_still", "n_vis"}
SWAP_STATE = np.array([A, H, D])  # reorders [p_H, p_A, p_D] when the teams are swapped
# Boundary classes: 0 no change near, 1 + label for a change to that label.
SWAP_BOUNDARY = np.array([0, 1 + A, 1 + H, 1 + D])
LOCAL_CONTEXTS = ("0", "w5", "past", "fut", "chg", "p5", "f5", "c5", "p12", "f12", "c12")


def base_of(name):
    return name.split(".")[1]


def context_of(name):
    return name.split(".")[-1]


EXPERTS = (
    ("ball", STATE, lambda name: base_of(name) in BALL_BASES),
    ("shape", STATE, lambda name: base_of(name) in SHAPE_BASES),
    ("dead", DEAD, lambda name: True),
    # Boundaries are local events: raw values, the shortest window and the before/after contrasts.
    ("transition", BOUNDARY, lambda name: context_of(name) in LOCAL_CONTEXTS),
)


class Expert(object):
    def __init__(self, name, target, select, params):
        self.name, self.target, self.params = name, target, params
        self._select = select
        self.columns = None
        self.model = None

    def outputs(self):
        if self.target == STATE:
            return [self.name + ".H", self.name + ".A", self.name + ".D"]
        if self.target == BOUNDARY:  # P(a change to H / A / D is near)
            return [self.name + ".BH", self.name + ".BA", self.name + ".BD"]
        return [self.name + ".D"]

    def fit(self, X, names, y_state, y_boundary):
        self.columns = np.array([i for i, n in enumerate(names) if self._select(n)])
        if self.target == STATE:
            target = y_state
        elif self.target == DEAD:
            target = (y_state == D).astype(np.int8)
        else:
            target = y_boundary
        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(X[:, self.columns], target)
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X[:, self.columns])
        if self.target == DEAD:
            return proba[:, list(self.model.classes_).index(1)][:, None]
        # STATE classes are H, A, D = 0, 1, 2; BOUNDARY classes 1..3 are changes to H, A, D.
        shift = 0 if self.target == STATE else 1
        out = np.zeros((len(X), 3))
        for j, cls in enumerate(self.model.classes_):
            if int(cls) >= shift:
                out[:, int(cls) - shift] = proba[:, j]
        return out


class ExpertBank(object):
    """All experts, trained on both team orders and evaluated symmetrically."""

    def __init__(self, params=None):
        params = params or config.HGB
        self.experts = [Expert(name, target, select, params)
                        for name, target, select in EXPERTS]
        self.names = None

    def outputs(self):
        return [o for e in self.experts for o in e.outputs()]

    def swap_index(self):
        """Column permutation that maps outputs for order (0, 1) to order (1, 0)."""
        index, offset = [], 0
        for e in self.experts:
            width = len(e.outputs())
            index.extend(offset + SWAP_STATE if e.target != DEAD else [offset])
            offset += width
        return np.array(index)

    def fit(self, table):
        """table: dict from pipeline.build_table -- X, names, y_state, y_boundary, is_stride."""
        self.names = table["names"]
        for e in self.experts:
            rows = slice(None) if e.target == BOUNDARY else table["is_stride"]
            e.fit(table["X"][rows], self.names, table["y_state"][rows],
                  table["y_boundary"][rows])
        return self

    def predict(self, features, rows, chunk=50000):
        """(len(rows), n_outputs), the average of both team orders mapped back to order (0, 1)."""
        swap = self.swap_index()
        out = np.empty((len(rows), len(self.outputs())))
        for low in range(0, len(rows), chunk):
            part = rows[low:low + chunk]
            ab = features.matrix((0, 1), part, self.names)
            ba = features.matrix((1, 0), part, self.names)
            p_ab = np.hstack([e.predict(ab) for e in self.experts])
            p_ba = np.hstack([e.predict(ba) for e in self.experts])
            out[low:low + len(part)] = 0.5 * (p_ab + p_ba[:, swap])
        return out
