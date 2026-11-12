"""The specialists, and the bank that trains and runs them as one.

Two are specialised by what they may look at, the rest by what they are asked:

    ball        H/A/D from ball-relative features only -- strong when the ball is seen
    shape       H/A/D from team shape and movement only -- the ball never enters, so it keeps
                working through the 41% of frames where the tracking has lost it
    dead        alive-or-dead from everything
    transition  "a change TO H / A / D is within BOUNDARY_TOLERANCE frames", from local features
    touch       "a player of team H / A touches the ball within TOUCH_TOLERANCE frames", from
                local features. Trained on copies with the ball hidden, it learns what a touch
                looks like in the players' movement as well as in the ball's
    short       "this frame is inside an H / A possession shorter than SHORT_POSSESSION frames",
                from everything (off by default: it measured no gain)
    risk        "possession passes between the teams RISK_CHANGES or more times within RISK_HALF
                frames", from everything: how fragmented possession is here, for the gate to
                weigh the other experts by and the decoder to set its change cost by
    tags        H/A/D from Smart Tagging alone (tagging.py; on with config.USE_TAGS)

Each is a gradient-boosted tree ensemble. Trees take NaN natively, so a missing ball is missing
rather than zero-filled.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import config
from possession.features import FIT_SHARED, FIT_TEAM
from possession.scoring import A, D, H

# Params
STATE, DEAD, BOUNDARY, SHORT, TOUCH, RISK = "state", "dead", "boundary", "short", "touch", "risk"
TARGETS = (STATE, BOUNDARY, SHORT, TOUCH, RISK)
SINGLE = (DEAD, RISK)  # one output, the same in either team order
BALL_BASES = {"d_ball", "d_ball2", "n_r2", "n_r5", "near", "bx_att", "ball_vis", "ball_avail",
              "since_seen", "until_seen", "gap_len", "ball_speed", "ball_dv", "ball_dspd",
              "ball_turn", "margin", "margin_goal", "margin_touch", "car_max", "car_sum",
              "car_top", "touch", "since_touch", "until_touch", "reach", "reach10", "reach25",
              "acc_near", "gap_last", "gap_next", "gap_disp", "contest", "contest_run"
              } | set(FIT_TEAM) | set(FIT_SHARED)
SHAPE_BASES = {"cnt", "cx_att", "cy_abs", "sx", "sy", "q10", "q90", "spd_mean", "still",
               "fwd_vel", "all_spd_mean", "all_still", "n_vis", "acc_max"}
TAG_BASES = {"tag_att", "tag_def", "tag_set", "tag_unknown", "tag_any", "tag_agree_att",
             "tag_agree_def", "tag_agree_set"}
SWAP_STATE = np.array([A, H, D])  # reorders [p_H, p_A, p_D] when the teams are swapped
# Boundary classes: 0 no change near, 1 + label for a change to that label.
SWAP_BOUNDARY = np.array([0, 1 + A, 1 + H, 1 + D])
# Short-possession and touch classes: 0 none, 1 + H, 1 + A.
SWAP_TEAMS = np.array([0, 1 + A, 1 + H])
LOCAL_CONTEXTS = ("0", "w5", "past", "fut", "chg", "p5", "f5", "c5", "p12", "f12", "c12")


def base_of(name):
    return name.split(".")[1]


def context_of(name):
    return name.split(".")[-1]


def untagged(name):
    """With config.TAGS_ISOLATED, Smart Tagging stays inside the tags expert: the others are the
    same with tags on or off, and the gate alone decides how far to trust the tags."""
    return not config.TAGS_ISOLATED or base_of(name) not in TAG_BASES


EXPERTS = (
    ("ball", STATE, lambda name: base_of(name) in BALL_BASES),
    ("shape", STATE, lambda name: base_of(name) in SHAPE_BASES),
    ("dead", DEAD, untagged),
    # Boundaries are local events: raw values, the shortest window and the before/after contrasts.
    ("transition", BOUNDARY, lambda name: context_of(name) in LOCAL_CONTEXTS and untagged(name)),
    ("touch", TOUCH, lambda name: context_of(name) in LOCAL_CONTEXTS and untagged(name)),
    ("short", SHORT, untagged),
    ("risk", RISK, untagged),
    ("tags", STATE, lambda name: base_of(name) in TAG_BASES),
)
# Experts config can turn off, by name.
SWITCHED = {"short": "SHORT_EXPERT", "touch": "TOUCH_EXPERT", "risk": "RISK_EXPERT",
            "tags": "USE_TAGS"}


class Expert(object):
    def __init__(self, name, target, select, params):
        self.name, self.target, self.params = name, target, params
        self._select = select
        self.columns = None
        self.model = None

    def __getstate__(self):
        """Pickle without the column selector: it is a lambda, and once fit() has turned it into
        column indices nothing needs it."""
        state = dict(self.__dict__)
        state["_select"] = None
        return state

    def outputs(self):
        if self.target == STATE:
            return [self.name + ".H", self.name + ".A", self.name + ".D"]
        if self.target == BOUNDARY:  # P(a change to H / A / D is near)
            return [self.name + ".BH", self.name + ".BA", self.name + ".BD"]
        if self.target == SHORT:  # P(inside a short H / A possession)
            return [self.name + ".SH", self.name + ".SA"]
        if self.target == TOUCH:  # P(a player of H / A touches the ball near this frame)
            return [self.name + ".TH", self.name + ".TA"]
        if self.target == RISK:  # P(possession fragments around this frame)
            return [self.name + ".F"]
        return [self.name + ".D"]

    def fit(self, X, names, targets):
        """targets: {STATE, BOUNDARY, SHORT, TOUCH, RISK} -> class per row of X."""
        self.columns = np.array([i for i, n in enumerate(names) if self._select(n)])
        if self.target == DEAD:
            target = (targets[STATE] == D).astype(np.int8)
        else:
            target = targets[self.target]
        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(X[:, self.columns], target)
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X[:, self.columns])
        if self.target in SINGLE:
            classes = list(self.model.classes_)
            return (proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X)))[:, None]
        # STATE classes are H, A, D = 0, 1, 2; BOUNDARY classes 1..3 are changes to H, A, D;
        # SHORT and TOUCH classes 1..2 are H, A.
        shift = 0 if self.target == STATE else 1
        out = np.zeros((len(X), len(self.outputs())))
        for j, cls in enumerate(self.model.classes_):
            if int(cls) >= shift:
                out[:, int(cls) - shift] = proba[:, j]
        return out


class ExpertBank(object):
    """All experts, trained on both team orders and evaluated symmetrically."""

    def __init__(self, params=None):
        params = params or config.HGB
        self.experts = [Expert(name, target, select, params)
                        for name, target, select in EXPERTS
                        if getattr(config, SWITCHED.get(name, ""), True)]
        self.names = None

    def outputs(self):
        return [o for e in self.experts for o in e.outputs()]

    def swap_index(self):
        """Column permutation that maps outputs for order (0, 1) to order (1, 0)."""
        index, offset = [], 0
        for e in self.experts:
            width = len(e.outputs())
            if e.target in SINGLE:
                index.append(offset)
            else:  # [H, A, D] or [H, A]: swapping the teams swaps the first two
                index.extend(offset + SWAP_STATE[:width])
            offset += width
        return np.array(index)

    def fit(self, table):
        """table: dict from pipeline.build_table -- X, names, y_<target>, is_stride. The experts
        that look for brief events also train on the rows sampled densely around changes."""
        self.names = table["names"]
        for e in self.experts:
            rows = table["is_stride"] if e.target in (STATE,) + SINGLE else slice(None)
            e.fit(table["X"][rows], self.names,
                  {key: table["y_" + key][rows] for key in TARGETS})
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
