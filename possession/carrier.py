"""The carrier model: for every visible player, how likely they are the one on the ball.

The ATD loses the ball on 41% of graded frames, and those frames are where the experts are
weakest. But the players have not lost it: they converge on it, press it, slow down to receive
it. This model reads that. For each (frame, visible player) it predicts "this is the player
nearest the ball, within touching distance", from

    where the ball would be if it moved in a straight line between its last and next sightings,
    and how far this player is from that point, and from both sightings
    how long the ball has been, and will stay, unseen
    the player's speed, how many opponents are close and how many are running at them
    where on the pitch the player is

and the per-frame team summaries (most likely carrier of each team, total probability of each
team) become features for the experts like any other.

The target is self-supervised: in a training game the true ball is known, so the noise model
hides it and the true position says who was on it. No possession label is involved, and the same
recipe could be run on any tracking where the ball is sometimes seen (RULES.md allows learning
without human labels on the provided game; this package does not need to).
"""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import config
from possession.features import _attack_directions, _centred_diff
from possession.tracking import period_blocks

# Params
CONTROL_RADIUS = 2.0     # metres; nearest player within this of the true ball is "on it"
NEAR_RADIUS = 3.0        # metres; opponents and team-mates "close" to a player
PRESS_RADIUS = 10.0      # metres; opponents within this can be pressing
PRESS_SPEED = 1.5        # m/s; slower than this is not running at anyone
PRESS_COSINE = 0.7       # velocity within about 45 degrees of the direction to the player
CHUNK = 2000             # frames per pairwise block
NAMES = ("d_interp", "d_last", "d_next", "d_now", "since", "until", "rank", "margin", "speed",
         "n_opp", "n_mate", "d_opp", "press", "x_att", "y_abs", "d_centroid")


class Carrier(object):
    def __init__(self, params=None):
        self.params = dict(params or config.CARRIER_HGB)
        self.model = None

    def fit(self, pairs, stride):
        """pairs: [(noisy Tracking, true ball (T, 2))]. Rows are sampled, not every frame."""
        X, y = [], []
        for noisy, truth in pairs:
            frames = _training_frames(noisy, stride)
            Xg, slots, rows = candidates(noisy, frames)
            X.append(Xg)
            y.append(_targets(noisy, truth, rows, slots))
        self.model = HistGradientBoostingClassifier(**self.params).fit(np.vstack(X),
                                                                       np.concatenate(y))
        return self

    def series(self, tracking):
        """Per-frame summaries: ([team 0 dict, team 1 dict], shared dict), NaN outside play."""
        n = tracking.n_frames
        frames = np.flatnonzero(tracking.period > 0)
        X, slots, rows = candidates(tracking, frames)
        p = self.model.predict_proba(X)[:, list(self.model.classes_).index(1)] if len(X) \
            else np.zeros(0)
        team = [{"car_max": np.full(n, np.nan), "car_sum": np.full(n, np.nan)} for _ in (0, 1)]
        for k in (0, 1):
            mine = tracking.team[slots] == k
            total = np.zeros(n)
            np.add.at(total, rows[mine], p[mine])
            best = np.zeros(n)
            np.maximum.at(best, rows[mine], p[mine])
            team[k]["car_sum"][frames] = total[frames]
            team[k]["car_max"][frames] = best[frames]
        shared = {"car_top": np.fmax(team[0]["car_max"], team[1]["car_max"])}
        return team, shared


def _training_frames(tracking, stride):
    """Every stride-th frame where the ball is lost, every 4*stride-th where it is seen: the
    model matters most where the ball is missing."""
    play = tracking.period > 0
    lost = play & np.isnan(tracking.ball[:, 0])
    seen = play & ~lost
    return np.union1d(np.flatnonzero(lost)[::stride], np.flatnonzero(seen)[::4 * stride])


def _targets(noisy, truth, rows, slots):
    """1 where the slot is the visible player nearest the true ball, within CONTROL_RADIUS."""
    d = np.hypot(noisy.players[rows, slots, 0] - truth[rows, 0],
                 noisy.players[rows, slots, 1] - truth[rows, 1])
    d = np.where(np.isnan(d), np.inf, d)
    order = np.lexsort((d, rows))                       # by frame, then by distance
    first = np.ones(len(rows), dtype=bool)
    first[1:] = rows[order][1:] != rows[order][:-1]
    nearest = np.zeros(len(rows), dtype=bool)
    nearest[order[first]] = True
    return (nearest & (d < CONTROL_RADIUS)).astype(np.int8)


def candidates(tracking, frames):
    """(X, slots, rows): one row per visible player at each of the given frames."""
    out_X, out_slots, out_rows = [], [], []
    frames = np.asarray(frames)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for start, stop, _ in period_blocks(tracking.period):
            inside = frames[(frames >= start) & (frames < stop)]
            if len(inside):
                X, slots, rows = _block(tracking, start, stop, inside - start)
                out_X.append(X)
                out_slots.append(slots)
                out_rows.append(rows + start)
    if not out_X:
        return np.zeros((0, len(NAMES)), np.float32), np.zeros(0, int), np.zeros(0, int)
    return np.vstack(out_X), np.concatenate(out_slots), np.concatenate(out_rows)


def _block(tracking, start, stop, local):
    length, width = tracking.pitch
    players = tracking.players[start:stop].astype(np.float64)
    ball = tracking.ball[start:stop].astype(np.float64)
    n = stop - start
    team = tracking.team
    visible = ~np.isnan(players[..., 0])

    # Ball: last and next sighting, and the straight line between them.
    seen = ~np.isnan(ball[:, 0])
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    has_last, has_next = last >= 0, following < n
    last_pos = np.where(has_last[:, None], ball[np.clip(last, 0, n - 1)], np.nan)
    next_pos = np.where(has_next[:, None], ball[np.clip(following, 0, n - 1)], np.nan)
    span = np.maximum(following - last, 1)
    weight = np.clip((index - last) / span, 0, 1)[:, None]
    interp = np.where((has_last & has_next)[:, None], last_pos + weight * (next_pos - last_pos),
                      np.where(has_last[:, None], last_pos, next_pos))
    now = np.where(seen[:, None], ball, np.nan)
    since = np.where(has_last, np.minimum(index - last, config.CAP_FRAMES), config.CAP_FRAMES)
    until = np.where(has_next, np.minimum(following - index, config.CAP_FRAMES),
                     config.CAP_FRAMES)

    velocity = _centred_diff(players, config.VELOCITY_HALF) * float(tracking.fps)
    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    speed[speed > config.MAX_PLAYER_SPEED] = np.nan
    directions = _attack_directions(players[..., 0], team)
    x_att = np.where((directions[0] > 0) == (team == 0)[None, :], players[..., 0],
                     length - players[..., 0])

    def distance_to(point):
        return np.hypot(players[..., 0] - point[:, None, 0], players[..., 1] - point[:, None, 1])

    d_interp = distance_to(interp)
    ranked = np.where(visible, d_interp, np.inf)
    rank = np.argsort(np.argsort(ranked, axis=1), axis=1).astype(np.float64)
    sorted_d = np.sort(ranked, axis=1)
    best_other = np.where(rank == 0, sorted_d[:, 1:2] if sorted_d.shape[1] > 1 else np.inf,
                          sorted_d[:, :1])
    margin = np.where(np.isfinite(best_other), d_interp - best_other, np.nan)

    centroid = np.stack([np.nanmean(np.where((team == k)[None, :, None] & visible[..., None],
                                             players, np.nan), axis=1) for k in (0, 1)], axis=1)
    own = centroid[:, team]  # (n, slots, 2): each slot's own team centroid
    d_centroid = np.hypot(players[..., 0] - own[..., 0], players[..., 1] - own[..., 1])

    position, slot_cols = np.nonzero(visible[local])
    frame_rows = local[position]
    pair = _pairwise(players, velocity, team, visible, local)

    columns = {
        "d_interp": d_interp[frame_rows, slot_cols],
        "d_last": distance_to(last_pos)[frame_rows, slot_cols],
        "d_next": distance_to(next_pos)[frame_rows, slot_cols],
        "d_now": distance_to(now)[frame_rows, slot_cols],
        "since": since[frame_rows].astype(np.float64),
        "until": until[frame_rows].astype(np.float64),
        "rank": rank[frame_rows, slot_cols],
        "margin": margin[frame_rows, slot_cols],
        "speed": speed[frame_rows, slot_cols],
        "x_att": x_att[frame_rows, slot_cols],
        "y_abs": np.abs(players[frame_rows, slot_cols, 1] - width / 2.0),
        "d_centroid": d_centroid[frame_rows, slot_cols],
    }
    for key in ("n_opp", "n_mate", "d_opp", "press"):
        columns[key] = pair[key][position, slot_cols]
    X = np.column_stack([columns[name] for name in NAMES]).astype(np.float32)
    return X, slot_cols, frame_rows


def _pairwise(players, velocity, team, visible, local):
    """Neighbourhood counts per (frame in local, slot), computed in chunks of frames."""
    n_slots = players.shape[1]
    opponent = team[:, None] != team[None, :]
    mate = ~opponent & ~np.eye(n_slots, dtype=bool)
    out = {key: np.full((len(local), n_slots), np.nan) for key in ("n_opp", "n_mate", "d_opp",
                                                                   "press")}
    for low in range(0, len(local), CHUNK):
        frames = local[low:low + CHUNK]
        pos, vel, vis = players[frames], velocity[frames], visible[frames]
        # offset[f, i, j] = position of j minus position of i
        offset = pos[:, None, :, :] - pos[:, :, None, :]
        dist = np.hypot(offset[..., 0], offset[..., 1])
        both = vis[:, :, None] & vis[:, None, :]
        dist = np.where(both, dist, np.inf)
        part = slice(low, low + len(frames))
        out["n_opp"][part] = ((dist < NEAR_RADIUS) & opponent).sum(axis=2)
        out["n_mate"][part] = ((dist < NEAR_RADIUS) & mate).sum(axis=2)
        d_opp = np.where(opponent, dist, np.inf).min(axis=2)
        out["d_opp"][part] = np.where(np.isfinite(d_opp), d_opp, np.nan)
        # j presses i: j is an opponent within PRESS_RADIUS, running, towards i.
        speed_j = np.hypot(vel[..., 0], vel[..., 1])[:, None, :]
        towards = -(offset[..., 0] * vel[:, None, :, 0] + offset[..., 1] * vel[:, None, :, 1])
        cosine = towards / np.maximum(dist * speed_j, 1e-9)
        pressing = (opponent & (dist < PRESS_RADIUS) & (speed_j > PRESS_SPEED)
                    & (cosine > PRESS_COSINE))
        out["press"][part] = pressing.sum(axis=2)
    return out
