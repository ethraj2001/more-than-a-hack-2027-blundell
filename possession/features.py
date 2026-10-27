"""Per-frame features: invariant to track slots, symmetric between the two teams.

Two properties are built in rather than hoped for:

  * Slot invariance. Every team feature is a statistic over that team's visible players -- a
    centroid, a spread, a nearest distance -- never a single slot's trajectory. The ATD's slots
    are tracklets, not people (DATA.md), so nothing may depend on which slot is which.
  * Team symmetry. Features are stored per team and assembled in an order: "A" is one team and
    "B" the other. matrix((0, 1)) puts home in A; matrix((1, 0)) swaps them. Training on both
    orders with H and A swapped, and averaging both at prediction, makes the model incapable of
    learning "home tends to have the ball" from DFL games -- a bias that would not transfer.

Pitch direction is read off the data per period -- the team whose deepest player sits nearer
x = 0 defends that end -- so no constant says who attacks where.

Column names: "A.<base>.<context>", "B.<base>.<context>", "AmB.<base>.<context>" (A minus B),
"S.<base>.<context>" (shared). Contexts: "0" raw, "w5"/"w25"/"w75" centred means, "past"/"fut"
means over the second before and after, "chg" = fut - past, and for the few series that time a
touch "p5"/"f5"/"c5" and "p12"/"f12"/"c12", the same over 0.2 s and 0.5 s.
"""

import warnings

import numpy as np

from possession import config
from possession.tracking import period_blocks

# Params
TEAM_CONTEXT = ("d_ball", "n_r2", "near", "cx_att", "sx", "sy", "spd_mean", "still", "fwd_vel",
                "q10", "q90")
TEAM_TRANSITION = ("near", "d_ball", "cx_att", "fwd_vel", "q90", "sx")
SHARED_CONTEXT = ("ball_vis", "ball_speed", "ball_dv", "margin", "all_spd_mean", "all_still")
SHARED_TRANSITION = ("all_spd_mean", "ball_vis", "margin")
# A touch shows as the ball speeding up or slowing down next to one team's player, so these get
# the short before/after windows as well.
TEAM_SHORT = ("near", "d_ball", "n_r2")
SHARED_SHORT = ("ball_speed", "ball_dspd", "ball_vis")
DIFF_BASES = ("d_ball", "n_r2", "near", "cx_att", "fwd_vel", "q90", "sx", "spd_mean", "car_max",
              "car_sum")
# Series added later by the carrier model (carrier.py), and the contexts they get.
CARRIER_TEAM = ("car_max", "car_sum")
CARRIER_SHARED = ("car_top",)

TEAM_BASES = ("cnt", "cx_att", "cy_abs", "sx", "sy", "q10", "q90", "spd_mean", "still",
              "fwd_vel", "d_ball", "d_ball2", "n_r2", "n_r5", "near", "bx_att")
SHARED_BASES = ("ball_vis", "ball_avail", "since_seen", "until_seen", "gap_len", "ball_speed",
                "ball_dv", "ball_dspd", "ball_turn", "margin", "margin_goal", "margin_touch",
                "all_spd_mean", "all_still", "n_vis")


class FeatureSet(object):
    """Feature series for one match, assembled into matrices on demand."""

    def __init__(self, team, shared, n_frames):
        self.team = team      # [dict, dict]: key "<base>.<context>" -> (T,) float32
        self.shared = shared  # dict
        self.n_frames = n_frames

    def names(self):
        team_keys = sorted(self.team[0])
        return (["A." + k for k in team_keys] + ["B." + k for k in team_keys]
                + ["AmB." + k for k in team_keys if k.split(".")[0] in DIFF_BASES]
                + ["S." + k for k in sorted(self.shared)])

    def column(self, name, order=(0, 1)):
        prefix, key = name.split(".", 1)
        if prefix == "A":
            return self.team[order[0]][key]
        if prefix == "B":
            return self.team[order[1]][key]
        if prefix == "AmB":
            return self.team[order[0]][key] - self.team[order[1]][key]
        if prefix == "S":
            return self.shared[key]
        raise KeyError(name)

    def matrix(self, order=(0, 1), rows=None, names=None):
        """(len(rows), len(names)) float32. rows default to every frame, names to all columns."""
        names = self.names() if names is None else names
        rows = np.arange(self.n_frames) if rows is None else np.asarray(rows)
        out = np.empty((len(rows), len(names)), dtype=np.float32)
        for j, name in enumerate(names):
            out[:, j] = self.column(name, order)[rows]
        return out


def compute(tracking):
    """FeatureSet for every frame; NaN outside play, which the models treat as missing."""
    n = tracking.n_frames
    team = [{}, {}]
    shared = {}
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for start, stop, _ in period_blocks(tracking.period):
            block_team, block_shared = _block(tracking, start, stop)
            for k in (0, 1):
                _put(team[k], block_team[k], start, stop, n)
            _put(shared, block_shared, start, stop, n)
    return FeatureSet(team, shared, n)


def extend(feature_set, period, team_series, shared_series):
    """Add series computed outside this module -- the carrier model's -- with the same windows
    the built-in ones get, each kept inside its period."""
    n = feature_set.n_frames
    for start, stop, _ in period_blocks(period):
        for k in (0, 1):
            block = {key: values[start:stop] for key, values in team_series[k].items()}
            _put(feature_set.team[k], _contexts(block, CARRIER_TEAM, CARRIER_TEAM, CARRIER_TEAM),
                 start, stop, n)
        block = {key: values[start:stop] for key, values in shared_series.items()}
        _put(feature_set.shared, _contexts(block, CARRIER_SHARED, CARRIER_SHARED, ()),
             start, stop, n)
    return feature_set


def _put(store, block, start, stop, n):
    for key, values in block.items():
        if key not in store:
            store[key] = np.full(n, np.nan, dtype=np.float32)
        store[key][start:stop] = values


def _block(tracking, start, stop):
    """Features for one period. Every window stays inside the period."""
    length, width = tracking.pitch
    fps = float(tracking.fps)
    players = tracking.players[start:stop].astype(np.float64)
    x, y = players[..., 0], players[..., 1]
    visible = ~np.isnan(x)

    velocity = _centred_diff(players, config.VELOCITY_HALF) * fps
    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    speed[speed > config.MAX_PLAYER_SPEED] = np.nan

    raw_ball = tracking.ball[start:stop].astype(np.float64)
    ball_vis = ~np.isnan(raw_ball[:, 0])
    ball = _interpolate_gaps(raw_ball, config.BALL_INTERP_MAX_GAP)
    ball_avail = ~np.isnan(ball[:, 0])
    distance = np.hypot(x - ball[:, None, 0], y - ball[:, None, 1])

    directions = _attack_directions(x, tracking.team)
    team = [{}, {}]
    for k in (0, 1):
        mine = visible & (tracking.team == k)[None, :]
        x_att = np.where(mine, x if directions[k] > 0 else length - x, np.nan)
        y_k = np.where(mine, y, np.nan)
        speed_k = np.where(mine, speed, np.nan)
        dist_k = np.where(mine, distance, np.nan)
        sorted_dist = np.sort(dist_k, axis=1)  # NaN sorts last
        n_speed = (~np.isnan(speed_k)).sum(axis=1)

        f = team[k]
        f["cnt"] = mine.sum(axis=1).astype(np.float64)
        f["cx_att"] = np.nanmean(x_att, axis=1)
        f["cy_abs"] = np.abs(np.nanmean(y_k, axis=1) - width / 2.0)
        f["sx"] = np.nanstd(x_att, axis=1)
        f["sy"] = np.nanstd(y_k, axis=1)
        f["q10"], f["q90"] = np.nanpercentile(x_att, (10, 90), axis=1)
        f["spd_mean"] = np.nanmean(speed_k, axis=1)
        f["still"] = np.where(n_speed > 0, (speed_k < config.STILL_SPEED).sum(axis=1)
                              / np.maximum(n_speed, 1), np.nan)
        f["fwd_vel"] = _centred_diff(f["cx_att"], config.VELOCITY_HALF) * fps
        f["d_ball"] = sorted_dist[:, 0]
        f["d_ball2"] = sorted_dist[:, 1] if sorted_dist.shape[1] > 1 else np.nan
        for radius, key in zip(config.NEAR_RADII, ("n_r2", "n_r5")):
            f[key] = np.where(ball_avail, (dist_k < radius).sum(axis=1), np.nan)
        f["bx_att"] = ball[:, 0] if directions[k] > 0 else length - ball[:, 0]

    for k in (0, 1):
        mine, other = team[k]["d_ball"], team[1 - k]["d_ball"]
        near = np.where(np.isnan(other), 1.0, (mine < other).astype(np.float64))
        team[k]["near"] = np.where(np.isnan(mine), np.where(np.isnan(other), np.nan, 0.0), near)

    shared = {}
    shared["ball_vis"] = ball_vis.astype(np.float64)
    shared["ball_avail"] = ball_avail.astype(np.float64)
    shared["since_seen"], shared["until_seen"], shared["gap_len"] = _gap_clocks(ball_vis)
    half = config.VELOCITY_HALF
    before = (ball - _shift(ball, half)) * fps / half
    after = (_shift(ball, -half) - ball) * fps / half
    shared["ball_speed"] = np.hypot(*(_centred_diff(ball, half) * fps).T)
    shared["ball_dv"] = np.hypot(*(after - before).T)
    shared["ball_dspd"] = np.hypot(*after.T) - np.hypot(*before.T)  # > 0 kicked, < 0 received
    norms = np.hypot(*before.T) * np.hypot(*after.T)
    cosine = np.sum(before * after, axis=1) / np.where(norms > 1.0, norms, np.nan)
    shared["ball_turn"] = 1.0 - cosine
    shared["margin_goal"] = np.minimum(ball[:, 0], length - ball[:, 0])
    shared["margin_touch"] = np.minimum(ball[:, 1], width - ball[:, 1])
    shared["margin"] = np.minimum(shared["margin_goal"], shared["margin_touch"])
    n_speed = (~np.isnan(speed)).sum(axis=1)
    shared["all_spd_mean"] = np.nanmean(speed, axis=1)
    shared["all_still"] = np.where(n_speed > 0, (speed < config.STILL_SPEED).sum(axis=1)
                                   / np.maximum(n_speed, 1), np.nan)
    shared["n_vis"] = visible.sum(axis=1).astype(np.float64)

    out_team = [_contexts(team[k], TEAM_CONTEXT, TEAM_TRANSITION, TEAM_SHORT) for k in (0, 1)]
    return out_team, _contexts(shared, SHARED_CONTEXT, SHARED_TRANSITION, SHARED_SHORT)


def _contexts(base, context_keys, transition_keys, short_keys=()):
    """Raw series plus centred, past and future window means, as float32."""
    out = {}
    for key, values in base.items():
        out[key + ".0"] = values
        if key in context_keys:
            for half in config.WINDOWS:
                out["{}.w{}".format(key, half)] = rolling_mean(values, half, half)
        if key in transition_keys:
            past = rolling_mean(values, config.TRANSITION_WINDOW, -1)
            future = rolling_mean(values, -1, config.TRANSITION_WINDOW)
            out[key + ".past"], out[key + ".fut"] = past, future
            out[key + ".chg"] = future - past
        if key in short_keys:
            for half in config.SHORT_WINDOWS:
                past = rolling_mean(values, half, -1)
                future = rolling_mean(values, -1, half)
                out["{}.p{}".format(key, half)] = past
                out["{}.f{}".format(key, half)] = future
                out["{}.c{}".format(key, half)] = future - past
    return {key: np.asarray(values, dtype=np.float32) for key, values in out.items()}


def rolling_mean(values, back, ahead):
    """NaN-ignoring mean over [t - back, t + ahead]. back = -1 means "start after t", ahead = -1
    "stop before t", which is how past-only and future-only windows are expressed."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    valid = ~np.isnan(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(n)
    if back >= 0 and ahead >= 0:
        low, high = index - back, index + ahead + 1
    elif ahead < 0:   # past only: [t - back, t - 1]
        low, high = index - back, index
    else:             # future only: [t + 1, t + ahead]
        low, high = index + 1, index + ahead + 1
    low, high = np.clip(low, 0, n), np.clip(high, 0, n)
    count = counts[high] - counts[low]
    return np.where(count > 0, (sums[high] - sums[low]) / np.maximum(count, 1), np.nan)


def _shift(values, k):
    """values[t - k] at t, NaN where that falls outside the block."""
    out = np.full_like(values, np.nan)
    if k > 0:
        out[k:] = values[:-k]
    elif k < 0:
        out[:k] = values[-k:]
    else:
        out[:] = values
    return out


def _centred_diff(values, k):
    """(values[t + k] - values[t - k]) / 2k along axis 0."""
    return (_shift(values, -k) - _shift(values, k)) / (2.0 * k)


def _interpolate_gaps(ball, max_gap):
    """Linear bridge over missing runs of at most max_gap frames with a sighting on both sides."""
    out = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    edges = np.diff(np.concatenate(([1], seen.astype(np.int8), [1])))
    for first, stop in zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)):
        if first == 0 or stop >= len(ball) or stop - first > max_gap:
            continue
        weights = (np.arange(first, stop) - (first - 1)) / float(stop - (first - 1))
        out[first:stop] = ball[first - 1] + weights[:, None] * (ball[stop] - ball[first - 1])
    return out


def _gap_clocks(seen):
    """Frames since the ball was last seen, until it is next seen, and the length of the gap the
    frame sits in -- all capped at CAP_FRAMES, all 0 on frames where the ball is seen."""
    n = len(seen)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -10 ** 9))
    following = np.minimum.accumulate(np.where(seen, index, 10 ** 9)[::-1])[::-1]
    since = np.minimum(index - last, config.CAP_FRAMES)
    until = np.minimum(following - index, config.CAP_FRAMES)
    gap = np.where(seen, 0, np.minimum(since + until - 1, config.CAP_FRAMES))
    return since.astype(np.float64), until.astype(np.float64), gap.astype(np.float64)


def _attack_directions(x, team):
    """+1 for the team defending x = 0 in this period, -1 for the other.

    Compared on the median over the period of each team's deepest visible player, which is the
    goalkeeper most of the time and a centre-back otherwise -- either way it sits at its own end.
    """
    deepest = []
    for k in (0, 1):
        positions = np.where((team == k)[None, :], x, np.nan)
        deepest.append(np.nanmedian(np.nanmin(positions, axis=1)))
    home_defends_low = not (deepest[0] > deepest[1])
    return (1, -1) if home_defends_low else (-1, 1)
