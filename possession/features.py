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

Touches get series of their own: per team, how hard the ball changed velocity next to one of its
players, and the frames since that team last touched the ball and until it next does. Possession
is who touched it last, and the since/until clocks carry that across seconds. So does reach: how
soon each team's quickest player can get to the ball, now and where it will be, which says who
wins a loose ball or cuts out a pass.
"""

import warnings

import numpy as np

from possession import config
from possession.tracking import period_blocks

# Params
# Fitted ball kinematics (_fitted), one series of each per window in config.FIT_HALVES.
FIT_TEAM = tuple("fit_touch{}".format(half) for half in config.FIT_HALVES)
FIT_SHARED = tuple("fit_{}{}".format(kind, half) for half in config.FIT_HALVES
                   for kind in ("dv", "dspd"))
TEAM_CONTEXT = ("d_ball", "n_r2", "near", "cx_att", "sx", "sy", "spd_mean", "still", "fwd_vel",
                "q10", "q90")
TEAM_TRANSITION = ("near", "d_ball", "cx_att", "fwd_vel", "q90", "sx", "reach")
SHARED_CONTEXT = ("ball_vis", "ball_speed", "ball_dv", "margin", "all_spd_mean", "all_still")
SHARED_TRANSITION = ("all_spd_mean", "ball_vis", "margin")
# A touch shows as the ball speeding up or slowing down next to one team's player, so these get
# the short before/after windows as well.
TEAM_SHORT = ("near", "d_ball", "n_r2", "touch", "reach", "acc_max", "acc_near") + FIT_TEAM
SHARED_LOOSE = ("contest",)
SHARED_SHORT = ("ball_speed", "ball_dspd", "ball_vis")
DIFF_BASES = ("d_ball", "n_r2", "near", "cx_att", "fwd_vel", "q90", "sx", "spd_mean", "car_max",
              "car_sum", "touch", "since_touch", "until_touch", "reach", "reach10", "reach25",
              "acc_max", "acc_near", "tag_att", "tag_def", "tag_set", "gap_last", "gap_next"
              ) + FIT_TEAM
# Series added later by the carrier model (carrier.py), and the contexts they get.
CARRIER_TEAM = ("car_max", "car_sum")
CARRIER_SHARED = ("car_top",)
CARRIER_CONTEXTS = ((CARRIER_TEAM, CARRIER_TEAM, CARRIER_TEAM), (CARRIER_SHARED, CARRIER_SHARED, ()))

TEAM_BASES = ("cnt", "cx_att", "cy_abs", "sx", "sy", "q10", "q90", "spd_mean", "still",
              "fwd_vel", "d_ball", "d_ball2", "n_r2", "n_r5", "near", "bx_att", "touch",
              "since_touch", "until_touch", "reach", "reach10", "reach25", "acc_max", "acc_near"
              ) + FIT_TEAM
SHARED_BASES = ("ball_vis", "ball_avail", "since_seen", "until_seen", "gap_len", "ball_speed",
                "ball_dv", "ball_dspd", "ball_turn", "margin", "margin_goal", "margin_touch",
                "all_spd_mean", "all_still", "n_vis") + FIT_SHARED


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


def extend(feature_set, period, team_series, shared_series, contexts=CARRIER_CONTEXTS):
    """Add series computed outside this module -- the carrier model's, Smart Tagging's -- with
    windows like the built-in ones, each kept inside its period. contexts: (team, shared), each
    (context keys, transition keys, short keys) as _contexts() takes them."""
    n = feature_set.n_frames
    team_keys, shared_keys = contexts
    for start, stop, _ in period_blocks(period):
        for k in (0, 1):
            block = {key: values[start:stop] for key, values in team_series[k].items()}
            _put(feature_set.team[k], _contexts(block, *team_keys), start, stop, n)
        block = {key: values[start:stop] for key, values in shared_series.items()}
        _put(feature_set.shared, _contexts(block, *shared_keys), start, stop, n)
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
        f["q10"], f["q90"] = _row_percentiles(x_att, (10, 90))
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
    if config.TOUCH_FEATURES:
        _touches(team, ball, fps)
    if config.REACH_FEATURES:
        _reach(team, players, velocity, visible, tracking.team, ball, fps)
    if config.DYNAMICS_FEATURES:
        _dynamics(team, players, distance, tracking.team, fps)
    if config.GAP_FEATURES:
        _gap_ends(team, raw_ball, ball_vis, x, y, visible, tracking.team)

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
    if config.FIT_FEATURES:
        _fitted(team, shared, raw_ball, fps)
    if config.GAP_FEATURES:
        shared["gap_disp"] = _gap_displacement(raw_ball, ball_vis)
    if config.CONTEST_FEATURES:
        both = ((team[0]["d_ball"] < config.CONTEST_RADIUS)
                & (team[1]["d_ball"] < config.CONTEST_RADIUS))
        shared["contest"] = np.where(ball_avail, both.astype(np.float64), np.nan)
        shared["contest_run"] = np.where(both, np.minimum(_run_lengths(both), config.CAP_FRAMES),
                                         0.0)

    out_team = [_contexts(team[k], TEAM_CONTEXT, TEAM_TRANSITION, TEAM_SHORT) for k in (0, 1)]
    return out_team, _contexts(shared, SHARED_CONTEXT, SHARED_TRANSITION,
                               SHARED_SHORT + SHARED_LOOSE)


def _touches(team, ball, fps):
    """Per team: touch strength (m/s of velocity change, 0 when the other team is nearer or
    nobody is in reach), and the frames since and until that team's last and next touch."""
    half = config.TOUCH_HALF
    jump = np.hypot(*((_shift(ball, -half) - ball) - (ball - _shift(ball, half))).T) * fps / half
    for k in (0, 1):
        mine, other = team[k]["d_ball"], team[1 - k]["d_ball"]
        reach = (mine < config.TOUCH_RADIUS) & ~(other < mine)
        touch = np.where(np.isnan(jump), np.nan, np.where(reach, jump, 0.0))
        since, until, _ = _gap_clocks(np.nan_to_num(touch) > config.TOUCH_DV)
        team[k]["touch"], team[k]["since_touch"], team[k]["until_touch"] = touch, since, until


def _fitted(team, shared, raw_ball, fps):
    """Velocity change between straight-line fits before and after each frame (_fitted_velocity),
    per window in FIT_HALVES: its size and the speed gained (> 0 kicked, < 0 received), and per
    team that size where one of its players is in reach and the other team's are not nearer --
    the touch series, with the jitter averaged out rather than doubled."""
    for half in config.FIT_HALVES:
        before, after = _fitted_velocity(raw_ball, half)
        change = np.hypot(*((after - before) * fps).T)
        shared["fit_dv{}".format(half)] = change
        shared["fit_dspd{}".format(half)] = (np.hypot(*after.T) - np.hypot(*before.T)) * fps
        for k in (0, 1):
            mine, other = team[k]["d_ball"], team[1 - k]["d_ball"]
            reach = (mine < config.TOUCH_RADIUS) & ~(other < mine)
            team[k]["fit_touch{}".format(half)] = np.where(np.isnan(change), np.nan,
                                                           np.where(reach, change, 0.0))


def _fitted_velocity(ball, half):
    """(before, after): least-squares ball velocity, in metres per frame, over the half frames up
    to each frame and the half frames from it, from the frames where the ball was seen. NaN where
    a window holds fewer than three sightings."""
    seen = ~np.isnan(ball[:, 0])
    weight = seen.astype(np.float64)
    position = np.where(seen[:, None], ball, 0.0)
    out = []
    for offsets in (range(-half, 1), range(0, half + 1)):
        s0, s1, s2 = (np.zeros(len(ball)) for _ in range(3))
        sx, sux = np.zeros_like(position), np.zeros_like(position)
        for u in offsets:  # sums over the window of w, w u, w u^2, w x, w u x, at t + u
            w = np.nan_to_num(_shift(weight, -u))
            x = np.nan_to_num(_shift(position, -u))
            s0 += w
            s1 += u * w
            s2 += u * u * w
            sx += x
            sux += u * x
        det = s0 * s2 - s1 * s1
        with np.errstate(invalid="ignore", divide="ignore"):
            velocity = (s0[:, None] * sux - s1[:, None] * sx) / det[:, None]
        velocity[(s0 < 3) | (det <= 0)] = np.nan
        out.append(velocity)
    return out


def _sightings(seen):
    """(last, next): per frame, the index of the latest sighting at or before it and of the
    earliest at or after it; -1 and n where there is none."""
    n = len(seen)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    return last, following


def _gap_ends(team, raw_ball, seen, x, y, visible, slot_team):
    """Per team: its nearest player's distance to the ball at the last sighting at or before
    each frame (gap_last) and at the next one at or after it (gap_next). Equal to the distance now
    on frames where the ball is seen."""
    n = len(seen)
    last, following = _sightings(seen)
    distance = np.hypot(x - raw_ball[:, None, 0], y - raw_ball[:, None, 1])
    for k in (0, 1):
        mine = visible & (slot_team == k)[None, :]
        nearest = np.nanmin(np.where(mine, distance, np.nan), axis=1)
        team[k]["gap_last"] = np.where(last >= 0, nearest[np.clip(last, 0, n - 1)], np.nan)
        team[k]["gap_next"] = np.where(following < n, nearest[np.clip(following, 0, n - 1)],
                                       np.nan)


def _gap_displacement(raw_ball, seen):
    """How far the ball moved between the sightings either side of each frame: 0 when seen."""
    n = len(seen)
    last, following = _sightings(seen)
    both = (last >= 0) & (following < n)
    a, b = raw_ball[np.clip(last, 0, n - 1)], raw_ball[np.clip(following, 0, n - 1)]
    return np.where(both, np.hypot(*(b - a).T), np.nan)


def _run_lengths(mask):
    """Per frame: the length of the run of equal values it sits in."""
    edge = np.flatnonzero(np.diff(np.concatenate(([-1], mask.astype(np.int8), [-1]))) != 0)
    return np.repeat(np.diff(edge), np.diff(edge)).astype(np.float64)


def _dynamics(team, players, distance, slot_team, fps):
    """Per team: the largest acceleration of any of its players (m/s^2, braking included), and
    that of its player nearest the ball -- a receiver brakes, a passer's team-mates set off."""
    half = config.DYNAMICS_HALF
    velocity = _centred_diff(players, half) * fps
    accel = _centred_diff(velocity, half) * fps
    magnitude = np.hypot(accel[..., 0], accel[..., 1])
    for k in (0, 1):
        mine = (slot_team == k)[None, :] & ~np.isnan(magnitude)
        team[k]["acc_max"] = np.nanmax(np.where(mine, magnitude, np.nan), axis=1)
        reach = np.where(mine & ~np.isnan(distance), distance, np.inf)
        nearest = np.argmin(reach, axis=1)
        rows = np.arange(len(reach))
        team[k]["acc_near"] = np.where(np.isfinite(reach[rows, nearest]),
                                       magnitude[rows, nearest], np.nan)


def _reach(team, players, velocity, visible, slot_team, ball, fps):
    """Per team: seconds its quickest player needs to reach the ball -- reaction time, then
    straight there at REACH_SPEED from where his velocity has carried him by then. For the ball's
    position k frames later, the k frames are subtracted: below 0 means there before the ball."""
    reaction = config.REACH_REACTION
    start = players + np.nan_to_num(velocity) * reaction
    for ahead in (0,) + tuple(config.REACH_AHEAD):
        target = _shift(ball, -ahead) if ahead else ball
        distance = np.hypot(start[..., 0] - target[:, None, 0], start[..., 1] - target[:, None, 1])
        seconds = reaction + distance / config.REACH_SPEED - ahead / fps
        for k in (0, 1):
            mine = visible & (slot_team == k)[None, :]
            team[k]["reach{}".format(ahead or "")] = np.nanmin(np.where(mine, seconds, np.nan),
                                                               axis=1)


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


def _row_percentiles(values, percents):
    """np.nanpercentile(values, percents, axis=1), bit for bit, about 25x faster: numpy's NaN
    version loops over rows in Python. One sort, then numpy's own "linear" interpolation."""
    ordered = np.sort(values, axis=1)  # NaN sorts last
    count = (~np.isnan(values)).sum(axis=1)
    rows = np.arange(len(values))
    top = np.maximum(count - 1, 0)
    out = []
    for q in np.true_divide(percents, 100):
        virtual = (count - 1) * q
        below = np.floor(virtual)
        gamma = virtual - below
        low = np.clip(below.astype(np.int64), 0, top)
        a, b = ordered[rows, low], ordered[rows, np.minimum(low + 1, top)]
        diff = b - a
        value = np.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)
        out.append(np.where(count > 0, value, np.nan))
    return out


def _gap_clocks(seen):
    """Frames since the ball was last seen, until it is next seen, and the length of the gap the
    frame sits in -- all capped at CAP_FRAMES, all 0 on frames where the ball is seen. Works for
    any per-frame event, not only sightings."""
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
