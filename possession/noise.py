"""Make clean tracking look like the provided automatic tracking, and measure what "like" means.

The DFL games are corrected tracking: the ball is never missing and all 22 players are always
there. The provided ATD is automatic: the ball is missing on 41% of graded frames, tracks drop in
and out, and positions jitter. A model trained on the clean version would lean on a ball that
the provided game often does not have, so every training game is corrupted first.

measure() reads only the provided ATD's positions -- how often things are missing and for how long
-- never a label, so it is unsupervised and allowed (RULES.md: inspecting the data to design
features is fine). Jitter is estimated from the positions too: at 25 fps a real player's
acceleration moves the second difference x[t+1] - 2 x[t] + x[t-1] by millimetres, so its spread is
almost all measurement noise, sqrt(6) times the per-frame jitter for white noise. A tracker that
smooths its output hides some of that noise, so the estimate is a floor, and is reported as one.
It also counts frames where a team has more than eleven tracks at once -- referees or duplicate
tracks filed under a team, which corrupt every team statistic.

Not simulated: identity switches and fragmented slots. Every feature is a statistic over a team's
visible players, never a per-slot trajectory, so which slot a player is in cannot matter -- and
noise that cannot change a feature is not worth simulating.
"""

import json
import os
import warnings

import numpy as np
from scipy.signal import butter, filtfilt

from possession import config
from possession.tracking import period_blocks

# Params
PLAYERS_ON_PITCH = 22
# Ball losses are likelier in a crowd: losing the ball starts this many times more often when four
# or more players are within CROWD_RADIUS of it. An assumption, stated as one -- DATA.md says the
# ball "goes missing disproportionately on the frames that matter", not by how much.
CROWD_BIAS = 2.0
CROWD_RADIUS = 3.0
# Implied ball speed above which consecutive detections are treated as a false detection.
OUTLIER_SPEED = 45.0
TEAM_SIZE = 11
MAD_TO_SD = 1.4826  # median absolute deviation -> standard deviation, for normal noise
# Path shape. A tracker that joins its detections with straight lines leaves most frames exactly
# on a line and puts its velocity changes in kinks; one that smooths spreads them over several
# frames. Both read as near-zero jitter, so the profile also records the share of frames on a
# straight line (second difference under STRAIGHT_M) and how sharp accelerations are when read
# over each of SHAPE_LAGS frames -- what the copies' paths have to match for touches to look alike.
STRAIGHT_M = 0.005
SHAPE_LAGS = (1, 2, 4, 8, 16)
GAP_QUANTILES = (10, 25, 50, 75, 90)
TAIL_PERCENTS = (50, 90, 99, 99.5, 99.9)
# calibrate_paths: the smoothers it chooses from, matched on accelerations over every lag -- a
# Gaussian (sd in frames), or a zero-phase Butterworth low-pass of LOWPASS_ORDER (cutoff in Hz),
# whose sharper cut keeps the slow accelerations that a Gaussian with the same effect over a few
# frames removes. Then the white jitter, as a share of the profile's, whose addition matches the
# straight-line share: the profile's jitter is read off a path that is already curved, so adding all
# of it would roughen the copies past the ATD.
SMOOTHERS = ((("none", 0.0),) + tuple(("gauss", float(sd)) for sd in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12))
             + tuple(("lowpass", hz) for hz in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2,
                                            1.5, 2.0, 3.0)))
LOWPASS_ORDER = 4
JITTER_SHARES = tuple(np.linspace(0.0, 1.0, 11))
CALIBRATION_SLOTS = 10  # player tracks a calibration reads: enough for a stable percentile
# calibrate_losses: the mean lengths, in frames, of the ball's losses in play it chooses from.
LOSS_MEANS = (25, 50, 75, 100, 125, 150, 200, 300, 450, 600)
LOSS_DRAWS = 8
# ball_lead: the time shifts (frames) at which the ball is compared with the players, and the
# frames it reads -- the ball moving, a player near it.
LEAD_SHIFTS = tuple(range(-20, 21, 2))
LEAD_SPEED = 3.0   # m/s
LEAD_RADIUS = 3.0  # metres


def measure(tracking):
    """Missing-data statistics of a tracking source, over frames in play."""
    ball_seen, gaps, track_runs, visible = [], [], [], []
    for start, stop, _ in period_blocks(tracking.period):
        seen = ~np.isnan(tracking.ball[start:stop, 0])
        ball_seen.append(seen)
        gaps.extend(_run_lengths(~seen))
        slots_seen = ~np.isnan(tracking.players[start:stop, :, 0])
        visible.append(slots_seen.sum(axis=1))
        for slot in range(slots_seen.shape[1]):
            track_runs.extend(_run_lengths(slots_seen[:, slot]))

    seen = np.concatenate(ball_seen)
    jumps = np.hypot(*np.diff(tracking.ball, axis=0).T) * tracking.fps
    both = ~np.isnan(jumps)
    # Where false detections sit: in play or not (the rate above counts every frame), and in play,
    # inside a track -- a jump with the frames either side seen -- or not.
    anywhere = ~np.isnan(tracking.ball[:, 0])
    tracked = np.zeros(len(jumps), dtype=bool)
    tracked[1:-1] = anywhere[:-3] & anywhere[3:]
    in_play = (tracking.period[:-1] > 0) & (tracking.period[1:] == tracking.period[:-1])
    outlier = both & in_play & (jumps > OUTLIER_SPEED)
    play = tracking.period > 0
    per_team = [(~np.isnan(tracking.players[play][:, tracking.team == k, 0])).sum(axis=1)
                for k in (0, 1)]
    profile = dict(config.DEFAULT_NOISE)
    profile.update({
        "ball_missing_frac": float(1.0 - seen.mean()),
        "ball_gap_mean": float(np.mean(gaps)) if gaps else profile["ball_gap_mean"],
        "players_visible_mean": float(np.concatenate(visible).mean()),
        "player_track_mean": float(np.mean(track_runs)) if track_runs else None,
        "ball_outlier_rate": float(np.mean(jumps[both] > OUTLIER_SPEED)) if both.any() else
        profile["ball_outlier_rate"],
        "team_overfull_frac": float(np.mean((per_team[0] > TEAM_SIZE)
                                            | (per_team[1] > TEAM_SIZE))) if play.any() else 0.0,
        "team_visible_max": [int(v.max()) if len(v) else 0 for v in per_team],
        "measured": True,
        "source": tracking.name,
    })
    player_jitter = jitter_estimate(tracking.players, tracking.period)
    ball_jitter = jitter_estimate(tracking.ball[:, None, :], tracking.period)
    if player_jitter is not None:
        profile["player_jitter_m"] = player_jitter
    if ball_jitter is not None:
        profile["ball_jitter_m"] = ball_jitter
    profile["jitter_measured"] = player_jitter is not None and ball_jitter is not None
    for name, positions in (("ball", tracking.ball[:, None, :]), ("player", tracking.players)):
        straight, accel = path_shape(positions, tracking.period, tracking.fps)
        profile[name + "_straight_frac"] = straight
        profile[name + "_accel_p99"] = accel
        # the tail over one frame: a minority of unsmoothed frames would show above the 99th
        sharp = path_shape(positions, tracking.period, tracking.fps, TAIL_PERCENTS)[1]
        profile[name + "_accel_tail"] = sharp
    # The mean gap mixes stoppages, which lose the ball for their whole length, with losses in play.
    profile["ball_gap_quantiles"] = ([float(q) for q in np.percentile(gaps, GAP_QUANTILES)]
                                     if gaps else None)
    runs = [r for block in ball_seen for r in _run_lengths(block)]
    profile["ball_run_quantiles"] = ([float(q) for q in np.percentile(runs, GAP_QUANTILES)]
                                     if runs else None)
    profile["ball_outlier_rate_play"] = (float(outlier.sum() / (both & in_play).sum())
                                         if (both & in_play).any() else None)
    profile["ball_outlier_in_track"] = (float(np.mean(tracked[outlier])) if outlier.any()
                                        else None)
    profile["ball_lead_m"], profile["ball_lead_frames"] = ball_lead(tracking)
    return profile


def ball_lead(tracking):
    """([median distance per LEAD_SHIFTS], shift at the least): over frames in play whose ball moves
    faster than LEAD_SPEED with a player within LEAD_RADIUS, how far the ball is from the nearest
    player LEAD_SHIFTS frames later. DFL's ball, and a copy smoothed both ways in time, is nearest
    the players 4 frames on (the ball runs ahead of whoever plays it); a tracker that smooths
    forward in time only, as a live one must, lags its ball behind its players and moves that
    earlier. (None, None) without enough such frames."""
    ball = tracking.ball.astype(np.float64)
    players = tracking.players.astype(np.float64)
    reach = max(abs(s) for s in LEAD_SHIFTS)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.hypot(*np.gradient(ball, axis=0).T) * tracking.fps
        near = np.nanmin(np.hypot(players[..., 0] - ball[:, None, 0],
                                  players[..., 1] - ball[:, None, 1]), axis=1)
        rows = []
        for start, stop, _ in period_blocks(tracking.period):
            inner = np.arange(start + reach, stop - reach)
            rows.append(inner[(speed[inner] > LEAD_SPEED) & (near[inner] < LEAD_RADIUS)])
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=int)
        if len(rows) < 1000:
            return None, None
        medians = []
        for shift in LEAD_SHIFTS:
            later = players[rows + shift]
            medians.append(float(np.nanmedian(np.nanmin(np.hypot(
                later[..., 0] - ball[rows, None, 0], later[..., 1] - ball[rows, None, 1]), axis=1))))
    least = int(np.argmin(medians))
    shift = float(LEAD_SHIFTS[least])
    if 0 < least < len(medians) - 1:  # the vertex of the parabola through the least and its neighbours
        left, mid, right = medians[least - 1:least + 2]
        step = LEAD_SHIFTS[1] - LEAD_SHIFTS[0]
        curvature = left - 2 * mid + right
        if curvature > 0:
            shift += 0.5 * step * (left - right) / curvature
    return medians, shift


def path_shape(positions, period, fps, tail=None):
    """(share of frames on a straight line, [99th-percentile acceleration in m/s^2 read over each
    of SHAPE_LAGS frames]) over every track in positions (T, N, 2); (None, None) if too little is
    tracked. With tail, a list of percentiles: those of the acceleration over one frame instead."""
    straight, accel = [], {k: [] for k in SHAPE_LAGS}
    for start, stop, _ in period_blocks(period):
        p = positions[start:stop].astype(np.float64)
        for k in SHAPE_LAGS:
            if len(p) <= 2 * k:
                continue
            d2 = np.hypot(*np.moveaxis(p[2 * k:] - 2 * p[k:-k] + p[:-2 * k], -1, 0))
            d2 = d2[~np.isnan(d2)]
            if k == 1:
                straight.append(d2 < STRAIGHT_M)
            accel[k].append(d2 * (fps / float(k)) ** 2)
    flat = np.concatenate(straight) if straight else np.zeros(0, dtype=bool)
    if len(flat) < 1000:
        return None, None
    if tail is not None:
        return float(flat.mean()), [float(v) for v in np.percentile(np.concatenate(accel[1]), tail)]
    return float(flat.mean()), [float(np.percentile(np.concatenate(accel[k]), 99))
                                for k in SHAPE_LAGS]


def jitter_estimate(positions, period):
    """Per-frame positional noise, metres, from second differences: robust, so identity
    switches and kicks -- rare, huge -- do not move it. None if too little is tracked."""
    second = []
    for start, stop, _ in period_blocks(period):
        p = positions[start:stop].astype(np.float64)
        d2 = p[2:] - 2 * p[1:-1] + p[:-2]
        second.append(d2[~np.isnan(d2)].ravel())
    values = np.concatenate(second) if second else np.zeros(0)
    if len(values) < 1000:
        return None
    mad = np.median(np.abs(values - np.median(values)))
    return float(MAD_TO_SD * mad / np.sqrt(6.0))


def save_profile(profile, path=config.DEFAULT_PROFILE):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
    return path


def load_profile(path=config.DEFAULT_PROFILE):
    """The measured profile if one has been saved, else the documented defaults."""
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    return dict(config.DEFAULT_NOISE)


def corrupt(tracking, profile, rng, dead=None):
    """A noisy copy of clean tracking, matched to a profile. Deterministic given rng's state.
    dead, per frame, marks a labelled game's stoppages, through which the ball is mostly lost."""
    paths = calibrate_paths(profile, tracking, dead) if config.PATH_SMOOTHING else None
    jitter = {"player": profile["player_jitter_m"], "ball": profile["ball_jitter_m"]}
    if paths is not None:
        tracking = tracking.replace(
            players=smooth_paths(tracking.players, tracking.period, tracking.fps,
                                 paths["player"][0]),
            ball=smooth_paths(tracking.ball[:, None, :], tracking.period, tracking.fps,
                              paths["ball"][0])[:, 0])
        jitter = {name: shape[1] for name, shape in paths.items()}
    players = tracking.players.copy()
    ball = tracking.ball.copy()
    n_frames, n_slots = players.shape[:2]

    # Players: each slot alternates visible and missing runs, so that on average
    # players_visible_mean of the 22 are seen, in runs whose length matches the ATD's tracks.
    keep_frac = min(1.0, profile["players_visible_mean"] / PLAYERS_ON_PITCH)
    if keep_frac < 1.0:
        gap_mean = profile.get("player_gap_mean", config.DEFAULT_NOISE["player_gap_mean"])
        track_mean = profile.get("player_track_mean") or gap_mean * keep_frac / (1 - keep_frac)
        for slot in range(n_slots):
            on = _alternating(n_frames, track_mean, gap_mean, keep_frac, rng)
            players[~on, slot] = np.nan
    players += rng.normal(0.0, jitter["player"], players.shape).astype(np.float32)

    # Ball: false detections first (snapped onto a nearby player, the usual tracker mistake),
    # then losses through stoppages, then crowd-biased losses, then jitter. With path smoothing,
    # the false detections are split as the profile measured them in play: single frames inside
    # the tracks (each makes two jumps), the rest as flickers in the losses, added last.
    in_track, flicker = profile["ball_outlier_rate"], 0.0
    if paths is not None:
        rate = profile.get("ball_outlier_rate_play", profile["ball_outlier_rate"])
        share = profile.get("ball_outlier_in_track") or 0.0
        in_track, flicker = rate * share / 2.0, rate * (1.0 - share)
        # A false detection inside a track reads as at least OUTLIER_SPEED * fps over one frame.
        # A tail of one-frame accelerations that stops short of that says the fast jumps the
        # profile counts in its tracks are fast balls, which a smoothed copy has of its own.
        tail = profile.get("ball_accel_tail")
        if tail and tail[-1] < OUTLIER_SPEED * tracking.fps:
            in_track = 0.0
    gap_mean = profile["ball_gap_mean"]
    if dead is not None and config.LIVE_BALL_FIT:
        gap_mean = calibrate_losses(profile, tracking, dead)
    for start, stop, _ in period_blocks(tracking.period):
        segment = ball[start:stop]
        seen = ~np.isnan(segment[:, 0])
        outliers = (np.flatnonzero(seen & (rng.random(stop - start) < in_track))
                    if paths is None or in_track > 0 else np.zeros(0, dtype=int))
        for row in outliers:
            candidates = np.flatnonzero(~np.isnan(tracking.players[start + row, :, 0]))
            if len(candidates):
                pick = candidates[rng.integers(len(candidates))]
                segment[row] = tracking.players[start + row, pick] + rng.normal(0, 1.0, 2)
        crowd = _crowding(tracking.players[start:stop], tracking.ball[start:stop])
        segment[_ball_lost(seen, None if dead is None else dead[start:stop], crowd, profile,
                           gap_mean, rng)] = np.nan
        if any(config.OCCLUSION_RATES):
            near = _near(tracking.players[start:stop], tracking.ball[start:stop])
            segment[_occlusions(~np.isnan(segment[:, 0]), near, rng)] = np.nan
        if paths is not None:
            _flickers(segment, tracking.players[start:stop], flicker, rng)
    ball += rng.normal(0.0, jitter["ball"], ball.shape).astype(np.float32)
    if config.BALL_DRIFT_M > 0:
        ball += _drift(len(ball), tracking.fps, rng)

    if config.TEAM_SWITCH_RATE > 0:  # drawn last, so the rest of the copy is as it was without
        players, team = _misfiled(players, tracking.team, config.TEAM_SWITCH_RATE, rng)
        return tracking.replace(players=players, ball=ball, team=team)
    return tracking.replace(players=players, ball=ball)


def _misfiled(players, team, rate, rng):
    """Identity switches that cross the teams: each tracklet (a slot's run of visible frames) is,
    with chance rate, filed under the other team -- moved to a slot of its own on that side. A
    switch inside a team changes no feature, so only these are made."""
    moved, sides = [], []
    players = players.copy()
    for slot in range(players.shape[1]):
        seen = ~np.isnan(players[:, slot, 0])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], seen.astype(np.int8), [0]))))
        for first, stop in zip(edges[::2], edges[1::2]):
            if rng.random() < rate:
                track = np.full((len(players), 2), np.nan, dtype=players.dtype)
                track[first:stop] = players[first:stop, slot]
                players[first:stop, slot] = np.nan
                moved.append(track)
                sides.append(1 - int(team[slot]))
    if not moved:
        return players, team
    return (np.concatenate([players, np.stack(moved, axis=1)], axis=1),
            np.concatenate([team, np.asarray(sides, dtype=team.dtype)]))


def jittered(tracking, profile, rng):
    """A copy of tracking that is already a tracker's own output (skillcorner.py): the profile's
    jitter and the ball's slow drift, but no losses or false detections -- it has its own, and
    loses the ball about as often as the ATD."""
    players = tracking.players + rng.normal(0.0, profile["player_jitter_m"],
                                            tracking.players.shape).astype(np.float32)
    ball = tracking.ball + rng.normal(0.0, profile["ball_jitter_m"],
                                      tracking.ball.shape).astype(np.float32)
    if config.BALL_DRIFT_M > 0:
        ball += _drift(len(ball), tracking.fps, rng)
    return tracking.replace(players=players, ball=ball)


def _drift(n, fps, rng):
    """(n, 2) slow offsets: white noise low-passed at BALL_DRIFT_HZ, scaled to BALL_DRIFT_M per
    axis -- drawn last, so the rest of a copy is the same with or without it."""
    numerator, denominator = butter(2, config.BALL_DRIFT_HZ / (fps / 2.0))
    wander = filtfilt(numerator, denominator, rng.normal(0.0, 1.0, (n, 2)), axis=0)
    return (wander / wander.std(axis=0) * config.BALL_DRIFT_M).astype(np.float32)


def _flickers(ball, players, rate, rng):
    """False detections outside the tracks, in place: two-frame sightings inside the losses, at
    least the longest of SHAPE_LAGS from any other sighting, each frame on a different random
    player, as many as make rate of the sighted frame pairs jumps. The profile's accelerations over
    long lags, which a sighting near a track would inflate, say they sit that far out."""
    seen = ~np.isnan(ball[:, 0])
    pairs = int(np.sum(seen[1:] & seen[:-1]))
    reach = max(SHAPE_LAGS)
    window = 2 + 2 * reach  # the pair and reach frames either side, all lost
    busy = np.convolve(seen.astype(np.float64), np.ones(window), "valid") > 0.5
    clear = np.flatnonzero(~busy) + reach
    if not len(clear) or rate <= 0:
        return
    rows = rng.choice(clear, size=min(len(clear), rng.binomial(pairs, rate)), replace=False)
    for row in np.sort(rows):
        if not np.all(np.isnan(ball[row - reach:row + 2 + reach, 0])):
            continue  # an earlier flicker sits too near
        for t in (row, row + 1):
            candidates = np.flatnonzero(~np.isnan(players[t, :, 0]))
            if len(candidates):
                ball[t] = players[t, candidates[rng.integers(len(candidates))]] + rng.normal(0, 1.0, 2)


_CALIBRATED = {}


def calibrate_paths(profile, tracking, dead=None):
    """{"ball": (smoother, jitter m), "player": (...)}: the smoothing that makes this clean
    tracking's accelerations match the profile's, and the white jitter that then matches its
    straight-line share; None if the profile has no path shape. The ball's is read on live frames
    when dead marks the stoppages, since the copies lose it through most of them. Positions only; remembered
    per game and profile, since every copy of a game asks again."""
    if not profile.get("ball_accel_p99") or not profile.get("player_accel_p99"):
        return None
    key = (tracking.name, tracking.n_frames, dead is None, repr(sorted(
        (k, v) for k, v in profile.items() if k.endswith(("_accel_p99", "_straight_frac", "_jitter_m")))))
    if key not in _CALIBRATED:
        slots = [s for s in range(tracking.players.shape[1])
                 if (~np.isnan(tracking.players[:, s, 0])).sum() > 1000][:CALIBRATION_SLOTS]
        _CALIBRATED[key] = {
            "ball": _fit_shape(tracking.ball[:, None, :], tracking.period, tracking.fps, dead,
                               profile["ball_accel_p99"], profile.get("ball_straight_frac"),
                               profile["ball_jitter_m"]),
            "player": _fit_shape(tracking.players[:, slots], tracking.period, tracking.fps, None,
                                 profile["player_accel_p99"], profile.get("player_straight_frac"),
                                 profile["player_jitter_m"])}
    return _CALIBRATED[key]


def _fit_shape(positions, period, fps, dead, accel, straight, jitter):
    """(smoother, jitter m): the smoother whose accelerations over SHAPE_LAGS are nearest the
    target's in log terms, then the share of jitter whose addition puts the share of frames on a
    straight line nearest the target's; both read off frames not dead."""
    def shape(smoothed):
        return path_shape(smoothed if dead is None else
                          np.where(dead[:, None, None], np.nan, smoothed), period, fps)
    target = np.log(np.asarray(accel, dtype=np.float64))
    best = None
    for smoother in SMOOTHERS:
        _, got = shape(smooth_paths(positions, period, fps, smoother))
        if got is None:
            return ("none", 0.0), jitter
        error = float(np.sum((np.log(np.maximum(got, 1e-9)) - target) ** 2))
        if best is None or error < best[0]:
            best = (error, smoother)
    smoother = best[1]
    if straight is None:
        return smoother, jitter
    smoothed = smooth_paths(positions, period, fps, smoother)
    rng = np.random.default_rng(config.SEED)
    misses = [abs(shape(smoothed + rng.normal(0.0, share * jitter, smoothed.shape))[0] - straight)
              for share in JITTER_SHARES]
    return smoother, float(JITTER_SHARES[int(np.argmin(misses))] * jitter)


def smooth_paths(positions, period, fps, smoother):
    """(T, N, 2) positions smoothed inside each track's runs of seen frames: smoother is
    ("gauss", sd in frames), ("lowpass", cutoff in Hz) or ("none", 0)."""
    kind, size = smoother
    if kind == "none":
        return positions
    if kind == "gauss":
        kernel = _gaussian(size)
        half = len(kernel) // 2
        def smooth(values):  # normalised at the ends of the run, which may be shorter than kernel
            weight = np.convolve(np.ones(len(values)), kernel)[half:half + len(values)]
            return np.convolve(values, kernel)[half:half + len(values)] / weight
    else:
        numerator, denominator = butter(LOWPASS_ORDER, size / (fps / 2.0))
        pad = 3 * max(len(numerator), len(denominator))
        def smooth(values):
            return filtfilt(numerator, denominator, values, padlen=min(pad, len(values) - 1))
    out = positions.astype(np.float64).copy()
    for start, stop, _ in period_blocks(period):
        for slot in range(positions.shape[1]):
            edges = np.flatnonzero(np.diff(np.concatenate(
                ([0], (~np.isnan(positions[start:stop, slot, 0])).astype(np.int8), [0]))))
            for first, last in zip(edges[::2] + start, edges[1::2] + start):
                if last - first < 4:
                    continue
                for axis in range(2):
                    out[first:last, slot, axis] = smooth(out[first:last, slot, axis])
    return out.astype(np.float32)


def _gaussian(sd):
    half = int(np.ceil(3 * sd))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / float(sd)) ** 2)
    return kernel / kernel.sum()


def _run_lengths(mask):
    """Lengths of the True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist()


def _alternating(n, mean_on, mean_off, on_frac, rng):
    """Boolean mask of alternating geometric runs with the given mean lengths."""
    out = np.empty(n, dtype=bool)
    position, state = 0, bool(rng.random() < on_frac)
    while position < n:
        length = int(rng.geometric(1.0 / max(mean_on if state else mean_off, 1.0)))
        out[position:position + length] = state
        position += length
        state = not state
    return out


def _crowding(players, ball):
    """0..1 per frame: how many players are within CROWD_RADIUS of the true ball, out of four."""
    distance = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        near = (distance < CROWD_RADIUS).sum(axis=1)
    return np.minimum(near, 4) / 4.0


def _near(players, ball):
    """Per frame, the players within CROWD_RADIUS of the true ball."""
    distance = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        return (distance < CROWD_RADIUS).sum(axis=1)


def _occlusions(seen, near, rng):
    """Boolean per frame, True where a short loss hides the ball: after each seen frame, one
    starts with config.OCCLUSION_RATES' chance for 1, 2 or 3+ players near, and lasts a geometric
    number of frames, mean config.OCCLUSION_MEAN."""
    rate = np.r_[0.0, config.OCCLUSION_RATES][np.minimum(near, 3)]
    lost = np.zeros(len(seen), dtype=bool)
    starts = np.flatnonzero(seen & (rng.random(len(seen)) < rate))
    for t, length in zip(starts, rng.geometric(1.0 / config.OCCLUSION_MEAN, len(starts))):
        lost[t + 1:t + 1 + length] = True
    return lost


def _stoppage_losses(dead, rng):
    """Boolean per frame, True where a stoppage has lost the ball: each run of dead frames, with
    probability DEAD_BALL_LOSS, from a geometric number of frames after it starts (mean
    DEAD_BALL_EDGE) until as many before the restart."""
    lost = np.zeros(len(dead), dtype=bool)
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    p_edge = 1.0 / max(config.DEAD_BALL_EDGE, 1.0)
    for first, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() < config.DEAD_BALL_LOSS:
            low, high = first + rng.geometric(p_edge) - 1, end - rng.geometric(p_edge) + 1
            lost[low:max(low, high)] = True
    return lost


def _ball_lost(seen, dead, crowd, profile, gap_mean, rng):
    """Per frame of a period block, True where a copy loses the ball: through the stoppages when
    dead marks them, then in crowd-biased losses of mean length gap_mean. A source whose ball is
    already lost on some frames (seen False; metrica.py) loses only the rest: of every frame, to
    the profile's missing fraction, or of the live frames, to LIVE_BALL_LOSS when that is set and
    the stoppages are known."""
    lost = np.zeros(len(seen), dtype=bool)
    if dead is not None and config.DEAD_BALL_LOSS > 0:
        lost = _stoppage_losses(dead, rng)
        seen = seen & ~lost
    among = None
    if dead is not None and config.LIVE_BALL_LOSS is not None:
        among = ~dead
        already = float(np.mean(~seen[among])) if among.any() else 1.0
        target = config.LIVE_BALL_LOSS
    else:
        already = float(np.mean(~seen))
        target = profile["ball_missing_frac"]
    extra = (target - already) / (1.0 - already) if already < target else 0.0
    return lost | _ball_losses(crowd, extra, gap_mean, rng, among)


_LOSS_FITS = {}


def calibrate_losses(profile, tracking, dead):
    """The mean length, in frames, of the ball's losses in play, from LOSS_MEANS, whose copies'
    gaps and runs of sightings come nearest the profile's quantiles in log terms, averaged over
    LOSS_DRAWS generators of its own (one draw puts a game anywhere from 25 to 200); the profile's
    mean gap if it has no quantiles. Remembered per game and profile."""
    gap_q, run_q = profile.get("ball_gap_quantiles"), profile.get("ball_run_quantiles")
    if not gap_q or not run_q:
        return profile["ball_gap_mean"]
    key = (tracking.name, tracking.n_frames, config.PATH_SMOOTHING, repr((
        gap_q, run_q, profile["ball_missing_frac"], config.DEAD_BALL_LOSS, config.DEAD_BALL_EDGE,
        config.LIVE_BALL_LOSS)))
    if key not in _LOSS_FITS:
        target = np.log(np.asarray(list(gap_q) + list(run_q), dtype=np.float64))
        blocks = [(~np.isnan(tracking.ball[start:stop, 0]), dead[start:stop],
                   _crowding(tracking.players[start:stop], tracking.ball[start:stop]))
                  for start, stop, _ in period_blocks(tracking.period)]
        best = None
        for mean in LOSS_MEANS:
            errors = []
            for draw in range(LOSS_DRAWS):
                rng = np.random.default_rng(config.SEED + draw)
                gaps, runs = [], []
                for seen, block_dead, crowd in blocks:
                    kept = seen & ~_ball_lost(seen, block_dead, crowd, profile, float(mean), rng)
                    gaps.extend(_run_lengths(~kept))
                    runs.extend(_run_lengths(kept))
                if gaps and runs:
                    got = np.log(np.concatenate((np.percentile(gaps, GAP_QUANTILES),
                                                 np.percentile(runs, GAP_QUANTILES))))
                    errors.append(float(np.sum((got - target) ** 2)))
            if errors and (best is None or np.mean(errors) < best[0]):
                best = (float(np.mean(errors)), float(mean))
        _LOSS_FITS[key] = profile["ball_gap_mean"] if best is None else best[1]
    return _LOSS_FITS[key]


def _ball_losses(crowd, missing_frac, gap_mean, rng, among=None):
    """Boolean per frame, True where the ball is lost: a two-state chain whose loss hazard rises
    with crowding, calibrated so the long-run missing fraction matches missing_frac -- over the
    frames marked in among, if given, else over all of them."""
    if missing_frac <= 0:
        return np.zeros(len(crowd), dtype=bool)
    among = np.ones(len(crowd), dtype=bool) if among is None else among
    p_regain = 1.0 / max(gap_mean, 1.0)
    hazard = 1.0 + CROWD_BIAS * crowd
    uniforms = rng.random(len(crowd))
    p_lose = p_regain * missing_frac / (1.0 - missing_frac) / hazard[among].mean()
    lost = _simulate_losses(uniforms, np.clip(p_lose * hazard, 0, 1), p_regain)
    # One correction pass on the same random numbers: crowding is not independent of the chain,
    # so the closed form above lands near the target, not on it.
    realised = lost[among].mean()
    if 0 < realised < 1:
        target_odds = missing_frac / (1.0 - missing_frac)
        p_lose *= target_odds / (realised / (1.0 - realised))
        lost = _simulate_losses(uniforms, np.clip(p_lose * hazard, 0, 1), p_regain)
    return lost


def _simulate_losses(uniforms, p_lose, p_regain):
    lost = np.empty(len(uniforms), dtype=bool)
    state = False
    for t in range(len(uniforms)):
        state = (uniforms[t] >= p_regain) if state else (uniforms[t] < p_lose[t])
        lost[t] = state
    return lost
