"""Kick-offs, found from the tracking alone.

A period's tracking can start before its kick-off. The provided match's second half is graded from
frame 110,808 but kicks off at 113,742 (FAQ.md), and those 2,934 frames are all D: the teams walk
out and line up. Every training game starts at its kick-off, so the model has never seen that.

detect() finds the kick-off of each period without labels and without frame numbers: the first
time, within SEARCH_S of the period's start, that both teams have stood in their own halves for
LINEUP_S -- at most LINEUP_OVER players across the halfway line by more than OVER_M, with at
least LINEUP_PLAYERS tracked -- with no live play before it (the players' median speed above
LIVE_SPEED on at most LIVE_SHARE of the frames). The kick-off is where that line-up breaks, or,
when the tracking sees it, where the ball leaves the centre spot. hold_dead() makes every frame
of a period before its kick-off dead, before the decoder runs (config.KICKOFF_DEAD).
"""

import warnings

import numpy as np

from possession import config
from possession.features import _attack_directions, _centred_diff
from possession.tracking import period_blocks

# Params
SEARCH_S = 300.0       # seconds from a period's start in which a kick-off is looked for
LINEUP_S = 2.0         # seconds both teams must stand in their own halves
LINEUP_PLAYERS = 16    # tracked outfield players and keepers needed to call a line-up
LINEUP_OVER = 1        # players allowed across the halfway line (the kicker stands on it)
OVER_M = 1.0           # metres across it before a player counts as across
LIVE_SPEED = 3.0       # m/s: a median player speed above this is live play, not walking out
LIVE_SHARE = 0.05      # share of frames before the line-up allowed above LIVE_SPEED
CENTRE_M = 2.0         # metres from the centre spot: the ball is placed for the kick-off
LEAVE_M = 1.0          # metres it must move from where it sat to count as kicked
BALL_LOOK_S = 4.0      # seconds before the line-up breaks in which the kick is looked for
PROB_EPS = 1e-4        # the other labels' probability on a frame held dead


def detect(tracking):
    """[(period start, kick-off frame)] for each period whose tracking starts before its
    kick-off; a period that starts at its kick-off, or whose kick-off is not found, is left out."""
    out = []
    for start, stop, _ in period_blocks(tracking.period):
        kick = _kickoff(tracking, start, stop)
        if kick is not None and kick > start:
            out.append((start, kick))
    return out


def hold_dead(probs, tracking):
    """probs (T, 3) with every frame of a period before its kick-off set to dead."""
    out = probs.copy()
    for start, kick in detect(tracking):
        out[start:kick] = (PROB_EPS, PROB_EPS, 1.0 - 2 * PROB_EPS)
    return out


def _kickoff(tracking, start, stop):
    fps = float(tracking.fps)
    length, width = tracking.pitch
    end = min(stop, start + int(SEARCH_S * fps))
    players = tracking.players[start:end].astype(np.float64)
    x = players[..., 0]
    sides = (tracking.team == 0) | (tracking.team == 1)
    visible = ~np.isnan(x) & sides[None, :]
    directions = _attack_directions(tracking.players[start:stop, :, 0].astype(np.float64),
                                    tracking.team)
    across = np.zeros(len(x), dtype=np.int64)
    for k in (0, 1):
        mine = visible & (tracking.team == k)[None, :]
        x_att = x if directions[k] > 0 else length - x
        across += (mine & (x_att > length / 2.0 + OVER_M)).sum(axis=1)
    lined = (visible.sum(axis=1) >= LINEUP_PLAYERS) & (across <= LINEUP_OVER)

    velocity = _centred_diff(players, config.VELOCITY_HALF) * fps
    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    speed[~visible | (speed > config.MAX_PLAYER_SPEED)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        median = np.nanmedian(np.where(visible, speed, np.nan), axis=1)
    live = np.nan_to_num(median) > LIVE_SPEED

    need = int(LINEUP_S * fps)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], lined.astype(np.int8), [0]))))
    for first, after in zip(edges[::2], edges[1::2]):
        settled = live[first:max(first + 1, after - int(BALL_LOOK_S * fps))]
        if after - first < need or settled.mean() > LIVE_SHARE:
            continue  # too short, or play inside the halves rather than a line-up
        if first > 0 and live[:first].mean() > LIVE_SHARE:
            return None  # play was already live: this line-up is a restart after a goal
        if after >= len(lined):
            return None  # still lined up when the search ends
        return start + _ball_kick(tracking.ball[start:end].astype(np.float64), first, after,
                                  length, width, fps)
    return None


def _ball_kick(ball, first, after, length, width, fps):
    """The frame the ball leaves the centre spot, if the tracking sees it there in the
    BALL_LOOK_S before the line-up breaks; else where the line-up breaks."""
    look = max(first, after - int(BALL_LOOK_S * fps))
    centre = np.array([length / 2.0, width / 2.0])
    window = ball[look:after + int(fps)]
    near = np.hypot(*(window - centre).T) < CENTRE_M
    if not near.any():
        return after
    sat = int(np.flatnonzero(near)[-1])
    rest = window[np.flatnonzero(near)[0]]
    moved = np.flatnonzero(np.hypot(*(window[:sat + 1] - rest).T) > LEAVE_M)
    leave = int(moved[0]) if len(moved) else sat
    return look + leave
