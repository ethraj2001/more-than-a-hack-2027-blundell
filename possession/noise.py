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

import numpy as np

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
    return profile


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


def corrupt(tracking, profile, rng):
    """A noisy copy of clean tracking, matched to a profile. Deterministic given rng's state."""
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
    players += rng.normal(0.0, profile["player_jitter_m"], players.shape).astype(np.float32)

    # Ball: false detections first (snapped onto a nearby player, the usual tracker mistake),
    # then crowd-biased losses, then jitter.
    for start, stop, _ in period_blocks(tracking.period):
        segment = ball[start:stop]
        seen = ~np.isnan(segment[:, 0])
        outliers = np.flatnonzero(seen & (rng.random(stop - start) < profile["ball_outlier_rate"]))
        for row in outliers:
            candidates = np.flatnonzero(~np.isnan(tracking.players[start + row, :, 0]))
            if len(candidates):
                pick = candidates[rng.integers(len(candidates))]
                segment[row] = tracking.players[start + row, pick] + rng.normal(0, 1.0, 2)
        crowd = _crowding(tracking.players[start:stop], tracking.ball[start:stop])
        lost = _ball_losses(crowd, profile["ball_missing_frac"], profile["ball_gap_mean"], rng)
        segment[lost] = np.nan
    ball += rng.normal(0.0, profile["ball_jitter_m"], ball.shape).astype(np.float32)

    return tracking.replace(players=players, ball=ball)


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


def _ball_losses(crowd, missing_frac, gap_mean, rng):
    """Boolean per frame, True where the ball is lost: a two-state chain whose loss hazard rises
    with crowding, calibrated so the long-run missing fraction matches missing_frac."""
    if missing_frac <= 0:
        return np.zeros(len(crowd), dtype=bool)
    p_regain = 1.0 / max(gap_mean, 1.0)
    hazard = 1.0 + CROWD_BIAS * crowd
    uniforms = rng.random(len(crowd))
    p_lose = p_regain * missing_frac / (1.0 - missing_frac) / hazard.mean()
    lost = _simulate_losses(uniforms, np.clip(p_lose * hazard, 0, 1), p_regain)
    # One correction pass on the same random numbers: crowding is not independent of the chain,
    # so the closed form above lands near the target, not on it.
    realised = lost.mean()
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
