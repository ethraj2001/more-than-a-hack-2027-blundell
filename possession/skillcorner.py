"""SkillCorner's open broadcast tracking of the two 2019/20 Clasicos, labelled from StatsBomb's
open event data of the same games.

    https://github.com/SkillCorner/opendata   (commit COMMIT, the last with the 2019/20 matches)
    https://github.com/statsbomb/open-data    (commit STATSBOMB_COMMIT)

Why these: FC Barcelona's own style, with and without the ball, in the only open tracking of
Barcelona there is. SkillCorner took them out of the repository's head in 2025; the commit
pinned here still has them, under the MIT licence of that commit. StatsBomb's data is free under
its user agreement, which asks that StatsBomb be credited. Nothing here redistributes either:
download() fetches them into the cache, which is gitignored, as dfl.py and metrica.py do.

The tracking is SkillCorner's automatic broadcast tracking, uncorrected: 10 frames a second,
only what the broadcast shows (13 of the 22 players on average), nothing at all through replays
and close-ups (27% of the frames in play), the ball on about 60% of them. It is resampled to 25
frames a second by linear interpolation between neighbouring frames that both saw the object.
Players SkillCorner identified keep one slot each; the rest, known only by team and track, share
slots per team that their tracks never overlap in.

The labels come from StatsBomb's events, as metrica.py's do from Metrica's (event_labels()):

  * in play, the team of the latest event that plays the ball (OWNING: a pass, receipt, carry,
    shot, dribble, clearance, interception, recovery, keeper's action, or the touch that loses
    it), as metrica.py's rule has it -- so a change of possession sits at the new team's first
    touch. LABELS = "chain" takes StatsBomb's possession team instead, which changes only when
    the other team gains control: about 160 possessions a game to this rule's
    243-265 (DFL's mechanical labels have 370-511, Metrica's event labels 361-400);
  * the ball is dead from the end of the last event on the ball before a restart -- a throw-in,
    corner, free kick, goal kick, kick off or penalty -- to the restart itself, and from the
    start of each half to its kick off and from the last event to its end.

StatsBomb's times run from the start of each half and SkillCorner's clock from the kick off
(the second half from 45:00), but the two clocks are not the same: they differ by -0.4 to -0.8 s
and, in SC2841, jump by about 5 s three times (cuts in the broadcast the tracking was made from).
So each minute's events are moved by the shift that brings them closest to the tracked ball
(clock_shift()): the median distance from an event's location to the ball is then 2-3 m in every
window, where one shift per half left 10 m in SC2841's first half.

Touches -- the touch expert's targets -- come from the events too (event_touches()), since the
ball is lost on four frames in ten.

Measured as training games (not in matches.default()): the seven DFL games plus both Clasicos,
each DFL game held out in turn and Metrica's three by one model of all nine, the blend scored
-0.008 spellF1 (90% interval -0.015 to -0.002), better on 5 of 10 -- Metrica's three +0.001 to
+0.009, the DFL games' -0.042 to +0.009; the mixture alone -0.017 (-0.024 to -0.011), better on
1 of 10. The two Clasicos alone: -0.212 (-0.226 to -0.199), 0 of 10.

    python -m possession download --games SC2841 SC3442     # about 150 MB, once
"""

import json
import os
import shutil
import urllib.request
import warnings

import numpy as np

from possession import config, guard
from possession.scoring import A, D, H, X
from possession.tracking import Tracking

# Params
MATCHES = {
    "SC2841": (2841, 303596),  # FC Barcelona 0-0 Real Madrid, LaLiga, 18 Dec 2019
    "SC3442": (3442, 303470),  # Real Madrid 2-0 FC Barcelona, LaLiga, 1 Mar 2020
}
COMMIT = "1b99d401ddc82e366c5c20d6860878668fd5858f"
STATSBOMB_COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
SKILLCORNER_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/matches/{}/{}.json"
STATSBOMB_URL = "https://raw.githubusercontent.com/statsbomb/open-data/{}/data/events/{}.json"
SKILLCORNER_CACHE = os.path.join(config.CACHE_DIR, "skillcorner")
SOURCE_FPS = 10.0
PERIOD_GAP = 25  # frames of X between periods, as in dfl.py
HOME_GROUPS = ("home team", "home goalkeeper")
AWAY_GROUPS = ("away team", "away goalkeeper")
# Events on the ball, or the foul or offside that stops it; the rest (pressure aside) are
# substitutions, cards, injuries and the like.
ON_BALL = ("Pass", "Ball Receipt*", "Carry", "Shot", "Dribble", "Clearance", "Interception",
           "Ball Recovery", "Goal Keeper", "Duel", "Foul Committed", "Foul Won", "Miscontrol",
           "Block", "Dispossessed", "Offside", "50/50", "Shield", "Error", "Dribbled Past",
           "Pressure", "Own Goal Against", "Own Goal For")
RESTART_PASSES = ("Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off")
RESTART_SHOTS = ("Free Kick", "Penalty")
# Events by which a team owns the ball, for LABELS = "touch"; failed ones are left out (_owning).
OWNING = ("Pass", "Ball Receipt*", "Carry", "Shot", "Dribble", "Clearance", "Interception",
          "Ball Recovery", "Goal Keeper", "Miscontrol", "Dispossessed", "50/50")
LABELS = "touch"
# Events where a team plays the ball, for the touches.
TOUCHING = ("Pass", "Ball Receipt*", "Shot", "Clearance", "Interception", "Ball Recovery",
            "Goal Keeper", "Dribble", "Miscontrol", "Block", "50/50")
# Events whose location is the ball's, for the clock shift.
LOCATED = ("Pass", "Shot", "Ball Receipt*", "Ball Recovery", "Clearance", "Interception",
           "Miscontrol", "Dribble")
SHIFT_WINDOW = 60.0   # s of events per clock shift
SHIFT_RANGE = 15.0    # s, either way
SHIFT_STEP = 0.2      # s
SHIFT_MIN_EVENTS = 8  # events with the ball seen, for a window to have a shift of its own
# Bump when parsing or labelling changes, so stale .npz caches are rebuilt.
CACHE_VERSION = 2


def download(match_id, cache_dir=SKILLCORNER_CACHE):
    """Fetch one match's tracking, lineups and events into the cache. Skips files already there."""
    if match_id not in MATCHES:
        raise KeyError("unknown match {!r}; have {}".format(match_id, sorted(MATCHES)))
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    skillcorner, statsbomb = MATCHES[match_id]
    sources = {"match": SKILLCORNER_URL.format(COMMIT, skillcorner, "match_data"),
               "frames": SKILLCORNER_URL.format(COMMIT, skillcorner, "structured_data"),
               "events": STATSBOMB_URL.format(STATSBOMB_COMMIT, statsbomb)}
    names = {"match": "{}_match_data.json".format(skillcorner),
             "frames": "{}_structured_data.json".format(skillcorner),
             "events": "sb_{}.json".format(statsbomb)}
    paths = {}
    for kind, url in sorted(sources.items()):
        path = os.path.join(cache_dir, names[kind])
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            partial = path + ".part"
            request = urllib.request.Request(url, headers={"User-Agent": "mtah-possession"})
            with urllib.request.urlopen(request) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            os.rename(partial, path)
        paths[kind] = path
    return paths


def load(match_id, cache_dir=SKILLCORNER_CACHE):
    """(Tracking, labels) for one match, parsed once and then read from an .npz cache. The
    Tracking carries the event touches (Tracking.touches)."""
    cached = os.path.join(cache_dir, "{}_{}_v{}.npz".format(match_id, LABELS, CACHE_VERSION))
    if os.path.isfile(cached):
        stored = np.load(cached)
        tracking = Tracking(match_id, stored["players"], stored["team"], stored["ball"],
                            stored["period"], tuple(stored["pitch"]),
                            touches=(stored["touch_frames"], stored["touch_teams"]))
        return tracking, stored["labels"]

    paths = download(match_id, cache_dir)
    tracking, starts = parse_tracking(match_id, paths["match"], paths["frames"])
    data, home_sb = read_events(paths["events"], home_name(paths["match"]))
    events = statsbomb_events(data, home_sb, clock_shift(data, home_sb, tracking, starts))
    labels = event_labels(events, starts, tracking.period)
    touches = event_touches(events, starts, tracking.period)
    tracking = tracking.replace(touches=touches)
    np.savez_compressed(cached, players=tracking.players, team=tracking.team, ball=tracking.ball,
                        period=tracking.period, pitch=np.array(tracking.pitch), labels=labels,
                        touch_frames=touches[0], touch_teams=touches[1])
    return tracking, labels


# -- tracking ----------------------------------------------------------------------------------

def home_name(match_path):
    guard.check_input(match_path)
    with open(match_path) as handle:
        return json.load(handle)["home_team"]["short_name"]


def _seconds(clock):
    """SkillCorner's "MM:SS.ss" match clock in seconds."""
    minutes, seconds = clock.split(":")
    return 60 * int(minutes) + float(seconds)


def _slots(frames, team_of):
    """{(kind, key): slot} and the team per slot: one slot per identified player, and per team
    as few shared slots as the unidentified tracks need, each track in a slot no other track
    holds while it lasts."""
    spans = {}
    for index, frame in enumerate(frames):
        for item in frame["data"]:
            group = item.get("group_name")
            if item.get("trackable_object") is None and group in HOME_GROUPS + AWAY_GROUPS:
                key = (0 if group in HOME_GROUPS else 1, item["track_id"])
                first, _ = spans.get(key, (index, index))
                spans[key] = (first, index)
    identified = sorted(team_of)
    slot_of = {("player", p): i for i, p in enumerate(identified)}
    team = [team_of[p] for p in identified]
    for side in (0, 1):
        ends = []  # last frame of the track each shared slot of this team holds
        for key, (first, last) in sorted(((k, s) for k, s in spans.items() if k[0] == side),
                                         key=lambda kv: kv[1]):
            free = next((j for j, end in enumerate(ends) if end < first), None)
            if free is None:
                ends.append(last)
                team.append(side)
                free = len(ends) - 1
                slot_of[("shared", side, free)] = len(team) - 1
            ends[free] = last
            slot_of[("track",) + key] = slot_of[("shared", side, free)]
    return slot_of, np.array(team, dtype=np.int8)


def _resample(values, times, n_out):
    """Linear interpolation of (n, ...) samples taken at times (s) onto n_out frames at
    config.FPS; NaN wherever either neighbour is NaN."""
    position = np.interp(np.arange(n_out) / config.FPS, times, np.arange(len(times)))
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, len(times) - 1)
    fraction = position - low
    weight = fraction.reshape((-1,) + (1,) * (values.ndim - 1))
    out = values[low] * (1 - weight) + values[high] * weight
    exact = fraction < 1e-9  # on a sample: its own value, whatever the next one is
    out[exact] = values[low[exact]]
    return out


def parse_tracking(name, match_path, frames_path):
    """The match as a Tracking at config.FPS, each period end to end with PERIOD_GAP frames of
    nothing between them, metres from a corner; and the row of each period's kick off."""
    guard.check_input(match_path)
    guard.check_input(frames_path)
    with open(match_path) as handle:
        meta = json.load(handle)
    with open(frames_path) as handle:
        frames = [f for f in json.load(handle) if f["period"]]
    home = meta["home_team"]["id"]
    ball_id = meta["ball"]["trackable_object"]
    team_of = {p["trackable_object"]: 0 if p["team_id"] == home else 1 for p in meta["players"]}
    slot_of, team = _slots(frames, team_of)
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    origin = np.array([pitch[0] / 2, pitch[1] / 2])

    periods = sorted({f["period"] for f in frames})
    parts, starts, offset = [], {}, 0
    for p in periods:
        block = [f for f in frames if f["period"] == p]
        players = np.full((len(block), len(team), 2), np.nan)
        ball = np.full((len(block), 2), np.nan)
        for i, frame in enumerate(block):
            for item in frame["data"]:
                xy = (item["x"] + origin[0], item["y"] + origin[1])
                obj = item.get("trackable_object")
                if obj == ball_id:
                    ball[i] = xy
                elif obj in team_of:
                    players[i, slot_of[("player", obj)]] = xy
                elif obj is None and item.get("group_name") in HOME_GROUPS + AWAY_GROUPS:
                    side = 0 if item["group_name"] in HOME_GROUPS else 1
                    players[i, slot_of[("track", side, item["track_id"])]] = xy
        start = 45.0 * 60 * (p - 1)
        times = np.array([_seconds(f["time"]) for f in block]) - start
        n_out = int(np.floor(times[-1] * config.FPS)) + 1
        parts.append((p, _resample(players, times, n_out), _resample(ball, times, n_out)))
        starts[p] = offset
        offset += n_out + PERIOD_GAP
    total = offset - PERIOD_GAP
    out_players = np.full((total, len(team), 2), np.nan, dtype=np.float32)
    out_ball = np.full((total, 2), np.nan, dtype=np.float32)
    out_period = np.zeros(total, dtype=np.int8)
    for index, (p, players, ball) in enumerate(parts):
        span = slice(starts[p], starts[p] + len(ball))
        out_players[span], out_ball[span], out_period[span] = players, ball, index + 1
    return Tracking(name, out_players, team, out_ball, out_period, pitch), starts


# -- events ------------------------------------------------------------------------------------

def _period_seconds(stamp):
    hours, minutes, seconds = stamp.split(":")
    return 3600 * int(hours) + 60 * int(minutes) + float(seconds)


def read_events(path, home):
    """StatsBomb's events and the name StatsBomb gives the home team (named as SkillCorner names
    it)."""
    guard.check_input(path)
    with open(path) as handle:
        data = json.load(handle)
    names = {event["team"]["name"] for event in data}
    return data, next(n for n in names if n in home or home in n)


def clock_shift(data, home_sb, tracking, starts):
    """{period: shift per SHIFT_WINDOW s of the half}: the shift of the events' clock that brings
    their locations closest to the tracked ball (median distance), smoothed by a running median
    of three windows; a window with too few events seen takes its neighbours'. Each team's
    attacking direction in each half is whichever of StatsBomb's two orientations fits."""
    length, width = tracking.pitch
    shifts = np.arange(-SHIFT_RANGE, SHIFT_RANGE + 1e-9, SHIFT_STEP)
    out = {}
    for p, start in starts.items():
        index = sorted(starts).index(p) + 1
        located = [e for e in data if e["period"] == p and e.get("location")
                   and e["type"]["name"] in LOCATED]
        if not located:
            out[p] = np.zeros(1)
            continue
        t = np.array([_period_seconds(e["timestamp"]) for e in located])
        at = np.array([e["location"] for e in located], dtype=np.float64)
        at = np.stack([at[:, 0] * length / 120.0, width - at[:, 1] * width / 80.0], axis=1)
        home = np.array([e["team"]["name"] == home_sb for e in located])
        rows = start + np.round((t[:, None] + shifts[None, :]) * config.FPS).astype(np.int64)
        inside = (rows >= 0) & (rows < len(tracking.period))
        rows = np.where(inside, rows, 0)
        inside &= tracking.period[rows] == index
        ball = tracking.ball[rows]

        def distance(points):
            gap = np.hypot(ball[..., 0] - points[:, None, 0], ball[..., 1] - points[:, None, 1])
            return np.where(inside, gap, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN medians: no ball seen
            straight, turned = distance(at), distance(np.array([length, width]) - at)
            gap = np.empty_like(straight)
            for side in (True, False):
                mine = home == side
                keep = (np.nanmin(np.nanmedian(straight[mine], axis=0))
                        <= np.nanmin(np.nanmedian(turned[mine], axis=0)))
                gap[mine] = (straight if keep else turned)[mine]
            best = np.full(int(t.max() // SHIFT_WINDOW) + 1, np.nan)
            for k in range(len(best)):
                window = gap[(t >= k * SHIFT_WINDOW) & (t < (k + 1) * SHIFT_WINDOW)]
                seen = np.sum(~np.isnan(window), axis=0)
                if len(window) and seen.max() >= SHIFT_MIN_EVENTS:
                    median = np.where(seen >= SHIFT_MIN_EVENTS, np.nanmedian(window, axis=0),
                                      np.inf)
                    best[k] = shifts[np.argmin(median)]
        known = np.flatnonzero(~np.isnan(best))
        best = (np.interp(np.arange(len(best)), known, best[known]) if len(known)
                else np.zeros(len(best)))
        padded = np.concatenate([best[:1], best, best[-1:]])
        out[p] = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    return out


def _owning(event):
    """Whether an event shows its team with the ball: an OWNING kind that did not fail."""
    kind = event["type"]["name"]
    if kind not in OWNING:
        return False
    detail = event.get(kind.lower().replace("*", "").replace(" ", "_"), {})
    outcome = (detail.get("outcome") or {}).get("name", "")
    if kind == "Ball Receipt*":
        return outcome != "Incomplete"
    if kind == "Ball Recovery":
        return not detail.get("recovery_failure")
    if kind == "Interception":
        return outcome not in ("Lost", "Lost In Play", "Lost Out")
    if kind == "Dribble":
        return outcome != "Incomplete"
    if kind == "Goal Keeper":
        return detail.get("type", {}).get("name") not in ("Goal Conceded", "Shot Faced",
                                                            "Penalty Conceded")
    return True


def statsbomb_events(data, home_sb, shift=None):
    """[(period, start s, end s, kind, team, owner, restart)] from StatsBomb's events, on the
    tracking's clock when given clock_shift(); teams 0 home, 1 away. owner is the team the event
    shows with the ball by LABELS' rule, or None."""
    out = []
    for event in data:
        kind = event["type"]["name"]
        start = _period_seconds(event["timestamp"])
        if shift is not None and event["period"] in shift:
            windows = shift[event["period"]]
            start += windows[min(int(start // SHIFT_WINDOW), len(windows) - 1)]
        restart = ((kind == "Pass" and event["pass"].get("type", {}).get("name") in RESTART_PASSES)
                   or (kind == "Shot" and event["shot"]["type"]["name"] in RESTART_SHOTS))
        team = 0 if event["team"]["name"] == home_sb else 1
        if LABELS == "chain":
            owner = (0 if event["possession_team"]["name"] == home_sb else 1) if kind in ON_BALL \
                else None
        else:
            owner = team if _owning(event) else None
        out.append((event["period"], start, start + (event.get("duration") or 0.0), kind, team,
                    owner, restart))
    return sorted(out, key=lambda e: (e[0], e[1]))


def _row(starts, period, n, p, seconds):
    """The row of a time in period p, or None outside the period's rows."""
    if p not in starts:
        return None
    row = starts[p] + int(round(seconds * config.FPS))
    return row if row < n and period[row] == sorted(starts).index(p) + 1 else None


def event_labels(events, starts, period):
    """Per-row labels: the owner of the latest event with one, D from the end of the last event
    on the ball before a restart to the restart, X outside the periods."""
    n = len(period)
    labels = np.full(n, X, dtype=np.int8)
    for p in sorted(starts):
        rows = np.flatnonzero(period == sorted(starts).index(p) + 1)
        play = [e for e in events if e[0] == p and e[3] in ON_BALL]
        if not play or not len(rows):
            continue
        first, last = rows[0], rows[-1] + 1
        at = np.array([min(max(starts[p] + int(round(e[1] * config.FPS)), first), last - 1)
                       for e in play])
        owned = np.array([e[5] is not None for e in play])
        team = np.array([H if e[5] == 0 else A for e in play], dtype=np.int8)[owned]
        latest = np.searchsorted(at[owned], rows, side="right") - 1
        labels[rows] = team[np.maximum(latest, 0)]
        labels[first:at[0]] = D
        previous_end = None
        for e, row in zip(play, at):
            if e[6] and previous_end is not None and previous_end < row:
                labels[previous_end:row] = D
            end = min(max(starts[p] + int(round(e[2] * config.FPS)), first), last)
            previous_end = end if previous_end is None else max(previous_end, end)
            if e[6]:
                previous_end = end
        labels[previous_end:last] = D
    labels[period == 0] = X
    return labels


def event_touches(events, starts, period):
    """(rows, teams) where a team plays the ball: every touching event's start."""
    rows, teams = [], []
    for p, start, _, kind, team, _, _ in events:
        if kind not in TOUCHING:
            continue
        row = _row(starts, period, len(period), p, start)
        if row is not None:
            rows.append(row)
            teams.append(team)
    order = np.argsort(rows, kind="stable")
    return np.array(rows, dtype=np.int64)[order], np.array(teams, dtype=np.int8)[order]
