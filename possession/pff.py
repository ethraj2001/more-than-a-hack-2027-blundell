"""PFF FC's open 2022 World Cup data -- broadcast tracking of all 64 games with PFF's own event
data synchronised to it -- as training games.

    https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data

Why these: international football at the top level, Spain's Barcelona-built side among it, in
full tracking. PFF gives it away through a sign-up form rather than a link, so download() cannot
fetch it: whoever runs this signs up, downloads it and unpacks it anywhere under PFF_CACHE. The
organisers can get it the same way, which is what RULES.md asks of external data. Nothing here
redistributes it; the cache is gitignored.

The format is read as kloppy's PFF deserializer reads it (kloppy/infra/serializers/tracking/
pff.py): one JSON object per line of {game_id}.jsonl.bz2, with the period, periodElapsedTime,
ballsSmoothed and home/awayPlayersSmoothed (by shirt number), and the game_event, if any, that the
frame carries; {game_id}.json metadata (fps, teams, pitch) and roster beside it. Positions are
metres from the centre spot. The smoothed positions cover every player, as the DFL games do, so
the copies are corrupted as theirs are; the frames are resampled from PFF's 29.97 to 25 a second.

The labels follow PFF's game events, as kloppy's ball state and owning team do (event_labels()):
dead from a ball out (OUT) or a period's end (END) until a kick off or the next event on the ball
(OTB); in play, the team the last event with one names as on the ball (home_ball) -- every event
on the ball, so changes sit at touches, like metrica.py's labels. Touches come from the same
events.

    unpack the download under cache/pff/, then:  python -m possession download --games PFF...
"""

import bz2
import json
import os

import numpy as np

from possession import config, guard
from possession.scoring import A, D, H, X
from possession.tracking import Tracking

# Params
PFF_CACHE = os.path.join(config.CACHE_DIR, "pff")
PREFIX = "PFF"
PERIOD_GAP = 25  # frames of X between periods, as in dfl.py
DEAD_EVENTS = ("OUT", "END")
LIVE_EVENTS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB")
# Bump when parsing or labelling changes, so stale .npz caches are rebuilt.
CACHE_VERSION = 1


def _files(cache_dir=PFF_CACHE):
    """{game id: {"tracking", "meta", "roster": path}} for every game found under cache_dir: the
    tracking as {id}.jsonl.bz2, the metadata and roster as {id}.json in folders whose names say
    which."""
    found = {}
    if not os.path.isdir(cache_dir):
        return found
    for folder, _, names in os.walk(cache_dir):
        where = folder.lower()
        for name in names:
            if name.endswith(".jsonl.bz2"):
                found.setdefault(name[:-len(".jsonl.bz2")], {})["tracking"] = os.path.join(folder, name)
            elif name.endswith(".json") and name[:-5].isdigit():
                kind = "roster" if "roster" in where else "meta" if "meta" in where else None
                if kind:
                    found.setdefault(name[:-5], {})[kind] = os.path.join(folder, name)
    return {k: v for k, v in found.items() if len(v) == 3}


def available(cache_dir=PFF_CACHE):
    """The games unpacked under cache_dir, by name."""
    return tuple(PREFIX + game for game in sorted(_files(cache_dir), key=int))


def download(match_id, cache_dir=PFF_CACHE):
    """The game's files, which have to be in the cache already: PFF hands them out by form."""
    files = _files(cache_dir).get(match_id[len(PREFIX):])
    if files is None:
        raise IOError("{} is not under {}: sign up for PFF FC's 2022 World Cup data (see "
                      "possession/pff.py) and unpack it there".format(match_id, cache_dir))
    return files


def load(match_id, cache_dir=PFF_CACHE):
    """(Tracking, labels) for one game, parsed once and then read from an .npz cache. The
    Tracking carries the event touches (Tracking.touches)."""
    cached = os.path.join(cache_dir, "{}_v{}.npz".format(match_id, CACHE_VERSION))
    if os.path.isfile(cached):
        stored = np.load(cached)
        tracking = Tracking(match_id, stored["players"], stored["team"], stored["ball"],
                            stored["period"], tuple(stored["pitch"]),
                            touches=(stored["touch_frames"], stored["touch_teams"]))
        return tracking, stored["labels"]
    paths = download(match_id, cache_dir)
    tracking, labels, touches = parse(match_id, paths)
    tracking = tracking.replace(touches=touches)
    np.savez_compressed(cached, players=tracking.players, team=tracking.team, ball=tracking.ball,
                        period=tracking.period, pitch=np.array(tracking.pitch), labels=labels,
                        touch_frames=touches[0], touch_teams=touches[1])
    return tracking, labels


def teams(match_id, cache_dir=PFF_CACHE):
    """(home, away) team names of a game, from its metadata."""
    meta = _meta(download(match_id, cache_dir)["meta"])
    return meta["homeTeam"]["name"], meta["awayTeam"]["name"]


def _meta(path):
    guard.check_input(path)
    with open(path) as handle:
        meta = json.load(handle)
    return meta[0] if isinstance(meta, list) else meta


def _resample(values, times, n_out):
    """Linear interpolation of (n, ...) samples taken at times (s) onto n_out frames at
    config.FPS; NaN wherever either neighbour is NaN."""
    position = np.interp(np.arange(n_out) / config.FPS, times, np.arange(len(times)))
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, len(times) - 1)
    fraction = position - low
    weight = fraction.reshape((-1,) + (1,) * (values.ndim - 1))
    out = values[low] * (1 - weight) + values[high] * weight
    exact = fraction < 1e-9
    out[exact] = values[low[exact]]
    return out


def parse(name, paths):
    """The game as a Tracking at config.FPS (each period end to end, PERIOD_GAP frames of nothing
    between them, metres from a corner), its labels and its touches."""
    meta = _meta(paths["meta"])
    pitch_info = meta["stadium"]["pitches"][0]
    pitch = (float(pitch_info["length"]), float(pitch_info["width"]))
    home_id = meta["homeTeam"]["id"]
    guard.check_input(paths["roster"])
    with open(paths["roster"]) as handle:
        roster = json.load(handle)
    slot_of = {}
    for entry in roster:
        side = 0 if entry["team"]["id"] == home_id else 1
        slot_of.setdefault((side, int(entry["shirtNumber"])), len(slot_of))

    guard.check_input(paths["tracking"])
    native = {}  # period -> [time, ball, {slot: xy}, (event type, home_ball)]
    with bz2.open(paths["tracking"], "rt") as handle:
        for line in handle:
            frame = json.loads(line)
            period = frame.get("period")
            if period is None or frame.get("periodElapsedTime") is None:
                continue
            ball = frame.get("ballsSmoothed") or {}
            at = {}
            for side, key in ((0, "homePlayersSmoothed"), (1, "awayPlayersSmoothed")):
                for player in frame.get(key) or []:
                    if player.get("x") is None or player.get("y") is None:
                        continue
                    slot = slot_of.setdefault((side, int(player["jerseyNum"])), len(slot_of))
                    at[slot] = (player["x"], player["y"])
            event = frame.get("game_event") or {}
            native.setdefault(int(period), []).append(
                (float(frame["periodElapsedTime"]),
                 (ball.get("x"), ball.get("y")) if ball.get("x") is not None
                 and ball.get("y") is not None else (np.nan, np.nan),
                 at, (event.get("game_event_type"), event.get("home_ball"))))

    team = np.zeros(len(slot_of), dtype=np.int8)
    for (side, _), slot in slot_of.items():
        team[slot] = side
    origin = np.array(pitch) / 2.0
    blocks, offset = [], 0
    for period in sorted(native):
        rows = sorted(native[period], key=lambda r: r[0])
        times = np.array([r[0] for r in rows])
        times, keep = np.unique(times, return_index=True)
        rows = [rows[k] for k in keep]
        times -= times[0]
        players = np.full((len(rows), len(team), 2), np.nan)
        ball = np.array([r[1] for r in rows], dtype=np.float64) + origin
        for i, row in enumerate(rows):
            for slot, xy in row[2].items():
                players[i, slot] = xy
        players += origin
        n_out = int(np.floor(times[-1] * config.FPS)) + 1
        # The ball's state and owner after each native frame's event, carried to the frames
        # resampled from it (the last native frame at or before each).
        state, owner, labels_native, touches = "dead", None, [], []
        for i, row in enumerate(rows):
            kind, home_ball = row[3]
            if kind in DEAD_EVENTS:
                state = "dead"
            elif kind in LIVE_EVENTS:
                state = "live"
            if home_ball is not None:
                owner = H if home_ball else A
                if kind == "OTB":
                    touches.append((i, 0 if home_ball else 1))
            labels_native.append(D if state == "dead" or owner is None else owner)
        source = np.searchsorted(times, np.arange(n_out) / config.FPS, side="right") - 1
        blocks.append((offset, _resample(players, times, n_out), _resample(ball, times, n_out),
                       np.array(labels_native, dtype=np.int8)[np.maximum(source, 0)],
                       [(int(round(times[i] * config.FPS)) + offset, side) for i, side in touches
                        if int(round(times[i] * config.FPS)) < n_out]))
        offset += n_out + PERIOD_GAP
    total = offset - PERIOD_GAP
    out_players = np.full((total, len(team), 2), np.nan, dtype=np.float32)
    out_ball = np.full((total, 2), np.nan, dtype=np.float32)
    out_period = np.zeros(total, dtype=np.int8)
    labels = np.full(total, X, dtype=np.int8)
    touch_rows, touch_teams = [], []
    for index, (start, players, ball, lab, touches) in enumerate(blocks):
        span = slice(start, start + len(ball))
        out_players[span], out_ball[span], out_period[span] = players, ball, index + 1
        labels[span] = lab
        touch_rows += [r for r, _ in touches]
        touch_teams += [s for _, s in touches]
    tracking = Tracking(name, out_players, team, out_ball, out_period, pitch)
    return (tracking, labels, (np.array(touch_rows, dtype=np.int64),
                               np.array(touch_teams, dtype=np.int8)))
