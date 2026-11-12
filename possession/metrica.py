"""Metrica Sports' open sample games: three anonymised matches, tracking plus event data.

    https://github.com/metrica-sports/sample-data

Why these: Metrica made the provided game's tracking. Its sample games are its own automatic
tracking -- the ball is lost on 32-41% of frames, as on the provided ATD's 41% -- with human event
data of the same games, which RULES.md allows for games other than the provided one. The
repository states no licence; it asks users to "be responsible with the use of this data" and to
acknowledge the source in anything public. Nothing here redistributes it: download() fetches it
into the cache, which is gitignored, as dfl.py does with the DFL matches.

No sample game has a per-frame possession column, so the labels are built from the events
(labels()):

  * a team owns the ball from the start of each of its on-ball events -- pass, carry, shot, set
    piece, recovery, ball lost, ball out -- until the start of the other team's next one;
  * the ball is dead from a ball out, a foul, a goal, a shot out or an offside until the next
    on-ball event, the restart.

So a change of possession sits at the new team's first event, a recovery or a pass, which the
events place a median of a frame or two from the touch. These are labels from human data about
other games, documented here and in the technical document, not mechanical ones like DFL's.

Touches -- the touch expert's targets -- come from the events too (event_touches()): with the
ball lost on a third of frames, the ball cannot show them (pipeline.touch_frames).

    python -m possession download --games MET1 MET2 MET3     # about 180 MB, once
"""

import csv
import json
import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from possession import config, guard
from possession.scoring import A, D, H, X
from possession.tracking import Tracking

# Params
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/{}/{}"
MATCHES = {
    "MET1": ("Sample_Game_1", {"events": "Sample_Game_1_RawEventsData.csv",
                               "home": "Sample_Game_1_RawTrackingData_Home_Team.csv",
                               "away": "Sample_Game_1_RawTrackingData_Away_Team.csv"}),
    "MET2": ("Sample_Game_2", {"events": "Sample_Game_2_RawEventsData.csv",
                               "home": "Sample_Game_2_RawTrackingData_Home_Team.csv",
                               "away": "Sample_Game_2_RawTrackingData_Away_Team.csv"}),
    "MET3": ("Sample_Game_3", {"events": "Sample_Game_3_events.json",
                               "meta": "Sample_Game_3_metadata.xml",
                               "tracking": "Sample_Game_3_tracking.txt"}),
}
METRICA_CACHE = os.path.join(config.CACHE_DIR, "metrica")
PITCH = (105.0, 68.0)  # the repository's stated pitch; positions come as fractions of it
PERIOD_GAP = 25        # frames of X between periods, as in dfl.py
# Events by which a team plays the ball, so owns it from their start frame.
ON_BALL = ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT")
# Bump when parsing or labelling changes, so stale .npz caches are rebuilt.
CACHE_VERSION = 1


def download(match_id, cache_dir=METRICA_CACHE):
    """Fetch one sample game's files into the cache. Skips files already there."""
    if match_id not in MATCHES:
        raise KeyError("unknown match {!r}; have {}".format(match_id, sorted(MATCHES)))
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    folder, files = MATCHES[match_id]
    paths = {}
    for kind, name in sorted(files.items()):
        path = os.path.join(cache_dir, name)
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            partial = path + ".part"
            request = urllib.request.Request(BASE_URL.format(folder, name),
                                             headers={"User-Agent": "mtah-possession"})
            with urllib.request.urlopen(request) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            os.rename(partial, path)
        paths[kind] = path
    return paths


def load(match_id, cache_dir=METRICA_CACHE):
    """(Tracking, labels) for one sample game, parsed once and then read from an .npz cache.
    The Tracking carries the event touches (Tracking.touches)."""
    cached = os.path.join(cache_dir, "{}_v{}.npz".format(match_id, CACHE_VERSION))
    if os.path.isfile(cached):
        stored = np.load(cached)
        tracking = Tracking(match_id, stored["players"], stored["team"], stored["ball"],
                            stored["period"], tuple(stored["pitch"]),
                            touches=(stored["touch_frames"], stored["touch_teams"]))
        return tracking, stored["labels"]

    paths = download(match_id, cache_dir)
    if "meta" in paths:
        players, team, ball, period_of, frames = parse_epts(paths["meta"], paths["tracking"])
        events = json_events(paths["events"], home_id(paths["meta"]))
    else:
        players, team, ball, period_of, frames = parse_csv(paths["home"], paths["away"])
        events = csv_events(paths["events"])
    tracking, row_of = lay_out(match_id, players, team, ball, period_of, frames)
    labels = event_labels(events, row_of, tracking.period)
    touches = event_touches(events, row_of)
    tracking = tracking.replace(touches=touches)
    np.savez_compressed(cached, players=tracking.players, team=tracking.team, ball=tracking.ball,
                        period=tracking.period, pitch=np.array(tracking.pitch), labels=labels,
                        touch_frames=touches[0], touch_teams=touches[1])
    return tracking, labels


# -- tracking ----------------------------------------------------------------------------------

def parse_csv(home_path, away_path):
    """The two-file CSV format: per team, Period, Frame, Time, then x, y per player, then the
    ball. Returns (players (F, N, 2) in metres, team (N,), ball (F, 2), period (F,), frame (F,))."""
    parts, teams = [], []
    for k, path in enumerate((home_path, away_path)):
        guard.check_input(path)
        values = np.genfromtxt(path, delimiter=",", skip_header=3, dtype=np.float64)
        if k == 0:
            period, frame, ball = values[:, 0], values[:, 1], values[:, -2:]
        elif not np.array_equal(values[:, 1], frame):
            raise ValueError("{}: frames differ from the home file's".format(path))
        xy = values[:, 3:-2]
        parts.append(xy.reshape(len(xy), -1, 2))
        teams += [k] * parts[-1].shape[1]
    players = np.concatenate(parts, axis=1)
    return (_metres(players), np.array(teams, dtype=np.int8), _metres(ball),
            period.astype(np.int64), frame.astype(np.int64))


def home_id(meta_path):
    """The local team's id in an EPTS metadata file."""
    guard.check_input(meta_path)
    score = ET.parse(meta_path).getroot().find(".//Score")
    return score.get("idLocalTeam")


def parse_epts(meta_path, tracking_path):
    """The FIFA EPTS format: "frame:x,y;x,y;...:ball_x,ball_y" per line, the player order set by
    the metadata's DataFormatSpecification for each frame range (it changes at substitutions)."""
    guard.check_input(meta_path)
    guard.check_input(tracking_path)
    root = ET.parse(meta_path).getroot()
    home = home_id(meta_path)
    player_team = {p.get("id"): (0 if p.get("teamId") == home else 1)
                   for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    slots = sorted(player_team)
    slot_of = {p: i for i, p in enumerate(slots)}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = [channel_player[ref.get("playerChannelId")]
                 for ref in spec.iter("PlayerChannelRef")][::2]  # x then y per player
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([slot_of[p] for p in order])))
    bounds = {p.findtext("Name"): p.findtext("Value")
              for p in root.iter("ProviderParameter")}
    halves = [(int(bounds[k + "_start"]), int(bounds[k + "_end"]))
              for k in ("first_half", "second_half", "first_extra_half", "second_extra_half")
              if bounds.get(k + "_start")]

    with open(tracking_path) as handle:
        lines = handle.read().split()
    n = len(lines)
    frame = np.empty(n, dtype=np.int64)
    players = np.full((n, len(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, line in enumerate(lines):
        number, people, at = line.split(":")
        frame[i] = int(number)
        spec = next(s for s in specs if s[0] <= frame[i] <= s[1])
        xy = np.array([pair.split(",") for pair in people.split(";")], dtype=np.float64)
        players[i, spec[2]] = xy
        ball[i] = np.array(at.split(","), dtype=np.float64)
    period = np.zeros(n, dtype=np.int64)
    for index, (first, last) in enumerate(halves):
        period[(frame >= first) & (frame <= last)] = index + 1
    team = np.array([player_team[p] for p in slots], dtype=np.int8)
    return _metres(players), team, _metres(ball), period, frame


def _metres(fractions):
    """Metrica's 0..1 fractions of the pitch to metres from a corner."""
    out = np.array(fractions, dtype=np.float64)
    out[..., 0] *= PITCH[0]
    out[..., 1] *= PITCH[1]
    return out


def lay_out(name, players, team, ball, period_of, frames):
    """A Tracking with each period's frames end to end and PERIOD_GAP frames of nothing between
    them, plus {Metrica frame number: row}."""
    blocks = [np.flatnonzero(period_of == p) for p in sorted(set(period_of.tolist())) if p > 0]
    total = sum(len(b) for b in blocks) + PERIOD_GAP * (len(blocks) - 1)
    out_players = np.full((total, players.shape[1], 2), np.nan, dtype=np.float32)
    out_ball = np.full((total, 2), np.nan, dtype=np.float32)
    out_period = np.zeros(total, dtype=np.int8)
    row_of, offset = {}, 0
    for index, rows in enumerate(blocks):
        span = slice(offset, offset + len(rows))
        out_players[span], out_ball[span] = players[rows], ball[rows]
        out_period[span] = index + 1
        row_of.update(zip(frames[rows].tolist(), range(offset, offset + len(rows))))
        offset += len(rows) + PERIOD_GAP
    return Tracking(name, out_players, team, out_ball, out_period, PITCH), row_of


# -- events ------------------------------------------------------------------------------------

def csv_events(path):
    """[(team 0 home / 1 away, type, subtype, start frame, end frame)] from the CSV events."""
    guard.check_input(path)
    out = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["Team"] not in ("Home", "Away"):
                continue
            out.append((0 if row["Team"] == "Home" else 1, row["Type"], row["Subtype"],
                        int(float(row["Start Frame"])), int(float(row["End Frame"]))))
    return out


def json_events(path, home):
    """The same, from the JSON events; subtypes joined with "-" as the CSV writes them."""
    guard.check_input(path)
    with open(path) as handle:
        data = json.load(handle)["data"]
    out = []
    for event in data:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        subtype = "-".join(s["name"] for s in subtypes)
        start, end = event["start"]["frame"] or 0, event["end"]["frame"] or 0
        out.append((0 if event["team"]["id"] == home else 1, event["type"]["name"], subtype,
                    int(start), int(end)))
    return out


def _stops(kind, subtype):
    """Whether an event ends live play: the ball out, a foul, a goal, a shot out, an offside."""
    if kind in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if kind == "SHOT":
        return "GOAL" in subtype or "OUT" in subtype
    return kind == "BALL LOST" and "OFFSIDE" in subtype


def event_labels(events, row_of, period):
    """Per-row labels: the team of the latest on-ball event, D from a stoppage to the next
    on-ball event after it, X outside the periods."""
    labels = np.full(len(period), X, dtype=np.int8)
    on_ball = sorted((row_of[s], team) for team, kind, _, s, _ in events
                     if kind in ON_BALL and s in row_of)
    if not on_ball:
        return labels
    starts = np.array([r for r, _ in on_ball])
    teams = np.array([H if t == 0 else A for _, t in on_ball], dtype=np.int8)
    play = np.flatnonzero(period > 0)
    latest = np.searchsorted(starts, play, side="right") - 1
    labels[play] = teams[np.maximum(latest, 0)]
    for team, kind, subtype, s, e in events:
        if not _stops(kind, subtype):
            continue
        frame = e if e > 0 and kind != "FAULT RECEIVED" else s
        if frame not in row_of:
            continue
        first = row_of[frame]
        after = np.searchsorted(starts, first, side="right")
        last = starts[after] if after < len(starts) else len(period)
        same = period[first:last] == period[first]
        labels[first:last][same] = D
    labels[period == 0] = X
    return labels


def event_touches(events, row_of):
    """(rows, teams) where a team plays the ball: every on-ball event's start, and every pass's
    end, where a team-mate receives it."""
    rows, teams = [], []
    for team, kind, _, s, e in events:
        if kind in ON_BALL and s in row_of:
            rows.append(row_of[s])
            teams.append(team)
        if kind == "PASS" and e in row_of:
            rows.append(row_of[e])
            teams.append(team)
    order = np.argsort(rows, kind="stable")
    return np.array(rows, dtype=np.int64)[order], np.array(teams, dtype=np.int8)[order]
