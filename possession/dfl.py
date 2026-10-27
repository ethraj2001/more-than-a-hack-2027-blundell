"""The open IDSSE/DFL Bundesliga matches: the labelled training and test games.

Bassek, Weber, Rein & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data -- seven full 2022/23 Bundesliga and 2. Bundesliga matches, released
under CC BY 4.0. None of them is the provided game.

Why this dataset: DFL positional data carries, on every 25 fps ball frame, a BallStatus (alive or
dead) and a BallPossession (home or away). That is the same kind of signal the challenge ground
truth is derived from -- "D where the corrected tracking marks the ball dead, otherwise the team
it names as owning the ball" (CHALLENGE.md) -- so the labels below are mechanical, not human.

One known difference, worth stating in the technical document: DFL's possession and ball-status
conventions are DFL's, not Metrica's. The model learns DFL's notion of "dead" and "owning". If the
two providers disagree systematically -- how early a throw-in counts as live, say -- that gap is
a domain shift no amount of noise injection removes.

    python -m possession download             # all seven, about 2.6 GB, once

Files are fetched from figshare's download host directly. kloppy's IDSSE helper points at a host
that now answers scripted requests with a bot challenge, which is why it is not used here.
"""

import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from possession import config, guard
from possession.scoring import A, D, H, X
from possession.tracking import Tracking

# Params
# match id -> figshare file ids, as published with the dataset (the same table kloppy carries).
MATCHES = {
    "J03WMX": {"meta": 51643475, "tracking": 51643514},  # 1. FC Koeln - FC Bayern Muenchen
    "J03WN1": {"meta": 51643472, "tracking": 51643517},  # VfL Bochum - Bayer 04 Leverkusen
    "J03WPY": {"meta": 51643487, "tracking": 51643526},  # Fortuna Duesseldorf - 1. FC Nuernberg
    "J03WOH": {"meta": 51643478, "tracking": 51643520},  # Fortuna Duesseldorf - Jahn Regensburg
    "J03WQQ": {"meta": 51643484, "tracking": 51643529},  # Fortuna Duesseldorf - FC St. Pauli
    "J03WOY": {"meta": 51643481, "tracking": 51643523},  # Fortuna Duesseldorf - Hansa Rostock
    "J03WR9": {"meta": 51643490, "tracking": 51643532},  # Fortuna Duesseldorf - Kaiserslautern
}
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{}"
DFL_CACHE = os.path.join(config.CACHE_DIR, "dfl")
SECTION_ORDER = ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")
# Frames of X written between two periods, so the scorer's valid_runs() splits them exactly as
# it splits the provided game's halftime -- no possession can straddle two halves.
PERIOD_GAP = 25
BALL_TEAM = "BALL"
# Bump when parse() changes, so stale .npz caches are rebuilt rather than silently reused.
CACHE_VERSION = 2


def download(match_id, cache_dir=DFL_CACHE):
    """Fetch one match's metadata and tracking XML into the cache. Skips files already there."""
    if match_id not in MATCHES:
        raise KeyError("unknown match {!r}; have {}".format(match_id, sorted(MATCHES)))
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    paths = {}
    for kind, file_id in sorted(MATCHES[match_id].items()):
        path = os.path.join(cache_dir, "{}_{}.xml".format(match_id, kind))
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            partial = path + ".part"
            request = urllib.request.Request(DOWNLOAD_URL.format(file_id),
                                             headers={"User-Agent": "mtah-possession"})
            with urllib.request.urlopen(request) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            with open(partial, "rb") as handle:
                head = handle.read(64)
            if not head.lstrip().startswith(b"<?xml"):
                os.remove(partial)
                raise IOError("{} {}: the server did not return XML -- download it by hand from "
                              "the dataset's figshare page into {}".format(match_id, kind, path))
            os.rename(partial, path)
        paths[kind] = path
    return paths


def load(match_id, cache_dir=DFL_CACHE):
    """(Tracking, labels) for one DFL match, parsed once and then read from an .npz cache.

    labels are the scorer's codes -- H, A, D, and X where DFL says nothing (gaps between periods,
    frames without a ball record).
    """
    cached = os.path.join(cache_dir, "{}_v{}.npz".format(match_id, CACHE_VERSION))
    if os.path.isfile(cached):
        stored = np.load(cached)
        tracking = Tracking(match_id, stored["players"], stored["team"], stored["ball"],
                            stored["period"], tuple(stored["pitch"]))
        return tracking, stored["labels"]

    paths = download(match_id, cache_dir)
    tracking, labels = parse(paths["meta"], paths["tracking"], match_id)
    np.savez_compressed(cached, players=tracking.players, team=tracking.team, ball=tracking.ball,
                        period=tracking.period, pitch=np.array(tracking.pitch), labels=labels)
    return tracking, labels


def club_team_ids(meta_path):
    """(home, guest) TeamIds, as DFL lists them with Role="home" and Role="guest"."""
    guard.check_input(meta_path)
    roles = {team.get("Role"): team.get("TeamId")
             for team in ET.parse(meta_path).getroot().iter("Team")}
    if "home" not in roles or "guest" not in roles:
        raise ValueError("{}: expected a home and a guest Team".format(meta_path))
    return roles["home"], roles["guest"]


def parse(meta_path, tracking_path, name="dfl"):
    """Stream a DFL positional XML into a Tracking plus per-frame labels.

    The file is one FrameSet per object per period, each a run of <Frame N= X= Y= .../>. Frame
    numbers N are a match-wide 25 fps counter; the ball's FrameSet defines each period's span and
    carries BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away). Coordinates are
    metres from the centre spot and are shifted to the corner origin used everywhere else.
    """
    guard.check_input(tracking_path)
    home, guest = club_team_ids(meta_path)
    pitch = (105.0, 68.0)
    series = {}  # (section, object id) -> lists
    owner = {}   # object id -> team id

    section = person = team_id = None
    current = None
    for event, element in ET.iterparse(tracking_path, events=("start", "end")):
        tag = element.tag
        if event == "start":
            if tag == "FrameSet":
                section, team_id = element.get("GameSection"), element.get("TeamId")
                person = element.get("PersonId")
                if team_id not in (BALL_TEAM, home, guest):
                    current = None  # referees: on the pitch, but in neither team
                    continue
                owner[person] = team_id
                current = series.setdefault((section, person), {"n": [], "x": [], "y": [],
                                                                "status": [], "poss": []})
            elif tag == "PitchSize":
                pitch = (float(element.get("X")), float(element.get("Y")))
            continue
        if tag == "Frame" and current is None:
            element.clear()
        elif tag == "Frame":
            attrib = element.attrib
            current["n"].append(int(attrib["N"]))
            current["x"].append(float(attrib["X"]))
            current["y"].append(float(attrib["Y"]))
            if team_id == BALL_TEAM:
                current["status"].append(int(attrib.get("BallStatus", -1)))
                current["poss"].append(int(attrib.get("BallPossession", 0)))
            element.clear()
        elif tag == "FrameSet":
            element.clear()

    sections = [s for s in SECTION_ORDER if any(k[0] == s for k in series)]
    sections += sorted({k[0] for k in series} - set(sections))
    ball_ids = sorted({obj for obj, team in owner.items() if team == BALL_TEAM})
    if len(ball_ids) != 1:
        raise ValueError("{}: expected one ball object, found {}".format(tracking_path, ball_ids))
    ball_id = ball_ids[0]
    slots = sorted(obj for obj, team in owner.items() if team != BALL_TEAM)
    slot_of = {obj: i for i, obj in enumerate(slots)}
    team = np.array([0 if owner[obj] == home else 1 for obj in slots], dtype=np.int8)

    # Each period's span is the ball's frame range; periods are laid end to end with a gap.
    spans, total = [], 0
    for s in sections:
        frames = np.asarray(series[(s, ball_id)]["n"])
        first, last = int(frames.min()), int(frames.max())
        spans.append((s, first, total))
        total += (last - first + 1) + PERIOD_GAP
    total -= PERIOD_GAP

    players = np.full((total, len(slots), 2), np.nan, dtype=np.float32)
    ball = np.full((total, 2), np.nan, dtype=np.float32)
    period = np.zeros(total, dtype=np.int8)
    labels = np.full(total, X, dtype=np.int8)
    half_length, half_width = pitch[0] / 2.0, pitch[1] / 2.0

    for index, (s, first, offset) in enumerate(spans):
        frames = np.asarray(series[(s, ball_id)]["n"])
        span = int(frames.max()) - first + 1
        period[offset:offset + span] = index + 1
        for (sec, obj), data in series.items():
            if sec != s:
                continue
            rows = np.asarray(data["n"]) - first + offset
            keep = (rows >= offset) & (rows < offset + span)
            rows = rows[keep]
            xy = np.stack([np.asarray(data["x"])[keep] + half_length,
                           np.asarray(data["y"])[keep] + half_width], axis=1)
            if obj == ball_id:
                ball[rows] = xy
                status = np.asarray(data["status"])[keep]
                poss = np.asarray(data["poss"])[keep]
                code = np.full(len(rows), X, dtype=np.int8)
                code[(status == 1) & (poss == 1)] = H
                code[(status == 1) & (poss == 2)] = A
                code[status == 0] = D
                labels[rows] = code
            else:
                players[rows, slot_of[obj]] = xy

    return Tracking(name, players, team, ball, period, pitch), labels
