"""Synthetic matches for tests: plausible enough that every expert has something to find.

Two teams of eleven drift towards formation targets that depend on who has the ball -- the team
in possession pushes up and spreads, the other drops and narrows -- while the ball is carried and
passed inside the possessing team, intercepted at a change, and kicked over a touchline at every
stoppage. None of it resembles real football closely, and nothing scored on it says anything
about the real match. It exists so the pipeline can be run end to end, and its parsers round-
tripped, without the gated data and without a third party ever seeing it.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from possession.scoring import A, D, H, X
from possession.tracking import Tracking

# Params
PITCH = (105.0, 68.0)
PERIOD_GAP = 25
# Formation in a team's attacking frame (x from its own goal line): GK, back four, three, three.
BASE = np.array([(5, 34), (22, 10), (20, 26), (20, 42), (22, 58), (40, 14), (38, 34), (40, 54),
                 (58, 20), (62, 34), (58, 48)], dtype=np.float64)
BALL_SPEED = 15.0          # m/s for passes and clearances
MEAN_POSSESSION = 150      # frames
MEAN_DEAD = 200
MEAN_TOUCH = 50            # frames between passes


def match(name="synthetic", n_half=6000, seed=0, fps=25):
    """(Tracking, labels) for a two-half synthetic match."""
    rng = np.random.default_rng(seed)
    n = 2 * n_half + PERIOD_GAP
    players = np.full((n, 22, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    labels = np.full(n, X, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for half, directions in ((1, (1, -1)), (2, (-1, 1))):
        start = 0 if half == 1 else n_half + PERIOD_GAP
        sequence = _labels(n_half, rng)
        P, B = _simulate(sequence, directions, rng, fps)
        players[start:start + n_half], ball[start:start + n_half] = P, B
        labels[start:start + n_half] = sequence
        period[start:start + n_half] = half
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return Tracking(name, players, team, ball, period, PITCH, fps), labels


def _labels(n, rng):
    out = np.empty(n, dtype=np.int8)
    position, state = 0, D
    while position < n:
        if state == D:
            length = max(40, int(rng.geometric(1.0 / MEAN_DEAD)))
        else:
            length = max(20, int(rng.geometric(1.0 / MEAN_POSSESSION)))
        out[position:position + length] = state
        position += length
        if state == D:
            state = H if rng.random() < 0.5 else A
        else:
            state = D if rng.random() < 0.25 else (A if state == H else H)
    return out


def _targets(state, directions, ball):
    length, width = PITCH
    out = np.empty((22, 2))
    for k in (0, 1):
        base = BASE.copy()
        ball_att = ball[0] if directions[k] > 0 else length - ball[0]
        if state == k:
            base[1:, 0] += 12.0
            base[1:, 1] = 34 + 1.15 * (base[1:, 1] - 34)
        elif state == 1 - k:
            base[1:, 0] = base[1:, 0].mean() + 0.8 * (base[1:, 0] - base[1:, 0].mean()) - 4.0
            base[1:, 1] = 34 + 0.8 * (base[1:, 1] - 34)
        base[1:, 1] += 0.25 * (ball[1] - width / 2)
        base[1:, 0] += 0.2 * (ball_att - length / 2)
        x = base[:, 0] if directions[k] > 0 else length - base[:, 0]
        out[11 * k:11 * (k + 1)] = np.stack([x, base[:, 1]], axis=1)
    return out


def _simulate(labels, directions, rng, fps):
    n = len(labels)
    length, width = PITCH
    step = BALL_SPEED / fps
    ball = np.array([length / 2, width / 2])
    positions = _targets(D, directions, ball)
    out_players = np.empty((n, 22, 2))
    out_ball = np.empty((n, 2))
    holder, flight = None, None  # flight: (target point or player index, is_player)

    def nearest_of(k):
        members = np.arange(11 * k + 1, 11 * (k + 1))  # outfield only
        return int(members[np.argmin(np.hypot(*(positions[members] - ball).T))])

    for t in range(n):
        state = labels[t]
        if t == 0 or state != labels[t - 1]:
            if state in (H, A):
                if t > 0 and labels[t - 1] == D:
                    ball = np.clip(ball, 0.5, [length - 0.5, width - 0.5])
                holder, flight = None, (nearest_of(state), True)
            else:
                side = -1.5 if ball[1] < width / 2 else width + 1.5
                holder, flight = None, (np.array([ball[0], side]), False)

        rate = 0.01 if state == D else 0.04
        target = _targets(state, directions, ball)
        positions = positions + rate * (target - positions) + rng.normal(0, 0.04, positions.shape)

        if flight is not None:
            goal = positions[flight[0]] if flight[1] else flight[0]
            gap = goal - ball
            distance = np.hypot(*gap)
            if distance <= step:
                ball = goal.copy()
                holder = flight[0] if flight[1] else None
                flight = None
            else:
                ball = ball + gap / distance * step
        elif holder is not None:
            ball = positions[holder] + np.array([0.4, 0.0])
            if state in (H, A) and rng.random() < 1.0 / MEAN_TOUCH:
                mates = [p for p in range(11 * state + 1, 11 * (state + 1)) if p != holder]
                holder, flight = None, (int(rng.choice(mates)), True)

        out_players[t], out_ball[t] = positions, ball
    return out_players, out_ball


def write_atd(tracking, folder, stem="SYN"):
    """Write a Tracking in the provided FIFA/EPTS layout, for round-tripping read_atd()."""
    if not os.path.isdir(folder):
        os.makedirs(folder)
    root = ET.Element("Metadata")
    teams = ET.SubElement(root, "Teams")
    for team_id, name in (("T1", "Home"), ("T2", "Away")):
        ET.SubElement(ET.SubElement(teams, "Team", id=team_id), "Name").text = name
    roster = ET.SubElement(root, "Players")
    for slot, k in enumerate(tracking.team):
        ET.SubElement(roster, "Player", id="P{}".format(slot + 1), teamId="T1" if k == 0 else "T2")
    meta_path = os.path.join(folder, stem + "_atd.xml")
    ET.ElementTree(root).write(meta_path)

    length, width = tracking.pitch
    raw_path = os.path.join(folder, stem + "_atd.txt")
    with open(raw_path, "w") as handle:
        for frame in range(tracking.n_frames):
            groups = []
            for x, y in tracking.players[frame]:
                groups.append("NaN,NaN,NaN" if np.isnan(x) else
                              "{:.5f},{:.5f},0.0".format(x / length, y / width))
            bx, by = tracking.ball[frame]
            ball = "NaN,NaN,NaN" if np.isnan(bx) else "{:.5f},{:.5f},0.0".format(
                bx / length, by / width)
            handle.write("{}:{}:{}\n".format(frame + 1, ";".join(groups), ball))
    return meta_path, raw_path
