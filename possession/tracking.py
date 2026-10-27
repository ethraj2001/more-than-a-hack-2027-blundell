"""One container for tracking from any source, and the reader for the provided ATD.

Every source -- the provided ATD, the DFL training games, the synthetic test matches -- becomes a
Tracking: positions in metres with the origin at a pitch corner, NaN wherever something was not
seen, a team per track slot, and a period per frame (0 outside play). Everything downstream sees
only this, so a feature cannot quietly depend on which provider it came from.
"""

import numpy as np

from possession import config, guard

# Params
HOME, AWAY = 0, 1


class Tracking(object):
    """Players and ball for one match, frame-aligned, in metres.

    players  (T, N, 2) float32   x along the pitch length, y across it; NaN = not tracked
    team     (N,)      int8      0 home, 1 away -- per track slot, which is not a person
    ball     (T, 2)    float32   NaN = not tracked
    period   (T,)      int8      1, 2 ... for play; 0 for frames outside the match
    """

    def __init__(self, name, players, team, ball, period, pitch, fps=config.FPS):
        self.name = name
        self.players = np.asarray(players, dtype=np.float32)
        self.team = np.asarray(team, dtype=np.int8)
        self.ball = np.asarray(ball, dtype=np.float32)
        self.period = np.asarray(period, dtype=np.int8)
        self.pitch = (float(pitch[0]), float(pitch[1]))
        self.fps = fps

        n_frames, n_slots = self.players.shape[:2]
        if self.players.shape != (n_frames, n_slots, 2):
            raise ValueError("players must be (T, N, 2), got {}".format(self.players.shape))
        if self.team.shape != (n_slots,):
            raise ValueError("team must have one entry per slot")
        if self.ball.shape != (n_frames, 2) or self.period.shape != (n_frames,):
            raise ValueError("ball and period must cover every frame")

    @property
    def n_frames(self):
        return self.players.shape[0]

    def replace(self, **changes):
        """A copy with some arrays swapped out. Used by the noise model."""
        fields = {"name": self.name, "players": self.players, "team": self.team,
                  "ball": self.ball, "period": self.period, "pitch": self.pitch, "fps": self.fps}
        fields.update(changes)
        return Tracking(**fields)


def period_blocks(period):
    """[(start, stop, period)] contiguous half-open spans of play, in order."""
    period = np.asarray(period)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], period, [0]))) != 0)
    out = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if period[start] > 0:
            out.append((int(start), int(stop), int(period[start])))
    return out


# -- the provided ATD --------------------------------------------------------------------------

def load_atd(game=None, data_dir=None):
    """The provided tactical ATD as a Tracking, periods taken from the published half blocks.

    The half blocks are the ones DATA.md publishes ("boundaries, not answers"): using them is
    what lets the pipeline write X before kickoff, at halftime and after the whistle.
    """
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    game = game or GAME
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, "tactical")
    return read_atd(meta_path, raw_path, n_frames=source_frames("tactical", game),
                    pitch=GAMES[game]["pitch"], blocks=half_video_blocks(game), name=game)


def read_atd(meta_path, raw_path, n_frames, pitch, blocks, name="atd"):
    """Parse a FIFA/EPTS metadata + raw pair into a Tracking.

    Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>", coordinates normalised 0-1, NaN where
    missing, frame_id counting from 1 (DATA.md). Row frame_id - 1 is the submission row.
    """
    from starter.baseline import team_of_each_track
    guard.check_input(meta_path)
    guard.check_input(raw_path)

    team = team_of_each_track(meta_path)
    n_slots = len(team)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            players[row] = _parse_players(parts[1], n_slots)
            fields = parts[2].split(",")
            if len(fields) >= 2 and fields[0][:1] not in ("N", "n", ""):
                ball[row] = float(fields[0]), float(fields[1])

    scale = np.array(pitch, dtype=np.float32)
    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in blocks:
        period[first:last + 1] = half
    return Tracking(name, players * scale, team, ball * scale, period, pitch)


def _parse_players(text, n_slots):
    """(n_slots, 2) positions from one line's player field.

    Fast path: every slot written as three numbers, so the whole field is one numpy conversion.
    Slow path, for any other layout: slot by slot, treating a group that starts with N as missing.
    """
    values = text.replace(";", ",").split(",")
    if len(values) == 3 * n_slots:
        return np.array(values, dtype=np.float32).reshape(n_slots, 3)[:, :2]

    out = np.full((n_slots, 2), np.nan, dtype=np.float32)
    for slot, group in enumerate(text.split(";")[:n_slots]):
        fields = group.split(",")
        if len(fields) >= 2 and fields[0][:1] not in ("N", "n", ""):
            out[slot] = float(fields[0]), float(fields[1])
    return out
