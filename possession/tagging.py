"""Smart Tagging: Metrica's automatic phases of play, as features -- and look-alike tags for the
DFL games, so the models can learn how far to trust them.

The provided game comes with 638 automatic tags: phases (Build Up, Progression, Ball in Final
Third, Defending in Middle Third, transitions ...) with a Team, and set pieces (DATA.md). No DFL
game has them, and a model cannot learn to use an input its training data lacks. So simulate()
writes look-alike tags for each DFL game from its labels and clean tracking, by rules that follow
each code's name, thinned to the provided file's rate per code, and then corrupted -- edges
moved, teams swapped, phases dropped, false phases added -- by amounts drawn afresh for every
copy. Trained across that range, the models learn to treat a tag as evidence to weigh against the
tracking, not as an answer.

What a tag's Team says depends on its code, and the rules here assume:

    ATTACKING codes   the tagged team has the ball
    DEFENDING codes   the tagged team is defending: the other team has the ball
    SET PIECES        the tagged team restarts play; throw-ins are tagged N/A in the provided file

If Metrica means something else by a code, the simulated tags teach the wrong thing for it. That
cannot be measured on DFL. What can be measured, on any game and without labels, is how well a
game's tags agree with its own tracking: how often the team a tag says has the ball is the team
nearest the ball. check() does that before a model is trained (pipeline.tag_plan). Against the
look-alike tags of the training games, on noisy copies like the game's own tracking, it finds:

    a code group whose teams read reversed   flipped back -- a wrong club match, or a Team that
                                             means the other side
    tags running late or early               moved back by the offset that lines them up
    a group that agrees no better than the   the tags are not read, and the model is the one
    worst look-alike, or than chance           trained without them

Set pieces are read for their timing alone (config.TAGS_SET_TEAMS), so the check judges only the
open-play groups: a match has about 33 set pieces with a team, too few to judge them by.

Models trained on harder corruption instead (TAGS_HARD, with agreement() as features,
TAGS_AGREEMENT) were measured and lost the gain; see config.py.
"""

import difflib

import numpy as np

from possession import config
from possession.features import _attack_directions
from possession.scoring import A, D, H
from possession.tracking import period_blocks

# Params
ATTACKING = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL", "BALL IN FINAL THIRD",
             "BALL IN THE BOX", "PLAYERS IN THE BOX", "ATTACKING TRANSITION")
DEFENDING = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
SET_PIECES = "SET PIECES"
# Instances of each code in the provided game's file (DATA.md), over its 158,857 graded frames.
# Simulated tags are thinned to the same rate per frame of play.
PER_MATCH = {"PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
             "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
             "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
             "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
             "DEFENDING IN ATTACKING THIRD": 22}
PER_MATCH_FRAMES = 158857
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16   # metres
SECOND = 25                                # frames
LONG_BALL_M = 30.0                         # metres travelled in two seconds
# Corruption, drawn per copy from these ranges: edge shift (s.d., frames), share of tags whose
# team is swapped, share dropped, and false tags added as a share of the real ones.
EDGE_SD = (10, 60)
SWAP_RATE = (0.0, 0.2)
DROP_RATE = (0.0, 0.3)
FALSE_RATE = (0.0, 0.15)
# Worse copies, each mode drawn independently per copy with these probabilities: no tags at all;
# every edge shifted by one offset (frames); up to half the teams swapped; one code group's
# meaning reversed.
NONE_RATE = 0.15
OFFSET_RATE, OFFSET = 0.25, (-50, 50)
BAD_SWAP_RATE, BAD_SWAP = 0.15, (0.2, 0.5)
FLIP_RATE = 0.15
GROUPS = {"att": ATTACKING, "def": DEFENDING, "set": (SET_PIECES,)}
TEAM_SERIES = ("tag_att", "tag_def", "tag_set")
SHARED_SERIES = ("tag_unknown", "tag_any")
AGREEMENT = tuple("tag_agree_" + group for group in sorted(GROUPS))
# check(): agreement is measured with every tag moved by each of SHIFTS (frames) and averaged
# over SMOOTH neighbouring shifts. The team nearest the ball lags the tags by two to four
# seconds, so a group's agreement is its best within WINDOW frames of where open-play agreement
# peaks for look-alike tags (REFERENCE_DRAWS sets per training game). A group with fewer than
# MIN_FRAMES tagged frames with the ball seen is too small to judge, or to matter.
SHIFTS = np.arange(-500, 501, 5)
SMOOTH = 5
WINDOW = 100
REFERENCE_DRAWS = 16
MIN_FRAMES = 200
# A game's tags may run late or early against its tracking. offset() finds by how much: the
# move, up to OFFSET_REACH frames, that lines its agreement curve up with the look-alikes' mean
# curve, compared over |shift| <= MATCH_SPAN. On look-alike tags moved by 0, 1, 2 or 4 s it is
# unbiased and within 20 frames for 80% of draws. Estimates under ALIGN_MIN are left alone.
OFFSET_REACH = 250
MATCH_SPAN = 300
ALIGN_MIN = 25
# A group is judged only where its floor beats chance on the game's own tags: the median of
# NULL_DRAWS copies with random teams, put through the same steps. With a full match's open-play
# tags it does; with a few dozen tags -- a match's ~60 set pieces -- it does not, and the group
# follows the judged ones instead.
NULL_DRAWS = 20


def _runs(mask, minimum=1):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= minimum]


def _label_runs(labels):
    edges = np.flatnonzero(np.diff(np.concatenate(([-9], labels.astype(int), [-9]))) != 0)
    return [(a, b, int(labels[a])) for a, b in zip(edges[:-1], edges[1:])]


def candidates(tracking, labels):
    """Every interval the rules would tag, before thinning: [(start, stop, code, team or None)],
    from the labels and the clean tracking of a DFL game."""
    out = []
    for start, stop, _ in period_blocks(tracking.period):
        out += [(s + start, e + start, code, team)
                for s, e, code, team in _block(tracking, labels, start, stop)]
    return out


def _block(tracking, labels, start, stop):
    length, width = tracking.pitch
    y = labels[start:stop]
    ball = tracking.ball[start:stop].astype(np.float64)
    players = tracking.players[start:stop].astype(np.float64)
    directions = _attack_directions(players[..., 0], tracking.team)
    with np.errstate(invalid="ignore"):
        x_att = [np.nan_to_num(ball[:, 0] if directions[k] > 0 else length - ball[:, 0],
                               nan=length / 2) for k in (0, 1)]
        in_width = np.abs(ball[:, 1] - width / 2.0) < BOX_HALF_WIDTH
    runs = _label_runs(y)
    out = []
    for index, (a, b, k) in enumerate(runs):
        if k not in (H, A) or b - a < SECOND:
            continue
        other = 1 - k
        xa, xo = x_att[k][a:b], x_att[other][a:b]

        def spans(mask, minimum):
            return [(a + s, a + e) for s, e in _runs(mask, minimum)]

        out += [(s, e, "BALL IN FINAL THIRD", k) for s, e in spans(xa > 2 * length / 3, SECOND)]
        box = (xa > length - BOX_DEPTH) & in_width[a:b]
        out += [(s, e, "BALL IN THE BOX", k) for s, e in spans(box, SECOND // 2)]
        px = players[a:b, :, 0] if directions[k] > 0 else length - players[a:b, :, 0]
        with np.errstate(invalid="ignore"):
            in_box = ((tracking.team == k)[None, :] & (px > length - BOX_DEPTH)
                      & (np.abs(players[a:b, :, 1] - width / 2.0) < BOX_HALF_WIDTH)).sum(axis=1)
        out += [(s, e, "PLAYERS IN THE BOX", k) for s, e in spans(in_box >= 2, SECOND)]
        if xa[0] < length / 3:  # build up: from the own third until the ball reaches halfway
            ahead = np.flatnonzero(xa >= length / 2)
            end = a + (ahead[0] if len(ahead) else b - a)
            if end - a >= 2 * SECOND:
                out.append((a, end, "BUILD UP", k))
        middle = (xa >= length / 3) & (xa <= 2 * length / 3)
        out += [(s, e, "PROGRESSION", k) for s, e in spans(middle, 2 * SECOND)
                if x_att[k][e - 1] - x_att[k][s] > 10]
        if xa.max() > length - 20:
            out.append((max(a, b - 5 * SECOND), b, "CREATING CHANCES", k))
        if b - a > 2 * SECOND:
            travel = np.hypot(*(ball[a + 2 * SECOND:b] - ball[a:b - 2 * SECOND]).T)
            out += [(a + s, a + s + 2 * SECOND, "LONG BALL", k)
                    for s, _ in _runs(np.nan_to_num(travel) > LONG_BALL_M)]
        if index > 0 and runs[index - 1][2] == other:
            end = min(b, a + 5 * SECOND)
            out += [(a, end, "ATTACKING TRANSITION", k), (a, end, "DEFENSIVE TRANSITION", other)]
        for code, mask in (("DEFENDING IN DEFENSIVE THIRD", xo < length / 3),
                           ("DEFENDING IN MIDDLE THIRD", (xo >= length / 3) & (xo <= 2 * length / 3)),
                           ("DEFENDING IN ATTACKING THIRD", xo > 2 * length / 3)):
            out += [(s, e, code, other) for s, e in spans(mask, 3 * SECOND)]
    for index, (a, b, label) in enumerate(runs[:-1]):
        following = runs[index + 1][2]
        if label == D and following in (H, A):
            by = ball[min(b, len(ball) - 1), 1]
            throw_in = not np.isnan(by) and min(by, width - by) < 2.0
            out.append((max(0, b - 3 * SECOND), min(stop - start, b + 2 * SECOND), SET_PIECES,
                        None if throw_in else following))
    return out


def simulate(tracking, labels, rng, hard=True):
    """Look-alike Smart Tagging for a DFL game: the rules' intervals, thinned to the provided
    file's rate per code, then corrupted by amounts drawn from rng. hard adds the worse modes --
    for training copies; a held-out copy gets typical tags, so its score is a clean reference."""
    if hard and rng.random() < NONE_RATE:
        return None
    pool = candidates(tracking, labels)
    scale = float((tracking.period > 0).sum()) / PER_MATCH_FRAMES
    edge_sd, swap, drop, false = (rng.uniform(*r) for r in (EDGE_SD, SWAP_RATE, DROP_RATE,
                                                            FALSE_RATE))
    offset = int(rng.integers(*OFFSET)) if hard and rng.random() < OFFSET_RATE else 0
    if hard and rng.random() < BAD_SWAP_RATE:
        swap = rng.uniform(*BAD_SWAP)
    flipped = {group for group in sorted(GROUPS) if hard and rng.random() < FLIP_RATE}
    reversed_codes = {code for group in flipped for code in GROUPS[group]}
    tags = []
    for code, per_match in sorted(PER_MATCH.items()):
        mine = [t for t in pool if t[2] == code]
        keep = int(round(per_match * scale * (1 - drop)))
        if len(mine) > keep:
            mine = [mine[i] for i in sorted(rng.choice(len(mine), keep, replace=False))]
        tags += mine
    n, play = tracking.n_frames, np.flatnonzero(tracking.period > 0)
    false_tags = []
    for _ in range(int(round(false * len(tags)))):
        s, e, code, team = tags[rng.integers(len(tags))]
        at = int(rng.choice(play))
        false_tags.append((at, at + (e - s), code, None if team is None else int(rng.integers(2))))
    out = []
    for s, e, code, team in tags + false_tags:
        s = int(round(s + offset + rng.normal(0, edge_sd)))
        e = int(round(e + offset + rng.normal(0, edge_sd)))
        if team is not None and (rng.random() < swap) != (code in reversed_codes):
            team = 1 - team
        s, e = max(0, s), min(n, e)
        if e > s:
            out.append((s, e, code, team))
    return sorted(out)


def readable(tags):
    """The tags as a model reads them: set pieces without their team unless
    config.TAGS_SET_TEAMS."""
    if tags is None or config.TAGS_SET_TEAMS:
        return tags
    return [(s, e, code, None if code == SET_PIECES else team) for s, e, code, team in tags]


def series(tags, period):
    """Per-frame tag series: ([team 0 dict, team 1 dict], shared dict), NaN outside play and
    everywhere when tags is None (no tagging for this game).

    tag_att: an attacking-phase tag names this team; tag_def: a defending-phase tag does;
    tag_set: a set piece is this team's; tag_unknown: a tag with no team; tag_any: any tag.
    Read through readable(), so set pieces count as tags with no team unless TAGS_SET_TEAMS.
    """
    tags = readable(tags)
    n = len(period)
    if tags is None:
        empty = lambda: np.full(n, np.nan)  # noqa: E731
        return ([{key: empty() for key in TEAM_SERIES} for _ in (0, 1)],
                {key: empty() for key in SHARED_SERIES})
    team = [{key: np.zeros(n) for key in TEAM_SERIES} for _ in (0, 1)]
    shared = {key: np.zeros(n) for key in SHARED_SERIES}
    for start, stop, code, side in tags:
        shared["tag_any"][start:stop] = 1
        if side is None:
            shared["tag_unknown"][start:stop] = 1
            continue
        key = "tag_set" if code == SET_PIECES else "tag_def" if code in DEFENDING else "tag_att"
        team[side][key][start:stop] = 1
    outside = period == 0
    for store in team + [shared]:
        for values in store.values():
            values[outside] = np.nan
    return team, shared


def agreement(tags, near_team0, seen, period):
    """Per code group, the share of its tagged frames (ball seen) on which the team the tag says
    has the ball is the team nearest the ball: ({name: (T,) constant}), NaN outside play and
    without tags, 0.5 for a group with no tagged frames. near_team0 is features' "near" for team
    0 -- 1 where it is nearer."""
    n = len(period)
    out = {name: np.full(n, np.nan) for name in AGREEMENT}
    tags = readable(tags)
    if tags is None:
        return out
    nearer = np.where(seen & ~np.isnan(near_team0), (near_team0 < 0.5).astype(int), -1)
    for group, codes in sorted(GROUPS.items()):
        hits = total = 0
        for start, stop, code, team in tags:
            if team is None or code not in codes:
                continue
            holder = 1 - team if group == "def" else team
            window = nearer[start:stop]
            window = window[window >= 0]
            hits += int(np.sum(window == holder))
            total += len(window)
        # A group with no tagged frames is no evidence either way.
        out["tag_agree_" + group][period > 0] = hits / float(total) if total else 0.5
    return out


# -- the check: are this game's tags worth reading? ------------------------------------------------

def holders(tracking):
    """(T,) per frame, the team (0 or 1) whose nearest player is nearer the ball, -1 where the ball
    or every player is unseen or the frame is out of play: the tracking's own guess at who has
    the ball, which needs no labels."""
    distance = np.hypot(tracking.players[..., 0] - tracking.ball[:, None, 0],
                        tracking.players[..., 1] - tracking.ball[:, None, 1])
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest = [np.min(np.where((tracking.team == k)[None, :], distance, np.inf), axis=1)
               for k in (0, 1)]
    out = (nearest[1] < nearest[0]).astype(np.int64)
    out[np.isinf(nearest[0]) & np.isinf(nearest[1])] = -1
    out[tracking.period == 0] = -1
    return out


def curve(tags, holder, codes):
    """(agreement at each of SHIFTS, frames): over the tags with one of these codes and a team,
    the share of frames, ball seen, on which holder is the team the tag says has the ball, with
    every tag moved by the shift; frames counts them unshifted."""
    rows = [(s, e, 1 - team if code in DEFENDING else team) for s, e, code, team in tags
            if team is not None and code in codes]
    if not rows:
        return np.full(len(SHIFTS), np.nan), 0
    start, stop, side = (np.array(values) for values in zip(*rows))
    counts = [np.concatenate(([0], np.cumsum(holder == k))) for k in (0, 1)]
    n = len(holder)
    low = np.clip(start[:, None] + SHIFTS[None, :], 0, n)
    high = np.clip(stop[:, None] + SHIFTS[None, :], 0, n)
    seen = [counts[k][high] - counts[k][low] for k in (0, 1)]
    hits = np.where(side[:, None] == 0, seen[0], seen[1]).sum(axis=0)
    total = (seen[0] + seen[1]).sum(axis=0)
    values = np.where(total > 0, hits / np.maximum(total, 1).astype(np.float64), np.nan)
    padded = np.pad(values, SMOOTH // 2, mode="edge")
    smooth = np.convolve(padded, np.ones(SMOOTH) / SMOOTH, mode="valid")
    return smooth, int(total[len(SHIFTS) // 2])


def reference(samples):
    """What look-alike tags score, from [(holder, tags)] of the training games: the shift where
    open-play agreement peaks (the median over samples); per group the lowest peak agreement
    within WINDOW of it -- the floor a game's tags must reach -- and, for the open-play groups,
    the mean curve offset() lines a game's tags up with."""
    open_play = ATTACKING + DEFENDING
    peaks = [SHIFTS[int(np.nanargmax(curve(tags, holder, open_play)[0]))]
             for holder, tags in samples]
    centre = int(np.median(peaks))
    inside = np.abs(SHIFTS - centre) <= WINDOW
    floor, mean = {}, {}
    for group, codes in sorted(GROUPS.items()):
        best, curves = [], []
        for holder, tags in samples:
            values, frames = curve(tags, holder, codes)
            if frames >= MIN_FRAMES:
                best.append(float(np.nanmax(values[inside])))
                curves.append(values)
        floor[group] = min(best) if best else 0.5
        if group != "set" and curves:  # set pieces are too few to time anything by
            mean[group] = np.nanmean(curves, axis=0)
    return {"centre": centre, "floor": floor, "curves": mean, "samples": len(samples)}


def _match(values, mean, move):
    """Correlation of values(d) with mean(d + move) over |d| <= MATCH_SPAN."""
    k = move // 5
    a = values[max(0, -k):len(values) - max(0, k)]
    b = mean[max(0, k):len(mean) - max(0, -k)]
    d = SHIFTS[max(0, -k):len(SHIFTS) - max(0, k)]
    keep = ~np.isnan(a) & ~np.isnan(b) & (np.abs(d) <= MATCH_SPAN)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return 0.0
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def offset(tags, holder, ref):
    """How late (frames; negative is early) a game's tags run against what look-alike tags show:
    the move whose shifted reference curves best match this game's, weighted by tagged frames.
    Tags that are m frames late agree best m shifts earlier, so they are moved by minus it."""
    curves = [(curve(tags, holder, GROUPS[group]), ref["curves"][group])
              for group in sorted(ref["curves"])]
    best, out = -np.inf, 0
    for move in range(-OFFSET_REACH, OFFSET_REACH + 1, 5):
        fit = sum(frames * _match(values, mean, move) for (values, frames), mean in curves)
        if fit > best:
            best, out = fit, move
    return out


def moved(tags, frames):
    return [(s + frames, e + frames, code, team) for s, e, code, team in tags]


def _flipped(tags, groups):
    codes = {code for group in groups for code in GROUPS[group]}
    return [(s, e, code, 1 - team if team is not None and code in codes else team)
            for s, e, code, team in tags]


def _judge(tags, holder, ref):
    """Per code group: (verdict, best agreement near the look-alikes' peak, the same read
    reversed, tagged frames)."""
    inside = np.abs(SHIFTS - ref["centre"]) <= WINDOW
    out = {}
    for group, codes in sorted(GROUPS.items()):
        values, frames = curve(tags, holder, codes)
        peak = float(np.nanmax(values[inside])) if frames else float("nan")
        flipped = 1.0 - float(np.nanmin(values[inside])) if frames else float("nan")
        floor = ref["floor"][group]
        if frames < MIN_FRAMES:
            verdict = "keep"
        elif floor <= 0.5:  # look-alikes no better than chance: given and reversed look alike
            verdict = "fail"
        elif peak >= floor and peak >= flipped:
            verdict = "keep"
        elif flipped >= floor and flipped > peak:
            verdict = "flip"
        else:
            verdict = "fail"
        out[group] = verdict, peak, flipped, frames
    return out


def _assess(tags, holder, ref):
    """check()'s steps: the groups read reversed at first sight, the offset, and _judge() of the
    tags as flipped and aligned."""
    first = {group for group, (verdict, _, _, _) in _judge(tags, holder, ref).items()
             if verdict == "flip"}
    late = offset(_flipped(tags, first), holder, ref)
    late = late if abs(late) >= ALIGN_MIN else 0
    return first, late, _judge(moved(_flipped(tags, first), -late), holder, ref)


def check(tags, holder, ref):
    """How a game's tags may be read, judged against its own tracking. First, which code groups
    read reversed, judged where look-alike tags agree best; then, with those flipped, the offset
    that lines the tags up with the play (offset(); left alone under ALIGN_MIN); then each group
    judged again as aligned: "keep", "flip" or "fail". A group too small to judge (MIN_FRAMES, or
    a floor no better than chance, NULL_DRAWS) flips if every judged group does -- the clubs were
    matched the wrong way round -- and is kept otherwise. Set pieces, read without their team
    (config.TAGS_SET_TEAMS off), are never judged: "blind". use: some group was judged and none
    failed. A wrong club match, or a code whose Team means the other side, flips; random teams
    fail."""
    first, late, final = _assess(tags, holder, ref)
    rng = np.random.default_rng(0)
    nulls = {group: [] for group in GROUPS}
    for _ in range(NULL_DRAWS):
        shuffled = [(s, e, code, None if team is None else int(rng.integers(2)))
                    for s, e, code, team in tags]
        for group, (_, peak, flipped, frames) in _assess(shuffled, holder, ref)[2].items():
            if frames:
                nulls[group].append(max(peak, flipped))
    chance = {group: float(np.median(v)) if v else float("nan") for group, v in nulls.items()}
    judged = {}
    for group, (verdict, _, _, frames) in final.items():
        if (frames >= MIN_FRAMES and ref["floor"][group] > chance[group]
                and (group != "set" or config.TAGS_SET_TEAMS)):
            judged[group] = (verdict if verdict == "fail" else
                             "flip" if (verdict == "flip") != (group in first) else "keep")
    reversed_clubs = bool(judged) and all(v == "flip" for v in judged.values())
    plan = {"use": bool(judged) and "fail" not in judged.values(), "late": late, "groups": {},
            "judged": {}, "agreement": {}, "chance": {}, "frames": {}}
    for group, (_, peak, flipped, frames) in final.items():
        plan["groups"][group] = ("blind" if group == "set" and not config.TAGS_SET_TEAMS else
                                 judged.get(group, "flip" if reversed_clubs else "keep"))
        plan["judged"][group] = group in judged
        plan["agreement"][group] = round(max(peak, flipped), 4) if frames else float("nan")
        plan["chance"][group] = round(chance[group], 4)
        plan["frames"][group] = frames
    return plan


def apply(tags, plan, n_frames):
    """The tags as the model should read them: moved back by the plan's offset, clipped to the
    match, and each flipped group's teams reversed."""
    flipped = [group for group, verdict in plan["groups"].items() if verdict == "flip"]
    return [(max(0, s), min(n_frames, e), code, team)
            for s, e, code, team in moved(_flipped(tags, flipped), -plan["late"])
            if min(n_frames, e) > max(0, s)]


def read(events, home_name, away_name):
    """The provided file's tags as [(start, stop, code, team or None)], from
    starter/load_smart_tagging.load(). Club names are matched to the tracking's by similarity --
    the file spells Athletic "Atletic Club" -- and anything matching neither is None."""
    names = (home_name.lower(), away_name.lower())
    out = []
    for event in events:
        code = (event["code"] or "").strip().upper()
        tag = (event["tags"].get("Team") or "").strip().lower()
        team = None
        if tag and tag != "n/a":
            similar = [difflib.SequenceMatcher(None, tag, name).ratio() for name in names]
            team = int(np.argmax(similar)) if max(similar) > 0.5 else None
        out.append((event["start_frame"], event["end_frame"], code, team))
    return out
