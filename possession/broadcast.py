"""The broadcast recording's ball on the tactical timeline: where the wide tactical camera has lost
the ball, the broadcast's ball-following camera may have it (DATA.md: the ball is seen on 56.9% of
tactical frames, 50.5% of broadcast ones).

The two recordings share pitch coordinates but not a timeline: the broadcast has its own frame
count, cuts, replays and close-ups, and no mapping is provided. So the alignment is read off the
tracking itself, in two stages:

1. Anchors. Ten-second windows of the broadcast's ball are slid along the tactical ball -- every
   offset at once, as FFT cross-correlations of the masked tracks -- and each keeps its best
   offset if that fits closely and no other offset comes near. A live feed runs in real time, so
   the anchors of live play share one offset per unbroken stretch of the feed, while a replay's
   anchor matches its own earlier moment and stands alone: offsets shared by CLUSTER_MIN anchors
   are the feed's alignments. The pitch orientation (either axis may be flipped) is the one with
   the most anchors.
2. Frames. Every broadcast frame is scored at each alignment by its players: the median distance
   from each player the broadcast shows to the nearest one the tactical tracking has, through a
   SMOOTH-frame running median, refined FINE frames either side. A frame is live, and mapped, when
   its best score is under MATCH_M metres; replays, close-ups and cut-aways match nothing.

The broadcast ball then fills tactical frames whose ball is missing, shifted by the lag at which
the two balls agree best where both are seen. Nothing here reads a label, and the same files give
the same fill (config.USE_BROADCAST; `python -m possession broadcast` prints the checks).
"""

import warnings
import xml.etree.ElementTree as ET

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from possession import config, guard
from possession.tracking import read_atd

# Params
ANCHOR = 250          # frames per anchor window (10 s)
ANCHOR_STEP = 250     # broadcast frames between anchor starts
ANCHOR_SEEN = 0.5     # an anchor needs its ball seen on this share of its frames
ANCHOR_MOVE_M = 15.0  # and a ball path this long, so that one offset stands out
MIN_OVERLAP = 0.3     # an offset is scored only where both balls are seen on this share of the window
ANCHOR_RMS_M = 2.0    # an anchor's best offset must fit to this RMS ball distance
ANCHOR_MARGIN = 3.0   # and every offset more than a window away must fit this many times worse
CLUSTER_FRAMES = 5    # anchor offsets this close are one alignment
CLUSTER_MIN = 3       # anchors an alignment needs
FINE = 2              # frames searched either side of an alignment
REFINE = 1500         # broadcast frames (a minute) sharing one refined shift
SMOOTH = 25           # frames in the running median of the players' score
MATCH_M = 2.0         # a broadcast frame is live when its smoothed score is under this
FRAME_MAX_M = 4.0     # and its own score, where it has one, under this
MIN_PLAYERS = 4       # players a broadcast frame must show to be scored
BALL_MATCH_M = 3.0    # and its ball, where both are seen, within this of the tactical one
MAX_LAG = 10          # frames searched for the lag between the two balls
CHUNK = 8192          # frames per block of the players' distances
FLIPS = ((False, False), (True, True), (True, False), (False, True))


def load(game=None, data_dir=None):
    """The broadcast ATD as a Tracking on its own timeline (period 1 throughout: it has no
    published halves), at the frame rate its metadata gives, else config.FPS."""
    from games import GAME, GAMES, require, source_frames
    game = game or GAME
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, "broadcast")
    n = source_frames("broadcast", game)
    tracking = read_atd(meta_path, raw_path, n_frames=n, pitch=GAMES[game]["pitch"],
                        blocks=[(1, 0, n - 1)], name=game + " broadcast")
    tracking.fps = frame_rate(meta_path) or config.FPS
    return tracking


def frame_rate(meta_path):
    """The metadata's FrameRate, or None."""
    guard.check_input(meta_path)
    for node in ET.parse(meta_path).getroot().iter():
        if node.tag.lower().endswith("framerate") and (node.text or "").strip():
            try:
                return float(node.text)
            except ValueError:
                return None
    return None


def fused(tactical, game=None, data_dir=None):
    """tactical with its ball's gaps filled from the broadcast; unchanged, with a note, when the
    broadcast files are not there."""
    from possession import pipeline
    try:
        other = load(game, data_dir)
    except IOError as problem:
        pipeline.log("note       no broadcast recording, so no ball from it ({})".format(problem))
        return tactical
    return fill(tactical, align(tactical, other))[0]


# -- alignment ---------------------------------------------------------------------------------

def align(tactical, broadcast):
    """What fill() and describe() read: the orientation, the anchors, the alignments, the per-frame
    map (t2b[j], the broadcast frame live on tactical frame j, or -1) and the balls' lag."""
    bp = _on_rate(broadcast.players, broadcast.fps, tactical.fps)
    bb = _on_rate(broadcast.ball, broadcast.fps, tactical.fps)
    best = None
    for flip in FLIPS:
        found = anchors(tactical.ball, _orient(bb, tactical.pitch, flip))
        sure = [a for a in found if confident(a)]
        key = (len(sure), -float(np.median([a[2] for a in sure])) if sure else 0.0)
        if best is None or key > best[0]:
            best = (key, flip, found)
    _, flip, found = best
    bp, bb = _orient(bp, tactical.pitch, flip), _orient(bb, tactical.pitch, flip)
    offsets = alignments(found)
    t2b = np.full(tactical.n_frames, -1, dtype=np.int64)
    live = np.zeros(len(bb), dtype=bool)
    lag = 0
    if offsets:
        shift, cost = frame_offsets(tactical.players, bp, [o for o, _ in offsets])
        live = cost < MATCH_M
        for d in np.unique(shift):  # and each frame on its own, so a replay's edges do not pass
            rows = shift == d
            own = _player_score(tactical.players, bp, int(d))[rows]
            live[rows] &= ~(own > FRAME_MAX_M)
        rows = np.flatnonzero(live)
        t2b[rows + shift[rows]] = rows
        lag = ball_lag(tactical.ball, bb, t2b)
        # Where both balls are seen, they must agree too: a replay can look like live play
        # by its players alone, never by where its ball is.
        apart = _smooth(_ball_gap(tactical.ball, bb, shift, lag))
        live &= ~(apart > BALL_MATCH_M)
        t2b[:] = -1
        rows = np.flatnonzero(live)
        t2b[rows + shift[rows]] = rows
    return {"flip": flip, "fps": float(broadcast.fps), "anchors": found, "offsets": offsets,
            "live": live, "t2b": t2b, "lag": lag, "ball": bb}


def anchors(tactical_ball, broadcast_ball):
    """[(start, offset, rms, next_rms)] for each window of the broadcast ball distinctive enough
    to place: offset = tactical frame - broadcast frame at its best fit, rms that fit's RMS ball
    distance (metres), next_rms the best fit more than a window away."""
    n_t, width = len(tactical_ball), ANCHOR
    if n_t < width:
        return []
    size = 1 << int(np.ceil(np.log2(n_t + width)))
    seen_t = np.isfinite(tactical_ball[:, 0])
    ball_t = np.where(seen_t[:, None], tactical_ball, 0.0).astype(np.float64)
    spectra = {"m": np.fft.rfft(seen_t.astype(np.float64), size),
               "sq": np.fft.rfft((ball_t ** 2).sum(axis=1), size),
               "x": np.fft.rfft(ball_t[:, 0], size), "y": np.fft.rfft(ball_t[:, 1], size)}
    length = n_t - width + 1
    out = []
    for start in range(0, len(broadcast_ball) - width + 1, ANCHOR_STEP):
        window = broadcast_ball[start:start + width].astype(np.float64)
        seen = np.isfinite(window[:, 0])
        if seen.mean() < ANCHOR_SEEN:
            continue
        path = np.diff(window[seen], axis=0)
        if np.hypot(path[:, 0], path[:, 1]).sum() < ANCHOR_MOVE_M:
            continue
        ball = np.where(seen[:, None], window, 0.0)
        mask = seen.astype(np.float64)
        overlap = _correlate(mask, spectra["m"], size, length)
        total = (_correlate((ball ** 2).sum(axis=1), spectra["m"], size, length)
                 + _correlate(mask, spectra["sq"], size, length)
                 - 2.0 * (_correlate(ball[:, 0], spectra["x"], size, length)
                          + _correlate(ball[:, 1], spectra["y"], size, length)))
        msd = np.where(overlap >= MIN_OVERLAP * width - 0.5, total / np.maximum(overlap, 1.0),
                       np.inf)
        fit = int(np.argmin(msd))
        if not np.isfinite(msd[fit]):
            continue
        far = np.abs(np.arange(length) - fit) > width
        next_fit = np.min(msd[far]) if far.any() else np.inf
        out.append((start, fit - start, float(np.sqrt(max(msd[fit], 0.0))),
                    float(np.sqrt(max(next_fit, 0.0)))))
    return out


def confident(anchor):
    _, _, rms, next_rms = anchor
    return rms <= ANCHOR_RMS_M and next_rms >= ANCHOR_MARGIN * max(rms, 0.5)


def alignments(found):
    """[(offset, anchors)]: offsets shared, to CLUSTER_FRAMES, by CLUSTER_MIN confident anchors."""
    offsets = sorted(a[1] for a in found if confident(a))
    groups = []
    for offset in offsets:
        if groups and offset - groups[-1][-1] <= CLUSTER_FRAMES:
            groups[-1].append(offset)
        else:
            groups.append([offset])
    return [(int(np.median(g)), len(g)) for g in groups if len(g) >= CLUSTER_MIN]


def frame_offsets(tactical_players, broadcast_players, offsets):
    """((Tb,) offset, (Tb,) smoothed score) per broadcast frame: the best of the alignments, then,
    per REFINE frames of the feed, the shift within FINE frames that scores best over all of them
    (a frame on its own cannot tell players a frame apart); the score is inf where nothing could
    be scored."""
    scores = np.stack([_smooth(_player_score(tactical_players, broadcast_players, d))
                       for d in offsets])
    scores = np.where(np.isfinite(scores), scores, np.inf)
    pick = np.argmin(scores, axis=0)
    base = np.array(offsets)[pick]
    shift, best = base.copy(), scores[pick, np.arange(len(pick))]
    for d in offsets:
        rows = base == d
        if not rows.any():
            continue
        steps = range(-FINE, FINE + 1)
        raw = np.stack([_player_score(tactical_players, broadcast_players, d + step)
                        for step in steps])
        for a in range(0, len(base), REFINE):
            block = np.flatnonzero(rows[a:a + REFINE]) + a
            scored = block[np.isfinite(raw[:, block]).all(axis=0)]
            if len(scored) < SMOOTH:
                continue
            step = list(steps)[int(np.argmin(raw[:, scored].mean(axis=1)))]
            if step:
                shift[block] = d + step
                best[block] = np.where(np.isfinite(_smooth(raw[step + FINE]))[block],
                                       _smooth(raw[step + FINE])[block], np.inf)
    return shift, best


def ball_lag(tactical_ball, broadcast_ball, t2b):
    """The lag (broadcast frames) at which the broadcast ball best matches the tactical one on
    mapped frames where both are seen; 0 when there are none."""
    rows = np.flatnonzero((t2b >= 0) & np.isfinite(tactical_ball[:, 0]))
    best, best_lag = np.inf, 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        source = t2b[rows] + lag
        ok = (source >= 0) & (source < len(broadcast_ball))
        other = broadcast_ball[source[ok]]
        both = np.isfinite(other[:, 0])
        if both.sum() < 100:
            continue
        gap = np.median(np.hypot(*(tactical_ball[rows[ok][both]] - other[both]).T))
        if gap < best:
            best, best_lag = gap, lag
    return best_lag


# -- the fill and its checks -------------------------------------------------------------------

def fill(tactical, alignment):
    """(tactical with the broadcast ball wherever its own is missing and a live broadcast frame
    has one, (T,) bool of the frames filled)."""
    source = _sources(tactical.n_frames, alignment)
    ball = tactical.ball.copy()
    other = np.full_like(ball, np.nan)
    ok = source >= 0
    other[ok] = alignment["ball"][source[ok]]
    filled = np.isnan(ball[:, 0]) & np.isfinite(other[:, 0])
    ball[filled] = other[filled]
    return tactical.replace(ball=ball), filled


def describe(tactical, alignment):
    """Aggregate checks of an alignment, safe to share: counts and shares, no positions."""
    graded = tactical.period > 0
    source = _sources(tactical.n_frames, alignment)
    other = np.full_like(tactical.ball, np.nan)
    other[source >= 0] = alignment["ball"][source[source >= 0]]
    both = np.isfinite(tactical.ball[:, 0]) & np.isfinite(other[:, 0])
    gap = np.hypot(*(tactical.ball[both] - other[both]).T)
    filled = fill(tactical, alignment)[0]
    sure = [a for a in alignment["anchors"] if confident(a)]
    return {
        "broadcast_fps": alignment["fps"],
        "pitch_flipped_xy": [bool(f) for f in alignment["flip"]],
        "anchors_tried": len(alignment["anchors"]),
        "anchors_placed": len(sure),
        "alignments_offset_anchors": [[int(o), int(n)] for o, n in alignment["offsets"]],
        "broadcast_live_share": round(float(alignment["live"].mean()), 4),
        "graded_frames_mapped_share": round(float((alignment["t2b"][graded] >= 0).mean()), 4),
        "ball_lag_frames": int(alignment["lag"]),
        "balls_both_seen_frames": int(both.sum()),
        "ball_agreement_m_median_p90": ([round(float(np.median(gap)), 3),
                                         round(float(np.percentile(gap, 90)), 3)]
                                        if len(gap) else None),
        "ball_missing_graded_before": round(float(np.isnan(tactical.ball[graded, 0]).mean()), 4),
        "ball_missing_graded_after": round(float(np.isnan(filled.ball[graded, 0]).mean()), 4),
    }


def _sources(n_frames, alignment):
    """(T,) the broadcast frame whose ball stands for each tactical frame (lag applied), or -1."""
    t2b = alignment["t2b"]
    source = np.where(t2b >= 0, t2b + alignment["lag"], -1)
    source[(source < 0) | (source >= len(alignment["ball"]))] = -1
    return source


def _ball_gap(tactical_ball, broadcast_ball, shift, lag):
    """(Tb,) distance between broadcast frame i + lag's ball and tactical frame i + shift[i]'s;
    NaN where either is not seen."""
    rows = np.arange(len(shift))
    target, source = rows + shift, rows + lag
    ok = (target >= 0) & (target < len(tactical_ball)) & (source >= 0) & (source < len(broadcast_ball))
    out = np.full(len(shift), np.nan)
    gap = tactical_ball[target[ok]] - broadcast_ball[source[ok]]
    out[ok] = np.hypot(gap[:, 0], gap[:, 1])
    return out


def _player_score(tactical_players, broadcast_players, offset):
    """(Tb,) at one offset: the median over the players a broadcast frame shows of the distance to
    the nearest tactical player on frame + offset; NaN under MIN_PLAYERS or off the end."""
    n_b, n_t = len(broadcast_players), len(tactical_players)
    out = np.full(n_b, np.nan)
    lo, hi = max(0, -offset), min(n_b, n_t - offset)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in range(lo, hi, CHUNK):
            b = min(hi, a + CHUNK)
            ours = broadcast_players[a:b].astype(np.float32)
            theirs = tactical_players[a + offset:b + offset].astype(np.float32)
            dist = np.sqrt(((ours[:, :, None, :] - theirs[:, None, :, :]) ** 2).sum(axis=-1))
            nearest = np.nanmin(dist, axis=2)
            shown = np.isfinite(nearest).sum(axis=1)
            out[a:b] = np.where(shown >= MIN_PLAYERS, np.nanmedian(nearest, axis=1), np.nan)
    return out


def _smooth(score):
    """A SMOOTH-frame running median that fills a frame without a score only between scored
    frames, never past the last one (a half's end, the start of a close-up)."""
    half = SMOOTH // 2
    padded = np.pad(score, half, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(sliding_window_view(padded, 2 * half + 1), axis=1)
    count = np.concatenate([[0], np.cumsum(np.isfinite(score))])
    rows = np.arange(len(score))
    before = count[rows] - count[np.maximum(rows - half, 0)]
    after = count[np.minimum(rows + half + 1, len(score))] - count[rows + 1]
    inside = np.isfinite(score) | ((before > 0) & (after > 0))
    return np.where(inside, out, np.nan)


def _correlate(window, spectrum, size, length):
    """out[o] = sum_i window[i] * g[i + o] for o < length, given spectrum = rfft(g, size)."""
    full = np.fft.irfft(np.fft.rfft(window[::-1], size) * spectrum, size)
    return full[len(window) - 1:len(window) - 1 + length]


def _orient(positions, pitch, flip):
    out = np.array(positions, dtype=np.float32, copy=True)
    for axis, (flipped, extent) in enumerate(zip(flip, pitch)):
        if flipped:
            out[..., axis] = extent - out[..., axis]
    return out


def _on_rate(values, fps_from, fps_to):
    """values resampled from fps_from to fps_to by nearest frame (unchanged when they agree)."""
    if abs(fps_from - fps_to) < 1e-6:
        return values
    n = int(len(values) * fps_to / fps_from)
    index = np.clip(np.round(np.arange(n) * fps_from / fps_to).astype(int), 0, len(values) - 1)
    return values[index]
