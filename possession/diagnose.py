"""How the provided match differs from the copies the model learns from and is tested on. Counts,
shares and quantiles only -- no labels, nothing frame by frame -- so `diagnose`'s output can be
shared. Each view is computed the same way on the provided match and on the DFL games' held-out
copies (pipeline.TEST_COPY under the same profile), except the model's, which only the provided
match can give (the model was trained on the DFL games):

  ball      where the tracking loses the ball: gaps started per 1,000 seen frames, by the players
            within CROWD_M of the ball, its speed and its pitch zone at the last sighting; the
            ball's distance to the nearest player and its speed where seen; and how often it is
            seen where play has stopped -- beyond the lines, or still for STILL_S either side.
            Also on copies that keep the ball through every stoppage (DEAD_BALL_LOSS 0), so the
            provided match can be placed between the two: the copies lose it through
            config.DEAD_BALL_LOSS of them, an assumption only labels could measure.
  players   where the tracking stands still: frames that repeat the frame before exactly -- every
            player seen in both (REPEAT_MIN_PLAYERS or more) exactly where he was, as a frozen
            tracker would leave them -- how long those repeats run, the same for the ball, and the
            share of players slower than config.STILL_SPEED, as the model's `still` reads them.
  features  each model feature's shift between the two (rank AUC, 0.5 is none) and its missing
            shares, the most shifted first; and how well a classifier tells the two apart.
  model     possessions per 90 minutes of play and their lengths, the dead-ball share, how sure
            the blend is, how many possessions start where the ball is not seen, and the share
            called dead by how long the ball has been unseen (GAP_BINS_S)."""
import numpy as np

from possession import config, features, pipeline
from possession.scoring import D, pm
from possession.tracking import period_blocks

# Params
CROWD_M = 3.0                          # noise.CROWD_RADIUS: the copies' losses rise with it
SPEED_EDGES = (5.0, 15.0, 25.0)        # m/s, the ball's speed at its last sighting
WIDE_M = 20.16                         # half the penalty box's width: wider than this is wide
QUANTILES = (10, 25, 50, 75, 90)
FEATURE_SAMPLE = 20000                 # graded frames per side for the feature view
CLASSIFIER_SAMPLE = 10000              # of those, per side, for the classifier
TOP_FEATURES = 20
LOW_CONFIDENCE = 0.6                   # a blended top probability below this is unsure
OUT_M = 1.0                            # seen this far beyond a line: out of play
STILL_S, STILL_M = 1.0, 1.0            # seen STILL_S either side, within STILL_M: a placed ball
GAP_BINS_S = (2.0, 10.0)               # the ball's gaps, by length
REPEAT_MIN_PLAYERS = 10                # players seen in both frames for a repeat to count
REPEAT_RUNS = (1, 5, 25)               # frames: repeat runs at least this long
SEED = 7


def _r(value, digits=3):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in value]
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _hazards(starts, seen, keys):
    """{key: [gaps started per 1,000 seen frames, share of seen frames]} per bin."""
    total = max(int(seen.sum()), 1)
    out = {}
    for key, mask in keys.items():
        n = int((seen & mask).sum())
        out[key] = [_r(1000.0 * (starts & mask).sum() / n if n else np.nan, 2), _r(n / total)]
    return out


def ball_view(trackings):
    """The ball view, pooled over trackings."""
    pooled = {k: [] for k in ("seen", "start", "near", "speed", "x", "y", "nearest", "out",
                              "still")}
    for tracking in trackings:
        length, width = tracking.pitch
        fps = float(tracking.fps)
        for start, stop, _ in period_blocks(tracking.period):
            ball = tracking.ball[start:stop].astype(np.float64)
            players = tracking.players[start:stop].astype(np.float64)
            seen = ~np.isnan(ball[:, 0])
            with np.errstate(invalid="ignore"):
                distance = np.hypot(players[..., 0] - ball[:, None, 0],
                                    players[..., 1] - ball[:, None, 1])
                near = (distance < CROWD_M).sum(axis=1)
            nearest = np.full(len(ball), np.nan)
            some = seen & ~np.all(np.isnan(distance), axis=1)
            nearest[some] = np.nanmin(distance[some], axis=1)
            speed = np.full(len(ball), np.nan)  # backward over two frames: known at a last sighting
            both = seen[2:] & seen[:-2]
            speed[2:][both] = np.hypot(*(ball[2:] - ball[:-2])[both].T) / 2.0 * fps
            # A gap starts after frame t when t is seen and t + 1 is not; the block's last frame
            # cannot start one, so it is left out of the seen frames that could.
            start_next = np.zeros(len(ball), dtype=bool)
            start_next[:-1] = seen[:-1] & ~seen[1:]
            could = seen.copy()
            could[-1] = False
            with np.errstate(invalid="ignore"):
                out = seen & ((ball[:, 0] < -OUT_M) | (ball[:, 0] > length + OUT_M)
                              | (ball[:, 1] < -OUT_M) | (ball[:, 1] > width + OUT_M))
            lag = int(round(STILL_S * fps))
            still = np.zeros(len(ball), dtype=bool)
            if len(ball) > 2 * lag:
                mid = ball[lag:-lag]
                with np.errstate(invalid="ignore"):
                    still[lag:-lag] = ((np.hypot(*(ball[2 * lag:] - mid).T) < STILL_M)
                                       & (np.hypot(*(mid - ball[:-2 * lag]).T) < STILL_M))
            for key, value in (("seen", could), ("start", start_next), ("near", near),
                               ("speed", speed), ("x", ball[:, 0] / length),
                               ("y", np.abs(ball[:, 1] - width / 2.0)), ("nearest", nearest),
                               ("out", out), ("still", still)):
                pooled[key].append(value)
    p = {k: np.concatenate(v) for k, v in pooled.items()}
    seen, starts = p["seen"], p["start"]
    edges = (0.0,) + SPEED_EDGES + (np.inf,)
    with np.errstate(invalid="ignore"):
        speed_bins = {"{:g}-{:g}".format(a, b) if np.isfinite(b) else ">{:g}".format(a):
                      (p["speed"] >= a) & (p["speed"] < b) for a, b in zip(edges[:-1], edges[1:])}
        speed_bins["unknown"] = np.isnan(p["speed"])
        middle = (p["x"] >= 1 / 3.0) & (p["x"] < 2 / 3.0)
    seen_any = ~np.isnan(p["nearest"])
    return {
        "seen_frames": int(seen.sum()),
        "gaps_per_1000_seen": _r(1000.0 * starts.sum() / max(seen.sum(), 1), 2),
        "by_players_within_3m": _hazards(starts, seen, {
            "0": p["near"] == 0, "1": p["near"] == 1, "2": p["near"] == 2, "3+": p["near"] >= 3}),
        "by_ball_speed_ms": _hazards(starts, seen, speed_bins),
        "by_zone": _hazards(starts, seen, {"end thirds": ~middle, "middle third": middle,
                                           "wide": p["y"] > WIDE_M, "central": p["y"] <= WIDE_M}),
        "nearest_player_m_quantiles": _r(np.percentile(p["nearest"][seen_any], QUANTILES), 2),
        "ball_speed_ms_quantiles": _r(np.nanpercentile(p["speed"], QUANTILES), 2),
        "seen_out_of_pitch_share": _r(p["out"].sum() / max(seen.sum(), 1), 4),
        "seen_still_share": _r(p["still"].sum() / max(seen.sum(), 1), 4),
    }


def player_view(trackings):
    """The players view, pooled over trackings."""
    counted = repeats = ball_seen = ball_repeats = slow = timed = 0
    runs = []
    for tracking in trackings:
        fps = float(tracking.fps)
        for start, stop, _ in period_blocks(tracking.period):
            players = tracking.players[start:stop].astype(np.float64)
            ball = tracking.ball[start:stop].astype(np.float64)
            both = ~np.isnan(players[1:, :, 0]) & ~np.isnan(players[:-1, :, 0])
            same = both & np.all(players[1:] == players[:-1], axis=2)
            enough = both.sum(axis=1) >= REPEAT_MIN_PLAYERS
            repeat = enough & (same.sum(axis=1) == both.sum(axis=1))
            counted, repeats = counted + int(enough.sum()), repeats + int(repeat.sum())
            runs.append(_runs(repeat))
            seen = ~np.isnan(ball[1:, 0]) & ~np.isnan(ball[:-1, 0])
            ball_seen += int(seen.sum())
            ball_repeats += int((seen & np.all(ball[1:] == ball[:-1], axis=1)).sum())
            velocity = features._centred_diff(players, config.VELOCITY_HALF) * fps
            speed = np.hypot(velocity[..., 0], velocity[..., 1])
            known = ~np.isnan(speed)
            slow, timed = slow + int((speed[known] < config.STILL_SPEED).sum()), timed + int(known.sum())
    runs = np.concatenate(runs) if runs else np.zeros(0, dtype=int)
    return {
        "frames_counted": counted,
        "all_players_repeat_share": _r(repeats / max(counted, 1), 4),
        "repeat_runs_per_1000_frames": {">={}".format(n): _r(1000.0 * (runs >= n).sum()
                                                              / max(counted, 1), 2)
                                        for n in REPEAT_RUNS},
        "longest_repeat_run": int(runs.max()) if len(runs) else 0,
        "ball_repeat_share": _r(ball_repeats / max(ball_seen, 1), 4),
        "players_still_share": _r(slow / max(timed, 1), 3),
    }


def _runs(mask):
    """Lengths of mask's runs of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[1::2] - edges[::2]


def _graded_sample(matrix, period, n, rng):
    rows = np.flatnonzero(period > 0)
    return matrix[rng.choice(rows, min(n, len(rows)), replace=False)].astype(np.float32)


def feature_view(provided, copies, names):
    """provided, copies: (rows, F) float32 samples of graded frames."""
    from scipy.stats import rankdata
    rows = []
    for j, name in enumerate(names):
        a, b = provided[:, j], copies[:, j]
        miss_a, miss_b = float(np.isnan(a).mean()), float(np.isnan(b).mean())
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        auc = np.nan
        if len(a) and len(b):
            ranks = rankdata(np.concatenate([a, b]))
            auc = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b))
        shift = max(abs(auc - 0.5) if np.isfinite(auc) else 0.0, abs(miss_a - miss_b))
        rows.append((shift, name, auc, np.median(a) if len(a) else np.nan,
                     np.median(b) if len(b) else np.nan, miss_a, miss_b))
    rows.sort(key=lambda r: -r[0])
    shifts = np.array([r[0] for r in rows])
    return {
        "columns": ["feature", "auc_provided_over_copies", "median_provided", "median_copies",
                    "missing_provided", "missing_copies"],
        "most_shifted": [[r[1]] + _r(list(r[2:])) for r in rows[:TOP_FEATURES]],
        "features": len(names),
        "shifted_over_0.1_0.2_0.3": [int((shifts > t).sum()) for t in (0.1, 0.2, 0.3)],
        "classifier_auc": _r(_classifier_auc(provided, copies)),
    }


def _classifier_auc(provided, copies):
    """Three-fold AUC of a boosted-tree classifier telling provided frames from copies' frames."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(SEED)
    a = provided[rng.permutation(len(provided))[:CLASSIFIER_SAMPLE]]
    b = copies[rng.permutation(len(copies))[:CLASSIFIER_SAMPLE]]
    x = np.vstack([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    model = HistGradientBoostingClassifier(max_iter=100, random_state=SEED)
    return float(np.mean(cross_val_score(model, x, y, cv=3, scoring="roc_auc")))


def model_view(codes, probs, tracking):
    """The model's possessions on one match, from its labels and the blended P(H, A, D)."""
    graded = tracking.period > 0
    fps = float(tracking.fps)
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    seconds = np.array([(b - a) / fps for a, b, _ in spells])
    minutes = graded.sum() / fps / 60.0
    seen = ~np.isnan(tracking.ball[:, 0])
    starts = np.array([a for a, _, _ in spells], dtype=int)
    joined = sum(1 for (_, b, _), (a, _, _) in zip(spells[:-1], spells[1:]) if a == b)
    top = probs[graded].max(axis=1)
    gap = np.zeros(len(seen))  # per frame, the length in seconds of the gap it is in; 0 if seen
    edges = np.flatnonzero(np.diff(np.r_[0, (~seen).astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        gap[a:b] = (b - a) / fps
    bins = (0.0,) + GAP_BINS_S + (np.inf,)
    by_gap = {"seen": graded & seen}
    for a, b in zip(bins[:-1], bins[1:]):
        by_gap["gap {:g}-{:g}s".format(a, b) if np.isfinite(b) else "gap >{:g}s".format(a)] = (
            graded & ~seen & (gap > a) & (gap <= b))
    return {
        "possessions": len(spells),
        "possessions_per_90": _r(len(spells) / minutes * 90.0, 1),
        "seconds_quantiles": _r(np.percentile(seconds, QUANTILES), 2) if len(seconds) else None,
        "share_under_2s_5s": _r([(seconds < 2).mean(), (seconds < 5).mean()]) if len(seconds) else None,
        "dead_share_of_graded": _r((codes[graded] == D).mean()),
        "top_probability_mean": _r(top.mean()),
        "share_unsure": _r((top < LOW_CONFIDENCE).mean()),
        "share_starting_ball_unseen": _r((~seen[starts]).mean()) if len(starts) else None,
        "ball_unseen_share_of_graded": _r((~seen[graded]).mean()),
        "share_straight_from_other_team": _r(joined / max(len(spells) - 1, 1)),
        "dead_share_by_ball": {key: [_r(mask.sum() / max(graded.sum(), 1)),
                                     _r((codes[mask] == D).mean()) if mask.any() else None]
                               for key, mask in by_gap.items()},
    }


def report(bundle, match, games, profile, log=pipeline.log):
    """The three views: match the provided Tracking (as `predict` reads it), games the labelled
    DFL games whose held-out copies stand for what the model was tested on."""
    rng = np.random.default_rng(SEED)
    copies = []
    for tracking, labels in games:
        copies.append(pipeline.corrupt(tracking, labels, profile, pipeline.TEST_COPY))
        log("diagnose   copy of {}".format(tracking.name))
    kept = []
    loss, config.DEAD_BALL_LOSS = config.DEAD_BALL_LOSS, 0.0
    try:
        for tracking, labels in games:
            kept.append(pipeline.corrupt(tracking, labels, profile, pipeline.TEST_COPY))
    finally:
        config.DEAD_BALL_LOSS = loss
    log("diagnose   copies that keep the ball through stoppages")
    out = {"ball": {"provided": ball_view([match]), "copies": ball_view(copies),
                    "copies_ball_kept_at_stoppages": ball_view(kept)},
           "players": {"provided": player_view([match]), "copies": player_view(copies)}}
    with pipeline.using_tags(False):  # tracking alone: the tags are checked separately
        fs = pipeline.compute_features(match)
        names = list(fs.names())
        provided = _graded_sample(fs.matrix((0, 1)), match.period, FEATURE_SAMPLE, rng)
        per_copy = FEATURE_SAMPLE // max(len(copies), 1)
        samples = []
        for copy in copies:
            matrix = pipeline.compute_features(copy).matrix((0, 1))
            samples.append(_graded_sample(matrix, copy.period, per_copy, rng))
    log("diagnose   features compared")
    out["features"] = feature_view(provided, np.vstack(samples), names)
    probs, boundary = pipeline.predict_probs(bundle, match)
    codes = pipeline.decode(bundle, probs, boundary, match.period)
    out["model"] = model_view(codes, probs, match)
    log("diagnose   model read")
    return out


def dumps(out, indent=""):
    """report()'s dict as JSON, one line per figure and per feature row, so it pastes compactly."""
    import json
    lines = []
    for i, (key, value) in enumerate(out.items()):
        comma = "," if i < len(out) - 1 else ""
        if isinstance(value, dict):
            lines.append('{} "{}": {}{}'.format(indent, key, dumps(value, indent + " "), comma))
        elif key == "most_shifted":
            rows = ",\n".join("{}   {}".format(indent, json.dumps(r)) for r in value)
            lines.append('{} "{}": [\n{}\n{} ]{}'.format(indent, key, rows, indent, comma))
        else:
            lines.append('{} "{}": {}{}'.format(indent, key, json.dumps(value), comma))
    return "{\n" + "\n".join(lines) + "\n" + indent + "}"
