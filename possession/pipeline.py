"""Train, predict, evaluate: the whole method, as library functions the CLI wraps.

Training uses labelled OTHER games only, and is cross-fitted by game:

  0. Each training game is corrupted to the provided ATD's noise profile, NOISE_COPIES times. The
     carrier model (who is on the ball) is fitted on those copies, once per left-out game.
  1. Each copy is turned into a table of feature rows, carrier features included.
  2. For each game g, the experts are trained on every other game and predict game g. That gives
     out-of-fold expert outputs for every game.
  3. The gate is trained on those out-of-fold outputs. For the decoder search, game g's gate is
     itself trained without game g.
  4. The decoder's few numbers are searched on the out-of-fold gate output.
  5. Final experts and gate are refit on all games and saved with everything needed to rerun.

Prediction is the same chain on the provided ATD, uncorrupted, with no fitting at all.

evaluate() wraps train() in an outer leave-one-game-out loop, so the held-out game is unseen by
every stage -- experts, gate and decoder search -- and its score is an honest estimate. It is the
number to compare versions on; the public window is not (EVALUATION.md, RULES.md). Inside it, the
expert bank fitted without games g and h is needed twice -- once holding out g, once holding out
h -- so evaluate() keeps fitted banks on disk (BankCache) and fits each once. train() for the
submission never uses that cache.
"""

import contextlib
import hashlib
import inspect
import os
import pickle
import platform
import sys
import time
import zlib

import numpy as np
import sklearn

from possession import config, decoder, experts, features, matches, noise, tagging
from possession.carrier import Carrier
from possession.experts import SWAP_BOUNDARY, SWAP_STATE, SWAP_TEAMS, ExpertBank
from possession.features import _shift
from possession.gate import GATE_OUTPUTS, Gate, context_names
from possession.scoring import A, D, H, X, score
from possession.tracking import period_blocks

# Params
TEST_COPY = 99  # noise seed for a held-out game, distinct from every training copy
TAG_COPY = 98   # noise seed for the training-game copies that set the Smart Tagging check
BANK_DIR = os.path.join(config.CACHE_DIR, "banks")
# Settings a fitted expert bank does not depend on: the later stages', and those it depends on
# only through its tables' contents. USE_TAGS, which TAGS_CHECK sets, is read when a bank is
# fetched.
AFTER_BANK = ("GATE_", "DECODER_", "THREADS", "TAGS_CHECK", "TAGS_AGREEMENT", "DEAD_BALL_",
              "LIVE_BALL_", "PATH_SMOOTHING", "BALL_DRIFT_", "FLOW_")
TABLE = ("X", "y_state", "y_boundary", "y_short", "y_touch", "y_risk", "is_stride")


def log(message):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), message))
    sys.stdout.flush()


def settings(quick=False):
    """The run's training settings, from config, with --quick overrides applied."""
    params = dict(config.HGB)
    gate_params = dict(config.GATE_HGB)
    carrier_params = dict(config.CARRIER_HGB)
    stride, copies = config.TRAIN_STRIDE, config.NOISE_COPIES
    carrier_stride = config.CARRIER_STRIDE
    if quick:
        params["max_iter"] = gate_params["max_iter"] = config.QUICK["max_iter"]
        carrier_params["max_iter"] = config.QUICK["max_iter"]
        stride, copies = config.QUICK["TRAIN_STRIDE"], config.QUICK["NOISE_COPIES"]
        carrier_stride = config.QUICK["CARRIER_STRIDE"]
    return {"hgb": params, "gate_hgb": gate_params, "carrier_hgb": carrier_params,
            "stride": stride, "copies": copies, "carrier_stride": carrier_stride}


def rng_for(name, copy, seed=config.SEED):
    """A generator that depends only on (seed, game, copy) -- never on run order."""
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8")), copy])


# -- labels -> training targets ----------------------------------------------------------------

def label_changes(labels, period):
    """Boolean per frame: the label differs from the previous frame, inside one period."""
    change = np.zeros(len(labels), dtype=bool)
    change[1:] = ((labels[1:] != labels[:-1]) & (labels[1:] != X) & (labels[:-1] != X)
                  & (period[1:] == period[:-1]) & (period[1:] > 0))
    return change


def near(mask, radius):
    """Boolean per frame: some True in mask lies within radius frames."""
    counts = np.concatenate(([0], np.cumsum(mask)))
    index = np.arange(len(mask))
    low = np.clip(index - radius, 0, len(mask))
    high = np.clip(index + radius + 1, 0, len(mask))
    return (counts[high] - counts[low]) > 0


def boundary_targets(labels, period, tolerance=None):
    """Per frame: 0 if no label change lies within tolerance frames (default
    config.BOUNDARY_TOLERANCE), else 1 + the label the nearest such change goes to."""
    tolerance = config.BOUNDARY_TOLERANCE if tolerance is None else tolerance
    out = np.zeros(len(labels), dtype=np.int8)
    frames = np.flatnonzero(label_changes(labels, period))
    if not len(frames):
        return out
    index = np.arange(len(labels))
    position = np.searchsorted(frames, index)
    before = frames[np.clip(position - 1, 0, len(frames) - 1)]
    after = frames[np.clip(position, 0, len(frames) - 1)]
    nearest = np.where(np.abs(index - before) <= np.abs(after - index), before, after)
    hit = np.abs(nearest - index) <= tolerance
    out[hit] = 1 + labels[nearest[hit]]
    return out


def short_targets(labels, period, longest=None):
    """Per frame: 1 + label inside an H or A run shorter than longest frames (default
    config.SHORT_POSSESSION), else 0. Runs end at label changes and at period edges."""
    longest = config.SHORT_POSSESSION if longest is None else longest
    out = np.zeros(len(labels), dtype=np.int8)
    edge = np.ones(len(labels) + 1, dtype=bool)
    edge[1:-1] = (labels[1:] != labels[:-1]) | (period[1:] != period[:-1])
    bounds = np.flatnonzero(edge)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if labels[start] in (H, A) and period[start] > 0 and stop - start < longest:
            out[start:stop] = 1 + labels[start]
    return out


def risk_targets(labels, period, half=None, changes=None):
    """Per frame: 1 if possession passes straight from one team to the other at least changes
    times (default config.RISK_CHANGES) within half frames either side (default config.RISK_HALF),
    inside one period, else 0. Changes through a dead ball are restarts, not counted."""
    half = config.RISK_HALF if half is None else half
    changes = config.RISK_CHANGES if changes is None else changes
    teams = np.isin(labels, (H, A))
    turnover = label_changes(labels, period)
    turnover[1:] &= teams[1:] & teams[:-1]
    out = np.zeros(len(labels), dtype=np.int8)
    for start, stop, _ in period_blocks(period):
        counts = np.concatenate(([0], np.cumsum(turnover[start:stop])))
        index = np.arange(stop - start)
        low = np.clip(index - half, 0, stop - start)
        high = np.clip(index + half + 1, 0, stop - start)
        out[start:stop] = (counts[high] - counts[low]) >= changes
    return out


def touch_frames(tracking):
    """(frames, teams) of the touches in CLEAN tracking: runs of frames where the ball's velocity
    jumps by more than TOUCH_TARGET_DV with a player within TOUCH_TARGET_RADIUS, each credited at
    its largest jump to the nearest player's team. A source that knows its touches
    (Tracking.touches) gives those instead."""
    if tracking.touches is not None:
        return tracking.touches
    frames, teams = [], []
    half = config.VELOCITY_HALF
    for start, stop, _ in period_blocks(tracking.period):
        ball = tracking.ball[start:stop].astype(np.float64)
        players = tracking.players[start:stop].astype(np.float64)
        with np.errstate(invalid="ignore"):
            jump = np.hypot(*((_shift(ball, -half) - ball) - (ball - _shift(ball, half))).T)
            jump = np.nan_to_num(jump * tracking.fps / half)
            distance = np.hypot(players[..., 0] - ball[:, None, 0],
                                players[..., 1] - ball[:, None, 1])
        distance = np.where(np.isnan(distance), np.inf, distance)
        nearest = np.argmin(distance, axis=1)
        hit = (jump > config.TOUCH_TARGET_DV) & (distance.min(axis=1) < config.TOUCH_TARGET_RADIUS)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], hit.astype(np.int8), [0]))))
        for first, stop_run in zip(edges[::2], edges[1::2]):
            at = first + int(np.argmax(jump[first:stop_run]))
            frames.append(start + at)
            teams.append(int(tracking.team[nearest[at]]))
    return np.array(frames, dtype=np.int64), np.array(teams, dtype=np.int8)


def touch_targets(tracking, tolerance=None):
    """Per frame: 1 + team of a touch within tolerance frames (default config.TOUCH_TOLERANCE;
    the nearest touch if several), else 0. tracking must be the clean one."""
    tolerance = config.TOUCH_TOLERANCE if tolerance is None else tolerance
    out = np.zeros(tracking.n_frames, dtype=np.int8)
    frames, teams = touch_frames(tracking)
    distance = np.full(tracking.n_frames, tolerance + 1)
    for frame, team in zip(frames, teams):
        low, high = max(0, frame - tolerance), min(tracking.n_frames, frame + tolerance + 1)
        gap = np.abs(np.arange(low, high) - frame)
        closer = gap < distance[low:high]
        out[low:high][closer] = 1 + team
        distance[low:high][closer] = gap[closer]
    out[tracking.period == 0] = 0
    return out


def training_rows(labels, period, stride):
    """(rows, is_stride, boundary target): every stride-th labelled frame, plus every frame near
    a label change so the transition expert sees enough boundaries."""
    labelled = np.flatnonzero((period > 0) & (labels != X))
    change = label_changes(labels, period)
    boundary = boundary_targets(labels, period)
    dense = near(change, config.BOUNDARY_TOLERANCE + config.BOUNDARY_CONTEXT)
    stride_rows = labelled[::stride]
    rows = np.union1d(stride_rows, labelled[dense[labelled]])
    return rows, np.isin(rows, stride_rows), boundary


def compute_features(tracking, carrier=None):
    """The FeatureSet of one match, with the carrier model's series added if there is one, and
    Smart Tagging's if config uses it (NaN where the match has none)."""
    feature_set = features.compute(tracking)
    if carrier is not None:
        features.extend(feature_set, tracking.period, *carrier.series(tracking))
    if config.USE_TAGS:
        team, shared = tagging.series(tracking.tags, tracking.period)
        if config.TAGS_AGREEMENT:
            shared.update(tagging.agreement(tracking.tags, feature_set.team[0]["near.0"],
                                            feature_set.shared["ball_vis.0"] == 1,
                                            tracking.period))
        features.extend(feature_set, tracking.period, team, shared,
                        (((), tagging.TEAM_SERIES, ()), ((), tagging.SHARED_SERIES, ())))
    return feature_set


def corrupt(tracking, labels, profile, copy):
    """A noisy copy of a labelled game, and -- when config uses tags -- its look-alike Smart
    Tagging, made from the clean game and corrupted on a generator of its own: the harder
    corruption for training copies, typical tags for the held-out copy (TEST_COPY)."""
    if tracking.name in matches.SKILLCORNER:  # a tracker's own output, already lossy
        noisy = noise.jittered(tracking, profile, rng_for(tracking.name, copy))
    else:
        noisy = noise.corrupt(tracking, profile, rng_for(tracking.name, copy), dead=labels == D)
    if config.USE_TAGS:
        noisy.tags = tagging.simulate(tracking, labels, rng_for(tracking.name, copy,
                                                                seed=config.SEED + 1),
                                      hard=config.TAGS_HARD and copy != TEST_COPY)
    return noisy


@contextlib.contextmanager
def using_tags(use):
    """config.USE_TAGS set to use for the duration: what TAGS_CHECK decided for one model."""
    saved = config.USE_TAGS
    config.USE_TAGS = bool(use)
    try:
        yield
    finally:
        config.USE_TAGS = saved


def tag_plan(games, profile, tracking):
    """Whether, and how, a model for tracking reads its Smart Tagging: tagging.check() of its
    tags against its own tracking, the floor set by look-alike tags of the training games on
    noisy copies like it. {"use": False} when it has no tags."""
    if tracking.tags is None:
        return {"use": False, "reason": "no Smart Tagging"}
    samples = []
    for clean, labels in games:
        holder = tagging.holders(noise.corrupt(clean, profile, rng_for(clean.name, TAG_COPY),
                                               dead=labels == D))
        samples += [(holder, tagging.simulate(clean, labels,
                                              rng_for(clean.name, draw, seed=config.SEED + 2),
                                              hard=False))
                    for draw in range(tagging.REFERENCE_DRAWS)]
    reference = tagging.reference(samples)
    plan = tagging.check(tracking.tags, tagging.holders(tracking), reference)
    plan["reference"] = reference
    return plan


def describe_tags(plan):
    """One line for the log."""
    if "groups" not in plan:
        return "{} ({})".format("read, unchecked" if plan["use"] else "not read",
                                plan.get("reason", "TAGS_CHECK off"))
    groups = "; ".join("{} {}{} {:.3f} (floor {:.3f}, chance {:.3f})".format(
        group, verdict, "" if plan["judged"][group] or verdict == "blind" else " unjudged",
        plan["agreement"][group],
        plan["reference"]["floor"][group], plan["chance"][group])
        for group, verdict in sorted(plan["groups"].items()))
    return "{}: {} frames late; {}".format("read" if plan["use"] else "not read", plan["late"],
                                           groups)


def build_table(noisy_copies, labels, run, carrier=None, clean=None):
    """Feature rows of one game's noisy copies, both team orders. clean, the uncorrupted
    tracking, supplies the touch targets; without it there are none."""
    touch = touch_targets(clean) if clean is not None else np.zeros(len(labels), dtype=np.int8)
    parts = []
    for noisy in noisy_copies:
        feature_set = compute_features(noisy, carrier)
        rows, is_stride, boundary = training_rows(labels, noisy.period, run["stride"])
        short = short_targets(labels, noisy.period)
        risk = risk_targets(labels, noisy.period)
        for order, swap_y, swap_b, swap_s in (
                ((0, 1), np.arange(3), np.arange(4), np.arange(3)),
                ((1, 0), SWAP_STATE, SWAP_BOUNDARY, SWAP_TEAMS)):
            parts.append({"X": feature_set.matrix(order, rows), "y_state": swap_y[labels[rows]],
                          "y_boundary": swap_b[boundary[rows]], "y_short": swap_s[short[rows]],
                          "y_touch": swap_s[touch[rows]], "y_risk": risk[rows],
                          "is_stride": is_stride})
        names = feature_set.names()
    table = {"names": names, **{key: np.concatenate([p[key] for p in parts]) for key in TABLE}}
    table["digest"] = _digest(table)
    return table


def _digest(table):
    """A fingerprint of a table's contents: names, rows and targets."""
    digest = hashlib.sha1(repr(table["names"]).encode("utf-8"))
    for key in TABLE:
        digest.update(np.ascontiguousarray(table[key]).tobytes())
    return digest.hexdigest()


def merge(tables):
    return {"names": tables[0]["names"],
            **{key: np.concatenate([t[key] for t in tables]) for key in TABLE}}


# -- running the chain ---------------------------------------------------------------------------

def expert_outputs(bank, tracking, carrier=None):
    """Expert outputs and gate context for every frame in play."""
    feature_set = compute_features(tracking, carrier)
    rows = np.flatnonzero(tracking.period > 0)
    return {"rows": rows, "E": bank.predict(feature_set, rows), "period": tracking.period,
            "ctx_ab": feature_set.matrix((0, 1), rows, context_names()),
            "ctx_ba": feature_set.matrix((1, 0), rows, context_names())}


def full_length(values, rows, n, fill):
    out = np.full((n,) + values.shape[1:], fill, dtype=np.float64)
    out[rows] = values
    return out


def stacked(outputs, probs, boundary):
    """The refiner's input: the experts' outputs with the first gate's beside them."""
    rows = outputs["rows"]
    return dict(outputs, E=np.hstack([outputs["E"], probs[rows], boundary[rows]]))


def chain_probs(gate, bank, outputs, refiner=None, first=None):
    """((n, 3) P(H, A, D), (n, 3) P(change to H / A / D near)) for the decoder.

    The gate's, refined by a second gate pass if there is one. first: the gate's own output, if
    already computed. The boundary comes from the gates, which see the experts over time, unless
    config says to use the transition expert's own output instead.
    """
    probs, boundary = first if first is not None else gate.predict(outputs)
    if refiner is not None:
        probs, boundary = refiner.predict(stacked(outputs, probs, boundary))
    if not config.GATE_BOUNDARY:
        names = bank.outputs()
        columns = [names.index("transition." + key) for key in ("BH", "BA", "BD")]
        boundary = full_length(outputs["E"][:, columns], outputs["rows"],
                               len(outputs["period"]), 0.1)
    return probs, boundary


def fit_bank(run, tables, which, names, banks=None):
    """Experts trained on the tables at the indices in which, from the cache if one is given."""
    def fit():
        return ExpertBank(run["hgb"]).fit(merge([tables[i] for i in which]))
    return fit() if banks is None else banks.get([tables[i]["digest"] for i in which], fit)


class BankCache(object):
    """Fitted expert banks on disk, keyed by the exact contents of the tables they were trained
    on, in order, and by everything else that shapes the fit: the experts' code, the run settings
    and every config value the later stages do not own. Two runs whose tables match byte for
    byte share banks, whatever else changed. Fitting is deterministic, so a cached bank is the
    bank a refit would produce; parallel folds that miss at the same moment both fit it and write
    the same file."""

    def __init__(self, run, profile, folder=BANK_DIR):
        self.folder, self.run, self.profile = folder, run, profile

    def fingerprint(self):
        """Everything but the tables that shapes a fit, as config stands now."""
        settings = sorted((key, repr(getattr(config, key))) for key in dir(config)
                          if key.isupper() and not key.startswith(AFTER_BANK))
        return hashlib.sha1(repr(
            (SOURCE_HASH, sklearn.__version__, settings, sorted(self.run.items()),
             sorted(self.profile.items()))).encode("utf-8")).hexdigest()

    def get(self, digests, fit):
        """The bank fitted on the tables with these digests, in this order."""
        key = hashlib.sha1(repr((self.fingerprint(), list(digests))).encode("utf-8")).hexdigest()
        path = os.path.join(self.folder, key[:24] + ".pkl")
        if os.path.exists(path):
            return load(path)
        bank = fit()
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder, exist_ok=True)
        partial = "{}.{}.part".format(path, os.getpid())
        save(bank, partial)
        os.replace(partial, path)  # atomic: no fold ever reads half a file
        return bank


def _source_hash():
    """The code that turns a table into a bank: the experts, and the table merge. Everything
    upstream of the table is in the table digests already."""
    digest = hashlib.sha1(inspect.getsource(experts).encode("utf-8"))
    digest.update(inspect.getsource(merge).encode("utf-8"))
    return digest.hexdigest()


SOURCE_HASH = _source_hash()  # read at import, so it describes the code this process runs


def fit_gate(bank, oof, stride, gate_params, refiner=False):
    """Gate on out-of-fold outputs, its rows sampled exactly as the experts' are. refiner: a second
    pass, whose items carry the first gate's outputs after the experts' (stacked())."""
    names, swap = bank.outputs(), bank.swap_index()
    if refiner:
        names = names + list(GATE_OUTPUTS)
        swap = np.concatenate([swap, len(swap) + SWAP_STATE, len(swap) + 3 + SWAP_STATE])
    gate = Gate(names, swap, gate_params, temporal=config.GATE_TEMPORAL)
    return gate.fit(oof, stride, training_rows)


# -- train ---------------------------------------------------------------------------------------

def train(games, profile, quick=False, tune_decoder=True, banks=None, first=0, tags=None):
    """games: [(Tracking, labels)] from OTHER matches. Returns a bundle dict for predict().

    banks: a BankCache, or None to fit every bank here. first: the inner fold to fit first --
    evaluate() staggers it so parallel held-out games start on different banks. tags:
    tag_plan() for the match the model is for, which says whether it reads Smart Tagging; None
    keeps config.USE_TAGS."""
    plan = tags if tags is not None else {"use": config.USE_TAGS, "reason": "TAGS_CHECK off"}
    with using_tags(plan["use"]):
        bundle = _train(games, profile, quick, tune_decoder, banks, first)
    bundle["tags"] = plan
    return bundle


def _train(games, profile, quick, tune_decoder, banks, first):
    if len(games) < 2:
        raise ValueError("cross-fitting needs at least two training games")
    run = settings(quick)
    started = time.time()

    copies = [[corrupt(tracking, labels, profile, copy) for copy in range(run["copies"])]
              for tracking, labels in games]

    # The carrier model, cross-fitted like everything else: game g's carrier features come from
    # a carrier that never saw game g, so the experts learn how far to trust it out of sample.
    carriers, final_carrier = [None] * len(games), None
    if config.USE_CARRIER:
        pairs = [[(noisy, tracking.ball) for noisy in copies[g]]
                 for g, (tracking, _) in enumerate(games)]
        for g, (tracking, _) in enumerate(games):
            log("carrier    without {}".format(tracking.name))
            carriers[g] = Carrier(run["carrier_hgb"]).fit(
                [p for i, game_pairs in enumerate(pairs) if i != g for p in game_pairs],
                run["carrier_stride"])
        log("carrier    on all {} games".format(len(games)))
        final_carrier = Carrier(run["carrier_hgb"]).fit(
            [p for game_pairs in pairs for p in game_pairs], run["carrier_stride"])

    tables = []
    for g, (tracking, labels) in enumerate(games):
        log("features   {} ({} noise copies)".format(tracking.name, run["copies"]))
        tables.append(build_table(copies[g], labels, run, carriers[g], clean=tracking))

    names = [tracking.name for tracking, _ in games]
    oof = [None] * len(games)
    for g in [(first + step) % len(games) for step in range(len(games))]:
        tracking, labels = games[g]
        log("fold       experts without {}".format(tracking.name))
        bank = fit_bank(run, tables, [i for i in range(len(games)) if i != g], names, banks)
        item = expert_outputs(bank, copies[g][0], carriers[g])
        item.update({"labels": labels, "name": tracking.name})
        oof[g] = item

    log_pi = decoder.transition_prior([l for _, l in games], [t.period for t, _ in games])
    params, decoder_score = dict(config.DECODER_DEFAULT), None
    reference = ExpertBank(run["hgb"])
    refine = config.GATE_PASSES > 1
    held = []  # each game's gate output, from a gate that never saw that game
    if tune_decoder or refine:
        for g, item in enumerate(oof):
            gate = fit_gate(reference, [o for i, o in enumerate(oof) if i != g], run["stride"],
                            run["gate_hgb"])
            held.append(gate.predict(item))
    if tune_decoder:
        cases = []
        for g, item in enumerate(oof):
            refiner = None
            if refine:  # game g's refiner never saw game g either
                log("refiner    without {}".format(item["name"]))
                refiner = fit_gate(reference, [stacked(o, *held[i]) for i, o in enumerate(oof)
                                               if i != g], run["stride"], run["gate_hgb"], True)
            probs, boundary = chain_probs(None, reference, item, refiner, first=held[g])
            cases.append((probs, boundary, item["period"], item["labels"]))
        log("decoder    searching {} on {} out-of-fold games".format(
            sorted(config.DECODER_GRID), len(cases)))
        params, decoder_score, _ = decoder.tune(cases, log_pi)
        log("decoder    best {} -> out-of-fold objective {:.4f}".format(params, decoder_score))

    log("final      experts and gate on all {} games".format(len(games)))
    bank = fit_bank(run, tables, range(len(games)), names, banks)
    gate = fit_gate(bank, oof, run["stride"], run["gate_hgb"])
    refiner = (fit_gate(bank, [stacked(o, *held[i]) for i, o in enumerate(oof)], run["stride"],
                        run["gate_hgb"], True) if refine else None)
    log("done       in {:.0f} min".format((time.time() - started) / 60.0))
    return {
        "bank": bank, "gate": gate, "refiner": refiner, "carrier": final_carrier,
        "decoder": params,
        "log_pi": log_pi,
        "train_games": [t.name for t, _ in games], "profile": profile, "run": run,
        "decoder_oof_objective": decoder_score,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
    }


def save(bundle, path):
    with open(path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=4)
    return path


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# -- the flow model --------------------------------------------------------------------------------

def train_flow(games, profile, plan, device=None):
    """The LSTM flow-matching model (flow.py) for a bundle, trained on FLOW_COPIES noisy copies of
    each labelled game and reading Smart Tagging as plan says, like the mixture it joins. device:
    "mps", "cpu", or None for the GPU where there is one."""
    import torch
    from possession import flow
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    sequences, names = [], None
    with using_tags(plan["use"]):
        for tracking, labels in games:
            for copy in range(config.FLOW_COPIES):
                feature_set = compute_features(corrupt(tracking, labels, profile, copy))
                names = feature_set.names()
                sequences.append({"x": feature_set.matrix((0, 1)).astype(np.float16),
                                  "y": labels.astype(np.int64),
                                  "valid": (tracking.period > 0) & (labels != X)})
            log("flow       copies of {}".format(tracking.name))
    scale = flow.scaler([s["x"] for s in sequences], np.random.default_rng(config.SEED))
    log("flow       training on {}, {} steps".format(device, flow.TRAIN_STEPS))
    net = flow.train(sequences, flow.swap_columns(names), scale, device, steps=flow.TRAIN_STEPS,
                     seed=config.SEED, log=log)
    return {"weights": flow.state(net), "scale": scale, "names": names,
            "weight": config.FLOW_WEIGHT, "temperature": config.FLOW_TEMPERATURE}


def sharpen(probs, temperature):
    """(T, 3) probabilities raised to 1 / temperature and renormalised: below 1, more decisive;
    1 leaves them as they are."""
    if temperature == 1.0:
        return probs
    p = np.clip(probs, 1e-9, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def flow_probs(model, tracking):
    """(T, 3) the flow model's P(H, A, D) for one match -- its first step -- from the features
    the mixture reads."""
    from possession import flow
    feature_set = compute_features(tracking)
    if feature_set.names() != model["names"]:
        raise ValueError("the flow model was trained on other features -- train it again")
    net = flow.restore(model["weights"], len(model["names"]))
    return flow.first_round(net, feature_set.matrix((0, 1)).astype(np.float16),
                            flow.swap_columns(model["names"]), model["scale"])


# -- predict ---------------------------------------------------------------------------------------

def predict(bundle, tracking):
    """(T,) label codes for one match. No fitting happens here. A model whose tag plan says to
    read Smart Tagging reads the match's tags as the plan says -- once check() has confirmed they
    are the tags the plan was made from. A bundle with a flow model blends its P(H, A, D) with the
    gate's, the flow's share its "weight"."""
    plan = bundle.get("tags", {"use": config.USE_TAGS})
    if plan["use"] and "groups" in plan:
        if tracking.tags is None:
            raise ValueError("this model reads Smart Tagging, and the match has none")
        again = tagging.check(tracking.tags, tagging.holders(tracking), plan["reference"])
        if (again["late"], again["groups"]) != (plan["late"], plan["groups"]):
            raise ValueError("the match's Smart Tagging is not the one this model was trained "
                             "for -- train again")
        tracking = tracking.replace(tags=tagging.apply(tracking.tags, plan, tracking.n_frames))
    with using_tags(plan["use"]):
        outputs = expert_outputs(bundle["bank"], tracking, bundle.get("carrier"))
        probs, boundary = chain_probs(bundle["gate"], bundle["bank"], outputs,
                                      bundle.get("refiner"))
        if bundle.get("flow"):
            weight = bundle["flow"]["weight"]
            temperature = bundle["flow"].get("temperature", config.FLOW_TEMPERATURE)
            probs = (1 - weight) * probs + weight * sharpen(flow_probs(bundle["flow"], tracking),
                                                            temperature)
    return decoder.decode(probs, boundary, tracking.period, bundle["decoder"], bundle["log_pi"])


def baseline(tracking, window=50):
    """The published baseline on any Tracking: nearest player's team, majority over +/- window
    frames, D where the ball is not seen. The reference every held-out score is compared to."""
    from starter.baseline import smooth
    distance = np.hypot(tracking.players[..., 0] - tracking.ball[:, None, 0],
                        tracking.players[..., 1] - tracking.ball[:, None, 1])
    seen = ~np.isnan(tracking.ball[:, 0]) & ~np.all(np.isnan(distance), axis=1)
    nearest = np.argmin(np.where(np.isnan(distance), np.inf, distance), axis=1)
    codes = np.where(seen, tracking.team[nearest], -1).astype(np.int8)
    letters = smooth(codes, window)
    out = np.select([letters == "H", letters == "A"], [H, A], D).astype(np.int8)
    out[tracking.period == 0] = X
    return out


# -- evaluate --------------------------------------------------------------------------------------

def evaluate(games, profile, quick=False, bootstrap_n=0, holdout=None, cache=True):
    """Outer leave-one-game-out: train on the rest, score the held-out game, noise copy TEST_COPY.

    Returns [{game, model: scores, baseline: scores}]. holdout limits which games are held out.
    cache shares fitted expert banks between held-out games (BankCache); scores do not change.
    With config.TAGS_CHECK, the held-out copy carries look-alike Smart Tagging and tag_plan()
    decides, as for the provided match, whether its model reads it.
    """
    results = []
    banks = BankCache(settings(quick), profile) if cache else None
    for h, (tracking, labels) in enumerate(games):
        if holdout and tracking.name not in holdout:
            continue
        log("== held out {}".format(tracking.name))
        rest = [g for i, g in enumerate(games) if i != h]
        with using_tags(config.USE_TAGS or config.TAGS_CHECK):
            noisy = corrupt(tracking, labels, profile, TEST_COPY)
        plan = None
        if config.TAGS_CHECK:
            plan = tag_plan(rest, profile, noisy)
            log("   tags {}".format(describe_tags(plan)))
        # Held-out game h starts on the game after it, so parallel folds start on distinct banks.
        bundle = train(rest, profile, quick, banks=banks, first=h % (len(games) - 1), tags=plan)
        predicted = predict(bundle, noisy)
        reference = baseline(noisy)
        results.append({"game": tracking.name, "labels": labels, "predicted": predicted,
                        "model": score(labels, predicted, bootstrap_n),
                        "baseline": score(labels, reference, bootstrap_n),
                        "decoder": bundle["decoder"],
                        "tags": {key: value for key, value in bundle["tags"].items()
                                 if key != "reference"}})
        log("   model spellF1 {:.4f}  baseline spellF1 {:.4f}".format(
            results[-1]["model"]["spell_f1_mean"], results[-1]["baseline"]["spell_f1_mean"]))
    return results


def evaluate_fold(task):
    """One held-out game, for a worker process of `evaluate --jobs`: it loads its own games and
    fits on one thread. Lives here, not in __main__, so worker processes can import it."""
    ids, held_out, profile, quick, bootstrap_n = task
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        games = [matches.load(match_id) for match_id in ids]
        return evaluate(games, profile, quick, bootstrap_n, holdout=[held_out])[0]
