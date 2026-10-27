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
number to compare versions on; the public window is not (EVALUATION.md, RULES.md).
"""

import pickle
import platform
import sys
import time
import zlib

import numpy as np
import sklearn

from possession import config, decoder, dfl, features, noise
from possession.carrier import Carrier
from possession.experts import SWAP_BOUNDARY, SWAP_STATE, ExpertBank
from possession.gate import CONTEXT, Gate
from possession.scoring import A, D, H, X, score

# Params
TEST_COPY = 99  # noise seed for a held-out game, distinct from every training copy


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
    """The FeatureSet of one match, with the carrier model's series added if there is one."""
    feature_set = features.compute(tracking)
    if carrier is not None:
        features.extend(feature_set, tracking.period, *carrier.series(tracking))
    return feature_set


def build_table(noisy_copies, labels, run, carrier=None):
    """Feature rows of one game's noisy copies, both team orders."""
    parts = []
    for noisy in noisy_copies:
        feature_set = compute_features(noisy, carrier)
        rows, is_stride, boundary = training_rows(labels, noisy.period, run["stride"])
        for order, swap_y, swap_b in (((0, 1), np.arange(3), np.arange(4)),
                                      ((1, 0), SWAP_STATE, SWAP_BOUNDARY)):
            parts.append({"X": feature_set.matrix(order, rows), "y_state": swap_y[labels[rows]],
                          "y_boundary": swap_b[boundary[rows]], "is_stride": is_stride})
        names = feature_set.names()
    return {"names": names, **{key: np.concatenate([p[key] for p in parts])
                               for key in ("X", "y_state", "y_boundary", "is_stride")}}


def merge(tables):
    return {"names": tables[0]["names"],
            **{key: np.concatenate([t[key] for t in tables])
               for key in ("X", "y_state", "y_boundary", "is_stride")}}


# -- running the chain ---------------------------------------------------------------------------

def expert_outputs(bank, tracking, carrier=None):
    """Expert outputs and gate context for every frame in play."""
    feature_set = compute_features(tracking, carrier)
    rows = np.flatnonzero(tracking.period > 0)
    return {"rows": rows, "E": bank.predict(feature_set, rows), "period": tracking.period,
            "ctx_ab": feature_set.matrix((0, 1), rows, list(CONTEXT)),
            "ctx_ba": feature_set.matrix((1, 0), rows, list(CONTEXT))}


def full_length(values, rows, n, fill):
    out = np.full((n,) + values.shape[1:], fill, dtype=np.float64)
    out[rows] = values
    return out


def chain_probs(gate, bank, outputs):
    """((n, 3) P(H, A, D), (n, 3) P(change to H / A / D near)) for the decoder.

    The boundary comes from the gate, which sees the experts over time, unless config says to
    use the transition expert's own output instead.
    """
    probs, boundary = gate.predict(outputs)
    if not config.GATE_BOUNDARY:
        names = bank.outputs()
        columns = [names.index("transition." + key) for key in ("BH", "BA", "BD")]
        boundary = full_length(outputs["E"][:, columns], outputs["rows"],
                               len(outputs["period"]), 0.1)
    return probs, boundary


def fit_gate(bank, oof, stride, gate_params):
    """Gate on out-of-fold outputs, its rows sampled exactly as the experts' are."""
    gate = Gate(bank.outputs(), bank.swap_index(), gate_params, temporal=config.GATE_TEMPORAL)
    return gate.fit(oof, stride, training_rows)


# -- train ---------------------------------------------------------------------------------------

def train(games, profile, quick=False, tune_decoder=True):
    """games: [(Tracking, labels)] from OTHER matches. Returns a bundle dict for predict()."""
    if len(games) < 2:
        raise ValueError("cross-fitting needs at least two training games")
    run = settings(quick)
    started = time.time()

    copies = [[noise.corrupt(tracking, profile, rng_for(tracking.name, copy))
               for copy in range(run["copies"])] for tracking, _ in games]

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
        tables.append(build_table(copies[g], labels, run, carriers[g]))

    oof = []
    for g, (tracking, labels) in enumerate(games):
        log("fold       experts without {}".format(tracking.name))
        bank = ExpertBank(run["hgb"]).fit(merge([t for i, t in enumerate(tables) if i != g]))
        item = expert_outputs(bank, copies[g][0], carriers[g])
        item.update({"labels": labels, "name": tracking.name})
        oof.append(item)

    log_pi = decoder.transition_prior([l for _, l in games], [t.period for t, _ in games])
    params, decoder_score = dict(config.DECODER_DEFAULT), None
    reference = ExpertBank(run["hgb"])
    if tune_decoder:
        cases = []
        for g, item in enumerate(oof):
            gate = fit_gate(reference, [o for i, o in enumerate(oof) if i != g], run["stride"],
                            run["gate_hgb"])
            probs, boundary = chain_probs(gate, reference, item)
            cases.append((probs, boundary, item["period"], item["labels"]))
        log("decoder    searching {} on {} out-of-fold games".format(
            sorted(config.DECODER_GRID), len(cases)))
        params, decoder_score, _ = decoder.tune(cases, log_pi)
        log("decoder    best {} -> out-of-fold spellF1 {:.4f}".format(params, decoder_score))

    log("final      experts and gate on all {} games".format(len(games)))
    bank = ExpertBank(run["hgb"]).fit(merge(tables))
    gate = fit_gate(bank, oof, run["stride"], run["gate_hgb"])
    log("done       in {:.0f} min".format((time.time() - started) / 60.0))
    return {
        "bank": bank, "gate": gate, "carrier": final_carrier, "decoder": params,
        "log_pi": log_pi,
        "train_games": [t.name for t, _ in games], "profile": profile, "run": run,
        "decoder_oof_spell_f1": decoder_score,
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


# -- predict ---------------------------------------------------------------------------------------

def predict(bundle, tracking):
    """(T,) label codes for one match. No fitting happens here."""
    outputs = expert_outputs(bundle["bank"], tracking, bundle.get("carrier"))
    probs, boundary = chain_probs(bundle["gate"], bundle["bank"], outputs)
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

def evaluate(games, profile, quick=False, bootstrap_n=0, holdout=None):
    """Outer leave-one-game-out: train on the rest, score the held-out game, noise copy TEST_COPY.

    Returns [{game, model: scores, baseline: scores}]. holdout limits which games are held out.
    """
    results = []
    for h, (tracking, labels) in enumerate(games):
        if holdout and tracking.name not in holdout:
            continue
        log("== held out {}".format(tracking.name))
        bundle = train([g for i, g in enumerate(games) if i != h], profile, quick)
        noisy = noise.corrupt(tracking, profile, rng_for(tracking.name, TEST_COPY))
        predicted = predict(bundle, noisy)
        reference = baseline(noisy)
        results.append({"game": tracking.name, "labels": labels, "predicted": predicted,
                        "model": score(labels, predicted, bootstrap_n),
                        "baseline": score(labels, reference, bootstrap_n),
                        "decoder": bundle["decoder"]})
        log("   model spellF1 {:.4f}  baseline spellF1 {:.4f}".format(
            results[-1]["model"]["spell_f1_mean"], results[-1]["baseline"]["spell_f1_mean"]))
    return results


def evaluate_fold(task):
    """One held-out game, for a worker process of `evaluate --jobs`: it loads its own games and
    fits on one thread. Lives here, not in __main__, so worker processes can import it."""
    ids, held_out, profile, quick, bootstrap_n = task
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        games = [dfl.load(match_id) for match_id in ids]
        return evaluate(games, profile, quick, bootstrap_n, holdout=[held_out])[0]
