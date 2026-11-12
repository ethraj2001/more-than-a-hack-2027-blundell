"""Command line for the possession pipeline. Run from the repository root.

    python -m possession download                       # open DFL + Metrica games -> cache/
    python -m possession profile                        # ATD noise statistics -> models/
    python -m possession train                          # -> models/moe.pkl, flow model included
    python -m possession train-flow                     # add the flow model to models/moe.pkl
    python -m possession predict --check                # -> evaluation/results/moe.csv, twice
    python -m possession evaluate --quick --jobs 4      # leave-one-game-out on DFL -> reports/
    python -m possession compare reports/a reports/b    # paired difference between two versions

`profile`, `train` and `predict` read the provided match -- `train` only to check its Smart Tagging
against its tracking (config.TAGS_CHECK) -- so run them on your own machine: the terms forbid
handing the match data, or extracts of it, to any hosted model.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

from possession import config, guard, matches, noise, pipeline
from possession.scoring import paired_spell_bootstrap, pm, score, validation_error, write_labels

# Params
DEFAULT_SUBMISSION = os.path.join(config.ROOT, "evaluation", "results", "moe.csv")


def threads(n):
    """Cap scikit-learn's OpenMP threads for the duration of a command, if asked."""
    if not n:
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=n)


def load_games(ids):
    games = []
    for match_id in ids:
        pipeline.log("loading    {}".format(match_id))
        games.append(matches.load(match_id))
    return games


def profile_for(path):
    profile = noise.load_profile(path)
    if not profile.get("measured"):
        pipeline.log("note       no measured noise profile at {} -- using the documented defaults. "
                     "Run `python -m possession profile` first.".format(path))
    return profile


def cmd_download(args):
    for match_id in args.games:
        pipeline.log("download   {}".format(match_id))
        matches.download(match_id)


def cmd_profile(args):
    from possession.tracking import load_atd
    profile = noise.measure(load_atd(broadcast=args.broadcast))
    if args.broadcast:  # train and predict read the ATD as this profile measured it
        profile["broadcast"] = True
    noise.save_profile(profile, args.out)
    print(json.dumps(profile, indent=2, sort_keys=True))
    print("-> {}".format(os.path.relpath(args.out, config.ROOT)))


def tag_plan_for(games, profile):
    """With config.TAGS_CHECK, the provided match's Smart Tagging checked against its tracking
    (pipeline.tag_plan) -- it decides whether the model reads the tags. None without the match."""
    if not config.TAGS_CHECK:
        return None
    from possession.tracking import load_atd
    try:
        match = load_atd(broadcast=bool(profile.get("broadcast")))
    except IOError as problem:
        pipeline.log("note       the provided match is not here, so its Smart Tagging cannot be "
                     "checked: training the model without tags ({})".format(problem))
        return {"use": False, "reason": "provided match not found"}
    plan = pipeline.tag_plan(games, profile, match)
    pipeline.log("tags       {}".format(pipeline.describe_tags(plan)))
    return plan


def cmd_train(args):
    games, profile = load_games(args.games), profile_for(args.profile)
    bundle = pipeline.train(games, profile, quick=args.quick, tune_decoder=not args.no_tune,
                            tags=tag_plan_for(games, profile))
    bundle["broadcast"] = bool(profile.get("broadcast"))
    if config.FLOW_WEIGHT > 0:
        bundle["flow"] = pipeline.train_flow(games, profile, bundle["tags"], args.device)
    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    pipeline.save(bundle, args.out)
    print("decoder {}".format(bundle["decoder"]))
    print("tags    {}".format(pipeline.describe_tags(bundle["tags"])))
    print("flow    {}".format(describe_flow(bundle)))
    print("-> {}".format(os.path.relpath(args.out, config.ROOT)))


def describe_flow(bundle):
    flow = bundle.get("flow")
    return "blended at {}".format(flow["weight"]) if flow else "none"


def cmd_train_flow(args):
    """The flow model alone, added to a trained bundle: the mixture is kept as it is."""
    bundle = pipeline.load(args.model)
    games, profile = load_games(args.games), profile_for(args.profile)
    plan = bundle.get("tags", {"use": config.USE_TAGS})
    pipeline.log("tags       {}".format(pipeline.describe_tags(plan)))
    bundle["flow"] = pipeline.train_flow(games, profile, plan, args.device)
    pipeline.save(bundle, args.model)
    print("flow    {}".format(describe_flow(bundle)))
    print("-> {}".format(os.path.relpath(args.model, config.ROOT)))


def cmd_predict(args):
    from games import GAME, GAMES
    from possession.tracking import load_atd
    n_frames = GAMES[GAME]["n_video_frames"]

    bundle = pipeline.load(args.model)
    uses = bundle.get("broadcast", False)  # as the profile the model was trained on was measured
    codes = pipeline.predict(bundle, load_atd(broadcast=uses))
    write_labels(args.out, codes)
    problem = validation_error(args.out, n_frames)
    if problem:
        print("FAIL  {}".format(problem))
        sys.exit(1)

    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    counts = {letter: int(np.sum(codes == code)) for code, letter in enumerate(pm.LABELS)}
    print("valid  {} rows  H {H}  A {A}  D {D}  X {X}  possessions {n}".format(
        n_frames, n=len(spells), **counts))
    plan = bundle.get("tags", {"use": config.USE_TAGS, "reason": "model made before the check"})
    print("tags   {}".format(pipeline.describe_tags(plan)))
    print("flow   {}".format(describe_flow(bundle)))
    print("ball   {}".format("gaps filled from the broadcast" if uses else "tactical only"))
    print("sha256 {}".format(guard.file_digest(args.out)))
    print("-> {}".format(os.path.relpath(args.out, config.ROOT)))

    if args.check:
        # A second run from scratch -- model and data reloaded -- must give the same bytes
        # (RULES.md, rule 3).
        again = pipeline.predict(pipeline.load(args.model), load_atd(broadcast=uses))
        with tempfile.TemporaryDirectory() as folder:
            second = write_labels(os.path.join(folder, "again.csv"), again)
            same = guard.file_digest(second) == guard.file_digest(args.out)
        print("reproducible: {}".format("yes, identical bytes" if same else "NO -- runs differ"))
        if not same:
            sys.exit(1)


def cmd_broadcast(args):
    """The broadcast ball aligned to the tactical timeline: aggregate checks only, nothing written."""
    from possession import broadcast
    from possession.tracking import load_atd
    tactical, other = load_atd(), broadcast.load()
    pipeline.log("broadcast  {} frames at {} fps; aligning".format(other.n_frames, other.fps))
    print(json.dumps(broadcast.describe(tactical, broadcast.align(tactical, other)), indent=2))


def cmd_evaluate(args):
    profile = profile_for(args.profile)
    if args.jobs > 1:
        # Folds are independent, so running them side by side changes the time, not the result.
        from concurrent.futures import ProcessPoolExecutor
        held_out = [g for g in args.games if not args.holdout or g in args.holdout]
        tasks = [(args.games, h, profile, args.quick, args.bootstrap) for h in held_out]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(pipeline.evaluate_fold, tasks))
    else:
        results = pipeline.evaluate(load_games(args.games), profile, quick=args.quick,
                                    bootstrap_n=args.bootstrap, holdout=args.holdout)
    if not os.path.isdir(args.report_dir):
        os.makedirs(args.report_dir)

    rows = []
    for item in results:
        write_labels(os.path.join(args.report_dir, "{}_gt.csv".format(item["game"])),
                     item["labels"])
        write_labels(os.path.join(args.report_dir, "{}_moe.csv".format(item["game"])),
                     item["predicted"])
        model, base = item["model"], item["baseline"]
        rows.append((item["game"], model["spell_f1_mean"], base["spell_f1_mean"],
                     model["macro_f1"], base["macro_f1"], model["n_pred_spells"],
                     model["n_gt_spells"]))

    print("\nheld-out DFL games (noise copy {}, {} settings)".format(
        pipeline.TEST_COPY, "quick" if args.quick else "full"))
    print("{:<8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}".format(
        "game", "spellF1", "baseline", "macroF1", "baseline", "n_pred", "n_true"))
    for row in rows:
        print("{:<8} {:>9.4f} {:>9.4f} {:>8.4f} {:>8.4f} {:>8d} {:>8d}".format(*row))
    deltas = np.array([r[1] - r[2] for r in rows])
    print("mean     {:>9.4f} {:>9.4f}   spellF1 gain over baseline {:+.4f} (min {:+.4f})".format(
        np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]), deltas.mean(),
        deltas.min()))

    summary = os.path.join(args.report_dir, "summary.json")
    with open(summary, "w") as handle:
        json.dump([{"game": r["game"], "model": r["model"], "baseline": r["baseline"],
                    "decoder": r["decoder"], "tags": r["tags"]} for r in results], handle,
                  indent=2, default=float)
    print("-> {}  (per-game CSVs score with: python evaluation/run_evaluation.py --gt <gt.csv>)"
          .format(os.path.relpath(summary, config.ROOT)))


def cmd_compare(args):
    """Paired comparison of two evaluate runs on the same games: B minus A."""
    def predictions(folder):
        with open(os.path.join(folder, "summary.json")) as handle:
            games = [item["game"] for item in json.load(handle)]
        return {g: pm.label_codes(pm.read_label_csv(os.path.join(folder, g + "_moe.csv")))
                for g in games}

    first, second = predictions(args.a), predictions(args.b)
    shared = sorted(set(first) & set(second))
    if not shared:
        print("FAIL  no game in common")
        sys.exit(1)
    rows = []
    for game in shared:
        gt = pm.label_codes(pm.read_label_csv(os.path.join(args.a, game + "_gt.csv")))
        rows.append((game, gt, first[game], second[game]))
    print("{:<8} {:>8} {:>8} {:>8}".format("game", "A", "B", "B - A"))
    deltas = []
    for game, gt, a, b in rows:
        score_a, score_b = score(gt, a)["spell_f1_mean"], score(gt, b)["spell_f1_mean"]
        deltas.append(score_b - score_a)
        print("{:<8} {:>8.4f} {:>8.4f} {:>+8.4f}".format(game, score_a, score_b, deltas[-1]))
    point, low, high = paired_spell_bootstrap([r[1:] for r in rows], n_draws=args.draws)
    better = sum(d > 0 for d in deltas)
    print("mean B - A {:+.4f}, 90% paired bootstrap [{:+.4f}, {:+.4f}], B better on {} of {} games"
          .format(point, low, high, better, len(deltas)))
    verdict = ("B is better" if low > 0 else "A is better" if high < 0 else
               "no difference shown")
    print("verdict: {}".format(verdict))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m possession",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=config.THREADS,
                        help="cap model-fitting threads (0 = library default)")
    sub = parser.add_subparsers(dest="command")
    sub.required = True
    all_games, default_games = list(matches.ALL), list(matches.default())

    command = sub.add_parser("download", help="fetch the open DFL matches and Metrica's samples")
    command.add_argument("--games", nargs="+", default=all_games, choices=all_games)
    command.set_defaults(func=cmd_download)

    command = sub.add_parser("profile", help="measure the provided ATD's missing-data statistics")
    command.add_argument("--out", default=config.DEFAULT_PROFILE)
    command.add_argument("--broadcast", action="store_true",
                         help="fill the ball's gaps from the broadcast first; train and predict follow")
    command.set_defaults(func=cmd_profile)

    command = sub.add_parser("broadcast", help="check the broadcast ball's alignment to tactical")
    command.set_defaults(func=cmd_broadcast)

    command = sub.add_parser("train", help="train experts, gate and decoder on labelled games")
    command.add_argument("--games", nargs="+", default=default_games, choices=all_games)
    command.add_argument("--profile", default=config.DEFAULT_PROFILE)
    command.add_argument("--out", default=config.DEFAULT_MODEL)
    command.add_argument("--quick", action="store_true", help="fewer rows, copies and trees")
    command.add_argument("--no-tune", action="store_true", help="keep the default decoder")
    command.add_argument("--device", choices=("mps", "cpu"),
                         help="where the flow model trains (default: the GPU where there is one)")
    command.set_defaults(func=cmd_train)

    command = sub.add_parser("train-flow", help="add the flow model to a trained bundle")
    command.add_argument("--games", nargs="+", default=default_games, choices=all_games)
    command.add_argument("--profile", default=config.DEFAULT_PROFILE)
    command.add_argument("--model", default=config.DEFAULT_MODEL)
    command.add_argument("--device", choices=("mps", "cpu"),
                         help="where it trains (default: the GPU where there is one)")
    command.set_defaults(func=cmd_train_flow)

    command = sub.add_parser("predict", help="write the submission for the provided match")
    command.add_argument("--model", default=config.DEFAULT_MODEL)
    command.add_argument("--out", default=DEFAULT_SUBMISSION)
    command.add_argument("--check", action="store_true",
                         help="run twice from scratch and require identical bytes")
    command.set_defaults(func=cmd_predict)

    command = sub.add_parser("evaluate", help="leave-one-game-out score on labelled games")
    command.add_argument("--games", nargs="+", default=default_games, choices=all_games)
    command.add_argument("--holdout", nargs="+", choices=all_games,
                         help="only hold out these games (each still trains on all the others)")
    command.add_argument("--profile", default=config.DEFAULT_PROFILE)
    command.add_argument("--quick", action="store_true")
    command.add_argument("--bootstrap", type=int, default=0, help="draws for spellF1 intervals")
    command.add_argument("--report-dir", default=os.path.join(config.REPORT_DIR, "dfl"))
    command.add_argument("--jobs", type=int, default=1,
                         help="held-out games run side by side, one thread each")
    command.set_defaults(func=cmd_evaluate)

    command = sub.add_parser("compare", help="paired spellF1 difference between two evaluate runs")
    command.add_argument("a", help="report folder of version A (the reference)")
    command.add_argument("b", help="report folder of version B")
    command.add_argument("--draws", type=int, default=1000)
    command.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    with threads(args.threads):
        args.func(args)


if __name__ == "__main__":
    main()
