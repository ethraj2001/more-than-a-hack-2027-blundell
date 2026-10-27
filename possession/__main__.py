"""Command line for the possession pipeline. Run from the repository root.

    python -m possession download                       # open DFL matches -> cache/dfl/
    python -m possession profile                        # ATD noise statistics -> models/
    python -m possession train                          # -> models/moe.pkl
    python -m possession predict --check                # -> evaluation/results/moe.csv, twice
    python -m possession evaluate --quick --jobs 4      # leave-one-game-out on DFL -> reports/
    python -m possession compare reports/a reports/b    # paired difference between two versions

`profile` and `predict` read the provided match, so run them on your own machine: the terms forbid
handing the match data, or extracts of it, to any hosted model.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

from possession import config, dfl, guard, noise, pipeline
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
        games.append(dfl.load(match_id))
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
        dfl.download(match_id)


def cmd_profile(args):
    from possession.tracking import load_atd
    profile = noise.measure(load_atd())
    noise.save_profile(profile, args.out)
    print(json.dumps(profile, indent=2, sort_keys=True))
    print("-> {}".format(os.path.relpath(args.out, config.ROOT)))


def cmd_train(args):
    bundle = pipeline.train(load_games(args.games), profile_for(args.profile), quick=args.quick,
                            tune_decoder=not args.no_tune)
    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    pipeline.save(bundle, args.out)
    print("decoder {}".format(bundle["decoder"]))
    print("-> {}".format(os.path.relpath(args.out, config.ROOT)))


def cmd_predict(args):
    from games import GAME, GAMES
    from possession.tracking import load_atd
    n_frames = GAMES[GAME]["n_video_frames"]

    codes = pipeline.predict(pipeline.load(args.model), load_atd())
    write_labels(args.out, codes)
    problem = validation_error(args.out, n_frames)
    if problem:
        print("FAIL  {}".format(problem))
        sys.exit(1)

    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    counts = {letter: int(np.sum(codes == code)) for code, letter in enumerate(pm.LABELS)}
    print("valid  {} rows  H {H}  A {A}  D {D}  X {X}  possessions {n}".format(
        n_frames, n=len(spells), **counts))
    print("sha256 {}".format(guard.file_digest(args.out)))
    print("-> {}".format(os.path.relpath(args.out, config.ROOT)))

    if args.check:
        # A second run from scratch -- model and data reloaded -- must give the same bytes
        # (RULES.md, rule 3).
        again = pipeline.predict(pipeline.load(args.model), load_atd())
        with tempfile.TemporaryDirectory() as folder:
            second = write_labels(os.path.join(folder, "again.csv"), again)
            same = guard.file_digest(second) == guard.file_digest(args.out)
        print("reproducible: {}".format("yes, identical bytes" if same else "NO -- runs differ"))
        if not same:
            sys.exit(1)


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
                    "decoder": r["decoder"]} for r in results], handle, indent=2, default=float)
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
    all_games = sorted(dfl.MATCHES)

    command = sub.add_parser("download", help="fetch the open DFL matches")
    command.add_argument("--games", nargs="+", default=all_games, choices=all_games)
    command.set_defaults(func=cmd_download)

    command = sub.add_parser("profile", help="measure the provided ATD's missing-data statistics")
    command.add_argument("--out", default=config.DEFAULT_PROFILE)
    command.set_defaults(func=cmd_profile)

    command = sub.add_parser("train", help="train experts, gate and decoder on DFL games")
    command.add_argument("--games", nargs="+", default=all_games, choices=all_games)
    command.add_argument("--profile", default=config.DEFAULT_PROFILE)
    command.add_argument("--out", default=config.DEFAULT_MODEL)
    command.add_argument("--quick", action="store_true", help="fewer rows, copies and trees")
    command.add_argument("--no-tune", action="store_true", help="keep the default decoder")
    command.set_defaults(func=cmd_train)

    command = sub.add_parser("predict", help="write the submission for the provided match")
    command.add_argument("--model", default=config.DEFAULT_MODEL)
    command.add_argument("--out", default=DEFAULT_SUBMISSION)
    command.add_argument("--check", action="store_true",
                         help="run twice from scratch and require identical bytes")
    command.set_defaults(func=cmd_predict)

    command = sub.add_parser("evaluate", help="leave-one-game-out score on DFL games")
    command.add_argument("--games", nargs="+", default=all_games, choices=all_games)
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
