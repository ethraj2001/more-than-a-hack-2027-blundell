"""models/moe_feat.pkl, the submitted model: the mixture trained as `train --quick` does with
passing lanes, keepers and turns on (config.LANE_FEATURES, KEEPER_FEATURES, TURN_FEATURES), the
dilated second stage (config.DILATED_STAGE, possession/dilated.py), and the flow model of
models/moe.pkl as it is. Run after `python -m possession train`:

    python build_submission.py [OUT.pkl]

The three settings are stored in the model, so `predict` turns them on again when it loads it.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"   # tree fitting is faster on one thread here (possession/README.md)
from threadpoolctl import threadpool_limits

from possession import config, matches, pipeline
from possession.__main__ import describe_flow, load_games, profile_for, tag_plan_for

# Params
SETTINGS = {"LANE_FEATURES": True, "KEEPER_FEATURES": True, "TURN_FEATURES": True,
            "DILATED_STAGE": True, "POSSESSION_RULES": True}


def main(out):
    for key, value in SETTINGS.items():
        setattr(config, key, value)
    production = pipeline.load(config.DEFAULT_MODEL)
    games, profile = load_games(list(matches.default())), profile_for(config.DEFAULT_PROFILE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games, profile, quick=True, tags=tag_plan_for(games, profile))
    bundle["broadcast"] = bool(profile.get("broadcast"))
    bundle["config"] = SETTINGS
    bundle["flow"] = production["flow"]
    pipeline.save(bundle, out)
    print("decoder {}".format(bundle["decoder"]))
    print("tags    {}".format(pipeline.describe_tags(bundle["tags"])))
    print("flow    {}".format(describe_flow(bundle)))
    print("-> {}".format(os.path.relpath(out, config.ROOT)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.MODEL_DIR, "moe_feat.pkl"))
