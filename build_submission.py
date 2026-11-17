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
# The decoder's weight on the gate's change evidence, set on Metrica's three held-out games (real
# tracker output) instead of the DFL search's 1.0: each Metrica game corrected by the other two,
# +0.009 (90% interval +0.000 to +0.018), better on 2 of 3; the seven DFL games -0.007 (-0.012 to
# -0.002).
DECODER_CORRECTION = {"beta": 0.5}
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
    bundle["decoder"] = dict(bundle["decoder"], **DECODER_CORRECTION)
    bundle["flow"] = production["flow"]
    pipeline.save(bundle, out)
    print("decoder {}".format(bundle["decoder"]))
    print("tags    {}".format(pipeline.describe_tags(bundle["tags"])))
    print("flow    {}".format(describe_flow(bundle)))
    print("-> {}".format(os.path.relpath(out, config.ROOT)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(config.MODEL_DIR, "moe_feat_tcn_b05.pkl"))
