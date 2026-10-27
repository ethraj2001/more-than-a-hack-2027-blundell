"""Every tunable number in the pipeline, in one place.

None of these was chosen by looking at a score on the provided game -- the public window is not a
tuning set (RULES.md). They are either physical facts, conventions of the data, or defaults that
`python -m possession train` overwrites by searching on held-out DFL games. Change them here, and
nowhere else, so the technical document can quote one file.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# games.py and starter/ live at the repository root; make them importable however this package
# was launched.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Params
SEED = 0
FPS = 25

# Where generated files go. Both are gitignored: the cache holds downloaded third-party data and
# features, and features of the provided game are frame-level derived data that must never be
# published (RULES.md, "Do not publish derived data that would reconstruct it").
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
MODEL_DIR = os.path.join(ROOT, "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "moe.pkl")
DEFAULT_PROFILE = os.path.join(MODEL_DIR, "noise_profile.json")
REPORT_DIR = os.path.join(ROOT, "reports")

# -- tracking ----------------------------------------------------------------------------------

# Ball gaps up to one second are bridged linearly before computing ball features; anything
# longer is left missing, because a straight line over a longer gap invents passes.
BALL_INTERP_MAX_GAP = 25
# Frame counters (time since the ball was last seen, and so on) saturate here: beyond ten
# seconds the exact number carries no information.
CAP_FRAMES = 250
# Player speeds above this are tracking artefacts -- identity switches teleport a slot -- not
# running. 12 m/s is above any sprint recorded in professional football.
MAX_PLAYER_SPEED = 12.0
# Half-width, in frames, of the finite differences behind every velocity.
VELOCITY_HALF = 2

# -- features ----------------------------------------------------------------------------------

# Centred context windows, as half-widths in frames: 0.2 s, 1 s, 3 s. The pipeline is offline, so
# looking ahead is legitimate -- the whole match is available before a single label is written.
WINDOWS = (5, 25, 75)
# Past-versus-future comparison window for the transition expert, in frames.
TRANSITION_WINDOW = 25
# Shorter past-versus-future windows for the few series that time a touch: 0.2 s and 0.5 s. A
# one-second window can place a change no better than about half a second either way.
SHORT_WINDOWS = (5, 12)
STILL_SPEED = 1.0        # m/s; a player slower than this is standing
NEAR_RADII = (2.0, 5.0)  # metres; "touching distance" and "pressing distance"

# -- training ----------------------------------------------------------------------------------

# Consecutive frames are nearly identical, so training on every frame buys time, not signal. One
# frame in ten is 0.4 s apart -- still far denser than possessions change.
TRAIN_STRIDE = 10
# A frame is a boundary example if a label change lies within this many frames of it. Three
# frames is where spellF1 starts to charge for lateness (EVALUATION.md: 3 frames late = 0.98).
# The target also says which label the change goes TO, so the decoder knows where to move.
BOUNDARY_TOLERANCE = 3
# Extra rows sampled densely around every boundary, for the transition expert.
BOUNDARY_CONTEXT = 12
# Independent corruptions of each training game. More copies teach more noise patterns.
NOISE_COPIES = 2

HGB = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    # No early stopping: its validation split is random, and a random split is one more thing
    # that could make two runs differ.
    "early_stopping": False,
    "random_state": SEED,
}
GATE_HGB = dict(HGB, max_iter=150, max_leaf_nodes=15)
# The carrier model (carrier.py): which player is on the ball, from the players around it. Rows
# are (frame, player) pairs, so it samples frames more sparsely than the experts do.
USE_CARRIER = False
CARRIER_STRIDE = 10
CARRIER_HGB = dict(HGB, max_iter=150)
# The gate reads the experts' outputs over +-10 s (gate.py), not just at the frame itself, and
# supplies the decoder's change probabilities as well as its per-frame ones.
GATE_TEMPORAL = True
GATE_BOUNDARY = True

# --quick trades accuracy for time, for smoke runs and the nested evaluation.
QUICK = {"TRAIN_STRIDE": 20, "NOISE_COPIES": 1, "max_iter": 60, "CARRIER_STRIDE": 25}

# Threads for model fitting. 0 lets scikit-learn decide. Measured on one machine, a single thread
# fitted 4-5x faster than twelve -- OpenMP oversubscription -- so time both on yours
# (`--threads 1` against the default) before a long run. Results do not depend on it.
THREADS = int(os.environ.get("MTAH_THREADS", "0"))

# -- decoder -----------------------------------------------------------------------------------

# The defaults are a starting point only; train searches DECODER_GRID on out-of-fold DFL output.
DECODER_DEFAULT = {
    "alpha": 0.15,        # weight on per-frame evidence; < 1 because frames are not independent
    "beta": 1.0,          # weight on the transition expert's boundary probability
    "min_possession": 10,  # frames; no possession is shorter than this
    "min_dead": 10,        # frames; no stoppage is shorter than this
    "switch_penalty": 2.0,  # extra cost of any label change, in log units
}
DECODER_GRID = {
    "alpha": (0.05, 0.15, 0.4),
    "beta": (0.5, 1.0),
    "min_possession": (5, 15),
    "min_dead": (10,),
    "switch_penalty": (0.0, 2.0, 4.0),
}

# -- noise -------------------------------------------------------------------------------------

# Used only when no profile has been measured from the provided ATD yet. The first two are
# published in DATA.md; the rest are assumptions, marked as such, which `python -m possession
# profile` replaces with measurements.
DEFAULT_NOISE = {
    "ball_missing_frac": 0.41,     # DATA.md: ball missing on 41% of graded frames
    "players_visible_mean": 20.5,  # DATA.md: mean tracked objects over graded frames
    "ball_gap_mean": 30.0,         # assumption
    "player_gap_mean": 25.0,       # assumption
    "ball_outlier_rate": 0.002,    # assumption
    "player_jitter_m": 0.10,       # assumption
    "ball_jitter_m": 0.30,         # assumption
    "measured": False,
}
