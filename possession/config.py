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
# Touches: the ball's velocity over the TOUCH_HALF frames before and after a frame differs by
# more than TOUCH_DV, and one team's player is the nearest, within TOUCH_RADIUS. 0.2 s either
# side rather than VELOCITY_HALF, so that ball jitter does not read as a touch. Measured on four
# held-out games: +0.012 spellF1 (90% interval +0.002 to +0.023), better on 3 of 4. TOUCH_DV
# was 5 m/s; a smoothed ball (PATH_SMOOTHING) spreads a kick's change of velocity over several
# frames, so 5 rarely fires, and the clocks since and until a touch lose their hold on short
# turnovers (AUC 0.718 to 0.583 for the clock since one). At 1 m/s they read 0.696 again. Seven
# held-out DFL games, path smoothing on, the measured profile: +0.007 (+0.0001 to +0.014),
# better on 4 of 7, 0.737 to 0.744.
TOUCH_FEATURES = True
TOUCH_HALF = 5
TOUCH_RADIUS = 3.0       # metres
TOUCH_DV = 1.0           # m/s
# Fitted ball kinematics. The touch series above difference single positions, and the ball's
# jitter (0.3 m assumed per frame until `profile` measures it) enters a second difference over
# TOUCH_HALF frames as about 4 m/s of noise -- most of a touch's 5 m/s. FIT_FEATURES fits a
# straight line to the sightings over FIT_HALVES frames before and after each frame instead, and
# takes the change between the two fitted velocities: overall, as speed gained or lost, and per
# team next to its nearest player. On the noisy held-out copy of two DFL games, how well each
# singles out the clean-data touches (AUC, 0.5 = chance): ball_dv 0.55, the touch series 0.71,
# fits over 6 frames 0.78, over 12 about 0.86. On seven held-out games, with Smart Tagging
# checked: +0.009 spellF1 (90% interval +0.003 to +0.016), better on 6 of 7; macroF1 0.931 to
# 0.933. The possessions it adds are where the ball is seen. Without tags: +0.003 (-0.002 to
# +0.008), better on 5 of 7 -- no harm shown, no gain shown on its own.
FIT_FEATURES = True
FIT_HALVES = (6, 12)
# Loose balls. GAP_FEATURES: per team, how near its nearest player was to the ball at the last
# sighting before a gap and at the first one after it, and how far the ball moved in between --
# what a gap longer than BALL_INTERP_MAX_GAP otherwise hides. CONTEST_FEATURES: both teams within
# CONTEST_RADIUS of the ball at once, and for how long.
# Off: together they measured +0.001 on four held-out games (90% interval -0.007 to +0.009).
GAP_FEATURES = False
CONTEST_FEATURES = False
CONTEST_RADIUS = 2.5     # metres
# Player acceleration: a receiver brakes, a passer's team-mates set off. Velocities over
# DYNAMICS_HALF frames either side, so tracking jitter does not swamp it. Off: with the touch
# expert on, it lost 0.006 on four held-out games (90% interval -0.016 to +0.004).
DYNAMICS_FEATURES = False
DYNAMICS_HALF = 5
# Reach: how soon each team's quickest player can get to the ball -- now, and to where the ball
# is REACH_AHEAD frames later -- after a reaction time, running flat out. The reaction time and
# speed are the usual simple pitch-control values (Spearman 2018), not fitted. Off: on four
# held-out games it lost 0.005 against touch features alone (90% interval -0.013 to +0.004).
REACH_FEATURES = False
REACH_AHEAD = (10, 25)   # frames: 0.4 s and 1 s
REACH_REACTION = 0.7     # seconds
REACH_SPEED = 5.0        # m/s

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
# Possessions shorter than this many frames (2 s) are a third of all possessions and the ones
# the model misses most. The short-possession expert (experts.py) is trained to find them. Off:
# measured on four held-out games it added nothing over the touch features (-0.002, 90% interval
# -0.012 to +0.006) and cost 46% more fitting time.
SHORT_EXPERT = False
SHORT_POSSESSION = 50
# Fragmentation risk (experts.py "risk"): P(possession passes straight between the teams at least
# RISK_CHANGES times within RISK_HALF frames either side) -- the scrambles where possession
# fragments, learned from every feature. The gate reads it beside the other experts, over time
# like theirs, so it can weigh them differently inside a scramble. (Lowering the decoder's change
# cost where the gate itself expects many changes was tried first, on saved out-of-fold output
# of seven games: 0.6410 against 0.6414 held out -- the gate's change probabilities already
# carry it.)
RISK_EXPERT = False
RISK_HALF = 125          # 5 s: possession changed hands twice or more within 5 s either side
RISK_CHANGES = 2
# Smart Tagging (tagging.py): the provided game's automatic phase tags as features and as a
# "tags" expert. The DFL games get look-alike tags, corrupted by amounts drawn per copy, so a
# held-out DFL score with tags on says what tags of that quality are worth -- not what the real
# ones are, which only the final score can show. Always on, it is a bet: on four held-out DFL
# games it gained +0.014 spellF1 (90% interval +0.002 to +0.026) with tags like the simulated
# ones, but lost 0.011-0.022 when the held-out tags were missing, had random teams, edges 2 s
# late or the Defending codes reversed, and 0.077 with every team reversed. Isolating the tags in
# their expert (TAGS_ISOLATED) or training on harder corruption (TAGS_HARD) made it no safer
# and removed the gain: with three to six training games there are too few games to learn when
# a game's tags deserve trust.
# So TAGS_CHECK decides per run instead (tagging.check): before training, the provided game's
# tags are scored against its own tracking, and the model reads them only if they agree with it
# as well as every look-alike tag set of the training games does -- a code group whose teams
# read reversed flipped back, tags running late moved back. Otherwise the model is the one
# without tags. On four held-out games it kept the +0.014 with look-alike tags and lost nothing
# with missing, random, reversed or late ones; on seven, with look-alike tags, +0.024 (90%
# interval +0.016 to +0.033). USE_TAGS is set by the decision for the run; with TAGS_CHECK off
# it is used as written.
TAGS_CHECK = True
USE_TAGS = False
TAGS_ISOLATED = False  # tag features only in the tags expert, not the dead/transition/touch ones
TAGS_HARD = False      # train on the harder tag corruption too (tagging.simulate)
TAGS_AGREEMENT = False  # tagging.agreement() as features: on in the TAGS_ISOLATED/HARD runs
# Set pieces are read for their timing, not their team (TAGS_SET_TEAMS off): each is "a restart
# here", like the throw-ins the provided file tags N/A, and the check judges the open-play groups
# alone. Only 33 of the provided game's 59 set pieces name a team -- too few to judge by. Under the
# refreshed profile they fell 0.006 short of a floor that had moved by 0.06 between noise
# versions, and as a judged group they took every tag down with them; on one held-out DFL game a
# look-alike set group was flipped wrongly. Seven held-out DFL games, look-alike tags: no tags
# 0.703; set pieces without their team 0.733, +0.030 (+0.022 to +0.038), better on 7 of 7; with
# their team 0.730, -0.003 (-0.007 to +0.001); dropped altogether 0.728, so the timing is worth
# +0.005 (+0.000 to +0.010).
TAGS_SET_TEAMS = False
# LSTM flow matching (flow.py): a second model from the same features, whose first step's P(H/A/D)
# is blended with the gate's before the decoder, FLOW_WEIGHT of it. Seven held-out DFL games,
# measured profile, look-alike tags, FLOW_COPIES noisy copies of each training game, 3000 steps:
# the mixture 0.733; the flow alone 0.712 through its own sampling (eight rounds, most confident
# frames first) and 0.721 from its first step's argmax -- sampling on costs -0.009 (-0.013 to
# -0.005); blended half and half through the mixture's decoder 0.753, +0.019 (+0.013 to +0.026)
# over the mixture, better on 7 of 7; its first step alone through that decoder 0.749, +0.016
# (+0.007 to +0.024). Metrica's three games (real tracker output), both trained on the seven DFL
# games: the mixture 0.704, the flow alone 0.703 sampled and 0.682 from its first step, blended
# 0.724, +0.020 (+0.009 to +0.032), better on 3 of 3. The weight was fixed beforehand; 0.5 and 1.0
# are the only ones tried. 0 leaves the flow out of `train`. Those runs computed the flow's
# probabilities on Apple's GPU in the process that had trained it, where batches shaped as the
# training ones come out wrong (reports/claude/diag/refirst.py); `predict` computes them on the
# CPU, and so measured, the blend is 0.737 over the seven DFL games and Metrica's three, not 0.744.
FLOW_WEIGHT = 0.5
# The flow's probabilities raised to 1 / FLOW_TEMPERATURE and renormalised before the blend
# (pipeline.sharpen). Chosen on the seven DFL games alone, from 1, 0.7, 0.5, 0.35 at weights 0.5 to
# 0.7: the grid is flat there (0.740 to 0.745) and 0.5 at weight 0.5 tops it, +0.002 over 1.
# Metrica's three games, not used in choosing: +0.012 (0.722 -> 0.735). All ten: +0.005 (+0.002
# to +0.008), better on 6 of 10. A bundle trained before this read it from here.
FLOW_TEMPERATURE = 0.5
FLOW_COPIES = 4
# The touch expert (experts.py): "a player of team K touches the ball within TOUCH_TOLERANCE
# frames". Touches are read off the clean DFL ball before the noise hides it: its velocity jumps
# by more than TOUCH_TARGET_DV with a player within TOUCH_TARGET_RADIUS. Checked against DFL's
# event data on two games: event times sit a median of 2 frames from such a touch by the same
# team, and 78-80% of possession changes have a touch by the new team within 5 frames. Measured
# on four held-out games: +0.017 spellF1 over the touch features alone (90% interval +0.008 to
# +0.027), better on 3 of 4; turnovers of 1-2 s found 0.22 of the time, from 0.15.
TOUCH_EXPERT = True
TOUCH_TOLERANCE = 3        # 6 measured worse: -0.008 (90% interval -0.016 to -0.000)
TOUCH_TARGET_DV = 2.0      # m/s
TOUCH_TARGET_RADIUS = 1.5  # metres
# Metrica's three open sample games (metrica.py) as training games beside the seven DFL ones:
# Metrica's own automatic tracking, possession labels built from their human event data. Measured
# without the fitted paths: on the seven DFL games held out, -0.001 spellF1 (90% interval -0.009
# to +0.006); on the Metrica games held out, each against a model of the DFL games alone, +0.078
# (+0.064 to +0.093), 0.650 to 0.728, better on 3 of 3. About half of that is dead-ball timing
# (+0.038 with the true dead ball given to both), which carries over only if the provided game's
# ground truth marks dead ball as the events do; the other half is who has the ball. With the
# fitted paths on: DFL held out -0.012 (-0.019 to -0.004), worse on 7 of 7; Metrica held out
# +0.083 (+0.069 to +0.097), 0.659 to 0.742. Metrica's ball path is smoothed (0.001 m of jitter),
# and the fitted paths read sharp velocity changes, so a model of both learns a blurred touch.
# With the ball lost through stoppages (DEAD_BALL_LOSS) as well: DFL -0.014 (-0.023 to -0.005);
# Metrica +0.005 (-0.006 to +0.016) -- the stoppage losses carry what the Metrica games taught.
METRICA_GAMES = False
# Noise (noise.py): the ball lost through stoppages. Metrica's own tracking has no ball from the
# frame after a stoppage starts to its restart (98% of dead-ball frames, 7-13% of live ones). The
# DFL games track it throughout, so their noisy copies lost it as often in a stoppage as in play
# (36-40% of dead frames against 42-46% of live ones), and the published baseline -- dead ball
# wherever no ball is seen within 2 s -- called 1% of their frames dead (dead-ball F1 0.02). On
# the provided game the same baseline scores macroF1 0.672, which needs a dead-ball F1 of 0.5 or
# more: the provided ATD loses the ball through stoppages too. Each stoppage of a labelled game
# loses the ball with probability DEAD_BALL_LOSS, from a geometric number of frames (mean
# DEAD_BALL_EDGE) after it starts until as many before the restart -- assumptions, since how
# exactly the ATD loses it takes labels to measure. 0 turns it off. Measured with fitted paths,
# seven DFL games: on held-out copies that lose the ball through stoppages, +0.017 spellF1 (90%
# interval +0.008 to +0.025), better on 6 of 7; on Metrica's three games, whose own tracking does,
# from DFL training alone, +0.070 (+0.053 to +0.089), 0.659 to 0.729, better on 3 of 3. If the
# ATD kept the ball through stoppages after all, -0.026 (-0.035 to -0.017).
DEAD_BALL_LOSS = 0.9
DEAD_BALL_EDGE = 25
# The share of live-play frames whose ball is lost, on labelled games; None: whatever the
# profile's missing fraction leaves once the stoppages have lost theirs (2-15% on the DFL games,
# which mark 42-46% of a match dead). profile cannot measure it: on Metrica's own tracking the
# share of frames missing in short gaps reads 0.3-1.7% against a true 7-13%, since live play
# loses the ball in long gaps too (mean 114 frames). Measured train x held-out copies, seven DFL
# games, plus Metrica's three from DFL training: 0.3 against None, +0.010 (+0.002 to +0.018) on
# copies like None's, +0.009 on copies at 0.2, +0.016 on copies at 0.3, but -0.022 (-0.038 to
# -0.007) on Metrica's own tracking, 0 of 3 -- the only column whose losses are not simulated;
# 0.2 against None, no difference shown on any. So None.
LIVE_BALL_LOSS = None
# How long the ball's losses in live play last. The profile's mean gap (431 frames on the provided
# ATD) mixes stoppages, which lose the ball for their whole length, with losses in play; used as
# the length of the latter, it gave copies whose gaps and runs of sightings read 1.3-10 times the
# provided ATD's at every quantile (shortest tenth of gaps 70-120 frames against 12). With
# LIVE_BALL_FIT, each labelled game's losses in play take the mean length, from
# noise.LOSS_MEANS, whose gap and run quantiles come nearest the profile's
# (noise.calibrate_losses): 75-125 frames on the seven DFL games, shorter where more of the game
# is stoppage, near the 114 Metrica's own tracking shows (above). Seven held-out DFL games, every
# held-out copy fitted, path smoothing on, the measured profile: trained on fitted copies 0.735,
# on the profile's mean 0.745 -- -0.010 (-0.018 to -0.002), better on 3 of 7. Long losses in
# training carry over to short ones, not the reverse; so off, and the copies' gaps stay longer.
LIVE_BALL_FIT = False
# A slow drift of the ball's position: low-passed white noise at BALL_DRIFT_HZ, BALL_DRIFT_M per
# axis, added to every copy after its jitter. At its nearest, the provided ATD's ball reads 1.19 m
# from the nearest player (noise.ball_lead) where the copies read 0.89-1.00; a drift too slow to
# show in the accelerations or the straight-line share would put that there (0.5 m: 1.07-1.13,
# and the distance at no shift as the ATD's). Seven held-out DFL games, path smoothing on, the
# measured profile: on held-out copies drifting 0.5 m, trained without 0.719, with 0.730 --
# +0.011 (+0.005 to +0.017), better on 6 of 7; on copies without drift, 0.741 and 0.751 --
# +0.009 (+0.003 to +0.015), 6 of 7. It helps whether or not the ATD drifts: under a smoothed
# ball, exact distances to it are worth less than the model trained without drift thinks.
# 1.0 m is worse than 0.5 on both: -0.027 (-0.035 to -0.020) without drift, -0.010 (-0.018 to
# -0.003) with. 0 turns it off.
BALL_DRIFT_M = 0.5
BALL_DRIFT_HZ = 0.2
# A tracker that smooths its output spreads every sharp change of velocity -- a kick, a touch --
# over several frames. `profile` measures how sharp the provided ATD's accelerations are when read
# over 1 to 16 frames (noise.path_shape): flat at about 37 m/s^2 for its ball from 1 to 8 frames,
# where DFL's own ball reads 210-530 over 1 frame; and the share of frames on a straight line. With
# PATH_SMOOTHING, each copy's clean paths are first low-passed to match that game's accelerations
# to the profile's, then given the share of the profile's jitter that matches its straight-line
# share (noise.calibrate_paths, positions only). Its false detections are split as the profile
# finds them in play, inside the tracks or in the losses -- none inside when the profile's tail
# of one-frame accelerations has no room for them: the provided ATD's stops at 108 m/s^2 at the
# 99.9th percentile, so the 0.16% of fast jumps it counts in play are fast balls. measure() of a
# copy then reads like the profile: ball 39-42 m/s^2 over 1-4 frames, 36-38 over 8, 29 over 16,
# straight 0.32-0.34, one-frame 99.9th percentile 80-131; false detections inside the tracks, as
# counted, put that at 19,000-26,000.
# Measured on seven held-out DFL games, every held-out copy made this way, the measured profile:
# trained on sharp copies 0.566, on copies made this way 0.737 -- +0.170 (90% interval +0.159 to
# +0.180), better on 7 of 7. The same model scores 0.783 on sharp held-out copies, which is what a
# smoothing tracker would cost a model that never saw one. A profile with no path shape (the
# defaults) leaves the copies sharp.
PATH_SMOOTHING = True
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
# The gate also reads, per team, frames since and until the touch expert last and next said
# P(touch) > 0.5 -- "who touched it last", carried far beyond the gate's +-2 s of shifts. Off:
# on four held-out games it changed spellF1 by -0.001 (90% interval -0.010 to +0.008).
GATE_TOUCH_CLOCKS = False
# What the gate's team model trains on. spellF1 counts every possession once, however long, but a
# frame-level model counts frames: a 60 s possession outweighs a 1 s turnover sixty times. Each
# row is weighted by (its possession's length) ** -GATE_SPELL_WEIGHT, 0 meaning unweighted, and
# GATE_STATE_ALL_ROWS adds the rows sampled densely around changes, where short possessions are.
# Both off: on four held-out games each found more real possessions (0.565 and 0.569 of them,
# from 0.542) but about as many false ones -- spellF1 -0.005 and -0.006, intervals spanning 0.
GATE_SPELL_WEIGHT = 0.0
GATE_STATE_ALL_ROWS = False
# A second gate pass (the refiner): a gate that reads the first gate's P(H/A/D) and P(change)
# over +-10 s beside the experts', trained on first-gate outputs from games that gate never saw.
# 1: measured on four held-out games, a second pass changed spellF1 by -0.003 (90% interval
# -0.011 to +0.006), better on 1 of 4.
GATE_PASSES = 1

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
# The decoder search maximises spellF1, the ranking metric; macroF1, the tie-break, enters with
# this small weight so it decides between near-equal settings and never outweighs spellF1.
DECODER_MACRO_WEIGHT = 0.01
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
