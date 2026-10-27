# Possession by a mixture of experts

Four specialist models, a learned gate that reads all of them over about ten seconds and decides
how far to trust each, and a decoder that turns per-frame probabilities into whole possessions.

```
tracking ─┬─ ball expert ───────┐
          ├─ shape expert ──────┤
          ├─ dead-ball expert ──┼─ gate: reads every expert over ±10 s
          └─ transition expert ─┘     ├─ P(H/A/D) per frame ──────────┐
                                      └─ P(change to H/A/D) per frame ┴─ decoder ── submission
```

## Run it

```bash
pip install -r requirements-possession.txt

python -m possession download                    # 7 open DFL matches, ~2.6 GB, once
python -m possession profile                     # noise statistics of the provided ATD
python -m possession evaluate --quick --jobs 7   # honest held-out score on DFL
python -m possession train                       # -> models/moe.pkl
python -m possession predict --check             # -> evaluation/results/moe.csv, made twice, compared
python -m pytest tests/ -q                       # synthetic data only, ~30 s
```

To compare two versions, evaluate each into its own folder and compare them:

```bash
python -m possession evaluate --quick --jobs 7 --report-dir reports/before
# ... change something ...
python -m possession evaluate --quick --jobs 7 --report-dir reports/after
python -m possession compare reports/before reports/after
```

`compare` prints each game's spellF1 under both versions and the mean difference with a 90%
*paired* bootstrap interval: each draw resamples every game's true possessions and scores both
versions on the same draw. Keep a change only if that interval excludes zero.

`profile` and `predict` read the provided match. Run them yourself: the terms forbid giving the
match data, or extracts of it, to any hosted model.

**Threads.** On the machine this was built on, model fitting was 4–5× *faster* on one thread
than on twelve (OpenMP oversubscription). Before a long run, time
`python -m possession --threads 1 train --quick` against `python -m possession train --quick`.
`evaluate --jobs N` runs N held-out games side by side, one thread each; on a machine with 8 or
more cores that is the fastest way to evaluate. Neither setting changes any result.

## The method

**Training data: other games only.** The seven IDSSE/DFL Bundesliga matches (Bassek et al.,
*Scientific Data* 2025, CC BY 4.0) carry a per-frame ball status and possessing team. That is the
same kind of signal the challenge ground truth is derived from ("D where the corrected tracking
marks the ball dead, otherwise the team it names as owning the ball", CHALLENGE.md), so these
labels are mechanical, not human. None of the matches is the provided game.

**Noise injection** (`noise.py`). DFL tracking is clean; the provided ATD is not. Every training
game is corrupted to match the ATD's measured statistics: ball dropout in runs, more likely when
players crowd the ball; players dropping in and out; occasional false ball detections; positional
jitter. `profile` measures all of these from the ATD's positions alone — never a label:

- missing-data rates and run lengths, directly;
- jitter from second differences. At 25 fps a real player's acceleration moves
  `x[t+1] − 2x[t] + x[t−1]` by millimetres, so its spread is measurement noise (√6 × the jitter,
  for white noise). Checked on DFL: 0.05 / 0.10 / 0.20 m of added jitter is measured as
  0.050 / 0.100 / 0.200 m, and clean DFL as 0.006 m. A tracker that smooths its output hides some
  noise, so on the ATD this is a floor;
- how often a team has more than eleven tracks at once (`team_overfull_frac`). Referees or
  duplicate tracks filed under a team corrupt every team statistic. A bug that did exactly this
  to the DFL training data cost 0.10 spellF1 on the two games it touched. If the ATD shows it,
  the next step is a filter for extra tracks, not more modelling.

**Features** (`features.py`). About 300 per-frame columns, all statistics over a team's visible
players, never a single track slot (slots are tracklets, not people). Centroids, spreads, line
heights, speeds, forward velocity, nearest-player distances, players near the ball, ball speed,
its change (a kick speeds the ball up, a reception slows it) and turning, distance to the lines,
how long since the ball was seen — each raw and averaged over centred windows of 0.2, 1 and 3 s,
plus past-versus-future contrasts over 1 s. The series that time a touch (who is nearest the
ball, how near, ball speed) also get contrasts over 0.2 and 0.5 s: a one-second window cannot
place a change more precisely than about half a second. Which end each team attacks is read
from the data per period. Features are stored per team and assembled in either order, so every
model is trained on both orders with H and A swapped and evaluated on both — no model can learn
"the home team usually has the ball".

**Experts** (`experts.py`), each a gradient-boosted tree ensemble that takes missing values
natively:

| expert | sees | predicts |
|---|---|---|
| ball | ball-relative features only | H / A / D |
| shape | team shape and movement only, no ball | H / A / D — works on the 41% of frames without a ball |
| dead | everything | ball dead or alive |
| transition | local features and before/after contrasts | a change *to H, to A or to D* within 3 frames |

**Gate** (`gate.py`) — also the sequence model. Two tree ensembles over the experts' outputs plus
context (is the ball seen, how long since, how many players are tracked). They see each output at
the frame itself, shifted by up to two seconds either way, and averaged over windows of up to ten
seconds, so they can learn what no single frame shows: the ball expert flipping a few frames
before the shape expert, a pass that stays a pass because the other team never gets near it. One
predicts P(H/A/D) per frame, the other P(a change to H/A/D is near). Both are trained only on
*out-of-fold* expert outputs: each game's predictions come from experts that never saw it.

**Decoder** (`decoder.py`). A Viterbi decode that balances per-frame evidence, the gate's
probability of a change *to each label* and a per-change cost, with exact minimum durations for
possessions and stoppages. Its five numbers are searched on out-of-fold DFL output. It is what
turns flickering frame labels into a realistic number of possessions with sharp edges, which is
what spellF1 scores.

## How it is tested

| tier | data | used for |
|---|---|---|
| held-out DFL | `evaluate`: outer leave-one-game-out, held-out game unseen by experts, gate *and* decoder search | **comparing versions** |
| synthetic | `tests/`: parsers, symmetry, decoder, end-to-end determinism | catching bugs |
| public window | `python evaluation/run_evaluation.py` on the final file | one pass/fail check that the file is aligned and plausible — never to choose between versions |

### Held-out result

Leave-one-game-out over four DFL matches, `--quick` settings (each held-out game predicted by a
model trained on the other three), default noise profile, one thread:

| held out | spellF1 | baseline | macroF1 | baseline | possessions predicted / true |
|---|---|---|---|---|---|
| J03WPY | 0.543 | 0.195 | 0.911 | 0.380 | 314 / 421 |
| J03WN1 | 0.523 | 0.226 | 0.884 | 0.336 | 315 / 500 |
| J03WMX | 0.491 | 0.211 | 0.907 | 0.395 | 280 / 511 |
| J03WOH | 0.597 | 0.199 | 0.913 | 0.358 | 258 / 402 |
| **mean** | **0.539** | 0.208 | 0.904 | 0.368 | |

The first run scored 0.497: the DFL parser put referees in the away team, and two of these files
carry referee tracks. Moving from that run to this one, the unaffected games moved by up to
0.02, which is the per-game noise level; a change smaller than that on most games is not a result.

What it does and does not say:

- It is DFL labels on DFL tracking with simulated noise. The provided game adds real ATD noise
  and Metrica's label conventions, so expect its score to be lower; this is not a forecast.
- `--quick`, three training games per fold. Full settings and six training games should help.

### Where the remaining error is

From the DFL labels alone (no model): merging every possession under 10 frames still scores
0.96, and moving every boundary by a random 5 frames (0.2 s) scores 0.93; at 10 frames it is 0.85.
Boundaries within a few tenths of a second matter far more than very short possessions.

From the held-out folds, swapping one input for the truth and re-tuning the decoder on that game
(upper bounds for choosing what to build, never scores):

| spellF1 on the held-out game | J03WMX | J03WPY |
|---|---|---|
| as submitted (decoder tuned on the training games) | 0.491 | 0.543 |
| decoder search grid widened (still tuned on the training games) | 0.491 | 0.543 |
| decoder re-tuned on the held-out game itself | 0.498 | 0.569 |
| ... plus perfect alive / dead | 0.520 | 0.581 |
| ... plus perfect boundary probabilities instead | 0.783 | 0.820 |
| ... plus perfect team on alive frames instead | 0.951 | 0.964 |

So the decoder and the dead-ball expert are not where the points are. Team errors on alive frames
are under 1% more than 5 s from a possession change and about 40% within 0.5 s of one: the
model knows who has the ball but switches at the wrong moment, and it finds few possessions
shorter than 2 s (recall 0.02–0.15), which are 30% of all possessions.

## Compliance with RULES.md

- **No hand labels.** Every label comes from DFL's tracking or from the model.
- **No ground truth of the provided game.** `guard.check_input` refuses `evaluation/gt/` and any
  `*_possession_gt*` file on every read path; a test checks it.
- **No human data about the provided game.** Training uses only the seven DFL matches.
- **No frame-specific constants.** The only frame numbers used are the published half blocks
  from `games.py`, which decide where X goes.
- **Exact regeneration.** Fixed seeds everywhere, no randomised early stopping, deterministic
  decoding. `predict --check` regenerates the file from scratch and compares sha256.
  Freeze package versions before the final run (`pip freeze > requirements-lock.txt`).
- **No data in the repository.** `cache/`, `models/` and `reports/` are gitignored.
- **No hosted model sees the data.** Everything runs locally.

### What may be committed

The code, `requirements-lock.txt` and, if you want faster regeneration, `models/moe.pkl` — it
holds trees fitted to DFL data plus the noise profile, which is a handful of aggregate rates
(missing fraction, mean gap length). Never commit `cache/` (it holds features of the provided
game) or a submission CSV other than the one you submit.

## Known limits

- **Label conventions.** The model learns DFL's notion of "dead" and "owning", not Metrica's. In
  one DFL match 44% of frames are dead ball and stoppages are long (median 23 s). If the providers
  disagree systematically — how soon a throw-in counts as live — no noise model removes that gap.
- **Noise realism.** Identity switches and fragmented slots are not simulated (no feature can see
  them). Jitter levels are assumptions.
- **Seven games** is a small training set. A held-out gap smaller than the spread across games is
  not a result.
- **Not used yet:** Smart Tagging (no equivalent exists in the DFL games, so the gate cannot learn
  how far to trust it), the homography, and the video.
