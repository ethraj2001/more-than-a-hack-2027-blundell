# Possession by a mixture of experts

Five specialist models, a learned gate that reads all of them over about ten seconds and decides
how far to trust each, and a decoder that turns per-frame probabilities into whole possessions.
A sixth, reading Metrica's Smart Tagging, joins only if the provided game's tags pass a check
against its own tracking. A second model, an LSTM trained by flow matching on the same features,
is blended with the gate before the decoder.

```
tracking ─┬─ ball expert ───────┐
          ├─ shape expert ──────┤
          ├─ dead-ball expert ──┼─ gate: reads every expert over ±10 s
          ├─ transition expert ─┤     ├─ P(H/A/D) per frame ──┬─ ½ ───┐
          └─ touch expert ──────┘     │   LSTM flow model ────┴─ ½ ───┤
                                      └─ P(change to H/A/D) per frame ┴─ decoder ── submission
```

Held out on seven DFL games, with look-alike Smart Tagging on the held-out game, it scores
**spellF1 0.750** (macroF1 0.950) against 0.296 for the published baseline's method on the same
games; without tags, 0.719 (0.943). The held-out copies lose the ball through stoppages, as
Metrica's tracking does, and are easier than the provided game for that baseline (0.296 against
its published 0.243) — see [Held-out result](#held-out-result). On Metrica's three sample games,
from DFL training alone: 0.729.

Those copies keep DFL's sharp ball paths. The provided ATD smooths its paths, and on copies
smoothed and degraded like the profile `profile` measured on it, the model scores **0.737**
(macroF1 0.946) — trained on sharp copies, 0.566. That is the closer estimate for the provided
game; see [Under the measured profile](#under-the-measured-profile).

## Run it

```bash
pip install -r requirements-possession.txt

python -m possession download                    # 7 DFL matches + Metrica's 3 samples, ~2.8 GB, once
python -m possession profile                     # noise statistics of the provided ATD
python -m possession evaluate --quick --jobs 7   # honest held-out score on DFL
python -m possession train                       # -> models/moe.pkl, the flow model included
python -m possession predict --check             # -> evaluation/results/moe.csv, made twice, compared
python -m pytest tests/ -q                       # synthetic data only, ~3 min
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

`profile`, `train` and `predict` read the provided match — `train` only to check its Smart
Tagging against its own tracking (below), and it prints one `tags` line saying whether the model
will read the tags. Run them yourself: the terms forbid giving the match data, or extracts of it,
to any hosted model. `train` corrupts its copies to `models/noise_profile.json`, so re-run
`profile` whenever `noise.measure` gains a statistic: an older profile lacks it, and the copies
fall back to defaults for it (no path shape at all means sharp copies). `evaluate` reads the same
file, so its score is for copies like the provided game.

**Threads.** On the machine this was built on, model fitting was 4–5× *faster* on one thread
than on twelve (OpenMP oversubscription). Before a long run, time
`python -m possession --threads 1 train --quick` against `python -m possession train --quick`.
`evaluate --jobs N` runs N held-out games side by side, one thread each; on a machine with 8 or
more cores that is the fastest way to evaluate. Neither setting changes any result.

**Speed.** Measured on one held-out fold: the decoder runs 12× faster than it did (each chain of
minimum-duration states collapses to a running sum), the feature pass 25× faster at its slowest
step (a sort-based percentile, bit-identical to numpy's), and `evaluate` keeps fitted expert
banks in `cache/banks/` keyed by the exact contents of the tables they were fitted on — the bank
fitted without games g and h serves both held-out g and held-out h, and a change that only
touches the gate or the decoder reuses every bank. On seven games that is 43% fewer expert fits.
All three give identical predictions; tests check each. `train` and `predict` never use the
cache; delete `cache/banks/` whenever you like.

## The method

**Training data: other games only.** The seven IDSSE/DFL Bundesliga matches (Bassek et al.,
*Scientific Data* 2025, CC BY 4.0) carry a per-frame ball status and possessing team. That is the
same kind of signal the challenge ground truth is derived from ("D where the corrected tracking
marks the ball dead, otherwise the team it names as owning the ball", CHALLENGE.md), so these
labels are mechanical, not human. None of the matches is the provided game.

**And Metrica's own sample games** (`metrica.py`). Metrica made the provided game's tracking, and
its three open sample games (github.com/metrica-sports/sample-data) are its own automatic
tracking — the ball lost on 32–41% of frames, as on the provided ATD's 41% — with human event data
of the same games. They carry no per-frame possession, so the labels are built from the events:
a team owns the ball from the start of each of its on-ball events (pass, carry, shot, set piece,
recovery, ball lost, ball out) until the other team's next one, and the ball is dead from a ball
out, a foul, a goal, a shot out or an offside until the restart. Their touches come from the
events too (every on-ball event's start, every pass's reception). The labels look like DFL's —
360–400 possessions a match, 23–26% of them turnovers under 2 s, the labelled team holding the
nearest player to the ball 93% of the time (DFL 90–92%) — except that they mark less of the match
dead, 27–36% against DFL's 41–42%.

**Noise injection** (`noise.py`). DFL tracking is clean; the provided ATD is not. Every training
game is corrupted to match the ATD's measured statistics (a game whose ball is already lost on
some frames, like Metrica's, loses only the rest): ball dropout in runs, more likely when
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

One thing `profile` cannot measure, because it takes labels: *when* the ball goes missing.
Metrica's own tracking has no ball from the frame after each stoppage starts to its restart —
on 98% of dead-ball frames, against 7–13% of live ones. DFL tracks the ball throughout, so the
noisy DFL copies first lost it as often in a stoppage as in play (36–40% of dead frames, 42–46%
of live ones), and the published baseline, which calls a frame dead wherever no ball is seen
within 2 s, called 1% of their frames dead. On the provided game that baseline's macroF1 is
0.672, which takes a dead-ball F1 of 0.5 or more, so the provided ATD loses the ball through
stoppages as well. Each stoppage of a training game now loses the ball with probability 0.9,
from about a second after it starts to about a second before the restart
(`config.DEAD_BALL_LOSS`, `DEAD_BALL_EDGE`), and the crowd-biased losses make up the rest of
the missing fraction. The published score was read only for that direction; no setting is
fitted to it. DFL marks 42–46% of each match dead, so stoppages now take most of the 41%, and
live play keeps 2–15% losses, likely fewer than the ATD has. Setting live play's share directly
(`config.LIVE_BALL_LOSS`) at 0.3 helped on every kind of held-out DFL copy (+0.009 to +0.016)
and lost −0.022 [−0.038, −0.007] on Metrica's own tracking, the only test whose ball losses are
real; at 0.2 it made no difference anywhere. It stays off.

**Path shape** (`config.PATH_SMOOTHING`, on). Jitter near zero can mean a clean tracker or one
that smooths, and a smoothing tracker spreads every touch over several frames, where DFL's ball
changes velocity within one. So `profile` also reads how sharp the ball's and players'
accelerations are over 1, 2, 4, 8 and 16 frames, and the share of frames on a straight line. The
provided ATD's ball reads about 37 m/s² over 1 to 8 frames and 29 over 16; DFL's own ball reads
210–530 over one frame. Each copy's clean paths are first low-passed to match its game's
accelerations to the profile's (a fourth-order zero-phase Butterworth, about 0.7 Hz for the
ball, about 2 Hz for players), then given the share of the profile's jitter that matches its
straight-line share (the profile's jitter is read off a path already curved, so all of it would
roughen the copies past the ATD). False detections follow `profile`'s in-play count, split as it
finds them: single frames inside the tracks, the rest as two-frame flickers in the gaps. On the
provided ATD that count is 0.16% of frame pairs in play (the 1.3% counted over every frame comes
almost all from frames outside the two halves), 99% of them inside tracks — yet its one-frame
accelerations stop at 108 m/s² at the 99.9th percentile, where a single false detection reads
over 1,000. So those jumps are fast balls, which smoothed copies have of their own (0.07–0.12%
with none added), and a tail like that adds no false detections inside the tracks; added as
counted, they put the copies' 99.9th percentile at 19,000–26,000. Whether flickers are there or
not hardly matters to the model either (each way round, trained with and tested without and the
reverse, within 0.007 and no interval clear of zero) but matters to the published baseline's
dead-ball call, which a flicker in a stoppage revives. Read back by `profile`, a copy then
matches the ATD within about 10% over the lags: ball 39–42 m/s² over 1–4 frames, 36–38 over 8,
29 over 16, straight 0.32–0.34, jitter 0.0028 m; and within about a quarter through its tail,
one-frame percentiles 5–6 / 20–22 / 39–43 / 47–59 / 80–131 against the ATD's 5.7 / 18 / 37 / 50
/ 108 (50th to 99.9th). The copies' filter is zero-phase, smoothing both ways in time; a live
tracker can only smooth forwards, which would leave its ball behind its players and every
boundary late. `profile` reads that too (`noise.ball_lead`): the time shift at which the ball is
nearest the players, +4.3 frames on the provided ATD (the ball runs ahead of whoever plays it),
+4.2 to +4.8 on the copies, −3 to −10 with forward-only filters. At that shift the ATD's ball
is 1.19 m from the nearest player, the copies' 0.89–1.00: a drift too slow to show in the
accelerations would do that, so the copies' ball also drifts, 0.5 m per axis at 0.2 Hz
(`config.BALL_DRIFT_M`). That helps on held-out copies with the drift and without it alike
(+0.011 and +0.009, [How it got here](#how-it-got-here)): under a smoothed ball, exact distances
to it deserve less trust than a model trained without drift gives them. On copies made this way the model trained on sharp copies falls from 0.783 to 0.566; trained on
them, it scores 0.737 — +0.170 [+0.159, +0.180], better on 7 of 7
([Under the measured profile](#under-the-measured-profile)).

**Features** (`features.py`). About 325 per-frame columns, all statistics over a team's visible
players, never a single track slot (slots are tracklets, not people). Centroids, spreads, line
heights, speeds, forward velocity, nearest-player distances, players near the ball, ball speed,
its change (a kick speeds the ball up, a reception slows it) and turning, distance to the lines,
how long since the ball was seen — each raw and averaged over centred windows of 0.2, 1 and 3 s,
plus past-versus-future contrasts over 1 s. The series that time a touch (who is nearest the
ball, how near, ball speed) also get contrasts over 0.2 and 0.5 s: a one-second window cannot
place a change more precisely than about half a second. Touches get series of their own: per
team, how hard the ball changed velocity next to one of its players, and the frames since that
team last touched the ball and until it next does, counting a touch from a change of 1 m/s
(5 m/s rarely fires once the ball is smoothed like the ATD's). A touch is a change in the ball's velocity,
and differencing single positions doubles the tracker's jitter: at 0.3 m a frame, a five-frame
difference carries about 4 m/s of noise against a touch's 5 m/s. So the ball's velocity is also
taken from straight lines fitted to its sightings over 6 and 12 frames either side of each
frame, and the change between the two fits, overall and per team next to its nearest player, is
a feature too: on noisy DFL copies it singles out true touches with AUC 0.86, against 0.71 for
the differenced series. Which end each team attacks is read from the data per period. Features are stored per team and assembled in either order, so every
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
| touch | local features and before/after contrasts | a player of H or of A touches the ball within 3 frames |

The touch expert's targets come from the *clean* DFL ball, before the noise hides it: a touch is
where its velocity jumps by more than 2 m/s with a player within 1.5 m. Checked against DFL's
event data for two of the games, event times sit a median of 2 frames from such a touch by the
same team, and 78–80% of possession changes have a touch by the new team within 5 frames. Trained
on copies with the ball hidden on 41% of frames, it learns what a touch looks like in the
players' movement as well as in the ball's. A sixth expert, **tags**, reads Smart Tagging alone;
it and the tag features are added only when the provided game's tags pass a check against its
own tracking ([Smart Tagging](#smart-tagging)). A short-possession expert also exists and is
off (measured, below).

**Gate** (`gate.py`) — also the sequence model. Two tree ensembles over the experts' outputs plus
context (is the ball seen, how long since, how many players are tracked). They see each output at
the frame itself, shifted by up to two seconds either way, and averaged over windows of up to ten
seconds, so they can learn what no single frame shows: the ball expert flipping a few frames
before the shape expert, a pass that stays a pass because the other team never gets near it. One
predicts P(H/A/D) per frame, the other P(a change to H/A/D is near). Both are trained only on
*out-of-fold* expert outputs: each game's predictions come from experts that never saw it.

**Flow model** (`flow.py`) — a second model, from the same features (Smart Tagging's included
when they are read), not the experts. A BiLSTM encodes them over 60-second windows; a second
BiLSTM, the denoiser, is trained by discrete flow matching: each window's labels are masked in
blocks (0.2 s to 25 s), each with probability 1 − t for t drawn uniformly, and the masked labels
predicted from the features and the labels left showing. What the pipeline reads is its first
step — P(H/A/D) per frame with every label masked — averaged half and half with the gate's
before the decoder. Run on as a generator (unmasking over eight rounds, most confident frames
first) it scored lower than its first step on the DFL games, and higher on Metrica's (below);
blended, its first step gains on both — +0.019 on the seven DFL games, +0.020 on Metrica's three
real-tracker games. It trains on four noisy copies of each game
for 3,000 steps, about 25 minutes on an M2 Max's GPU (`train` uses the GPU where there is one;
`--device cpu` is about ten times slower). A GPU's floats are not reproducible, so the weights
are part of the model file; prediction runs on the CPU in float64 with its probabilities rounded
to six decimals, so the same model file gives the same submission anywhere. `train-flow` adds
the flow model to a trained `models/moe.pkl` without retraining the mixture.

**Decoder** (`decoder.py`). A Viterbi decode that balances per-frame evidence, the gate's
probability of a change *to each label* and a per-change cost, with exact minimum durations for
possessions and stoppages. Its five numbers are searched on out-of-fold DFL output for the best
spellF1, with macroF1 (the tie-break) at weight 0.01 so it only decides between near-equal
settings. It is what turns flickering frame labels into a realistic number of possessions with
sharp edges, which is what spellF1 scores.

## How it is tested

| tier | data | used for |
|---|---|---|
| held-out DFL | `evaluate`: outer leave-one-game-out, held-out game unseen by experts, gate *and* decoder search | **comparing versions** |
| synthetic | `tests/`: parsers, symmetry, decoder, end-to-end determinism | catching bugs |
| public window | `python evaluation/run_evaluation.py` on the final file | one pass/fail check that the file is aligned and plausible — never to choose between versions |

### Held-out result

Leave-one-game-out over all seven DFL matches, `--quick` settings (each held-out game predicted
by a model trained on the other six), default noise profile, the ball lost through stoppages on
training and held-out copies alike. The held-out game carries look-alike Smart Tagging, and the
check decides, exactly as `train` does for the provided game, whether its model reads them —
here it did on all seven (one realigned by 30 frames):

| held out | spellF1 | baseline | without tags | macroF1 | baseline | possessions predicted / true |
|---|---|---|---|---|---|---|
| J03WMX | 0.675 | 0.283 | 0.668 | 0.945 | 0.757 | 386 / 511 |
| J03WN1 | 0.708 | 0.307 | 0.719 | 0.938 | 0.743 | 385 / 500 |
| J03WOH | 0.778 | 0.306 | 0.744 | 0.953 | 0.747 | 386 / 402 |
| J03WOY | 0.809 | 0.302 | 0.732 | 0.958 | 0.746 | 428 / 456 |
| J03WPY | 0.752 | 0.299 | 0.707 | 0.955 | 0.744 | 328 / 421 |
| J03WQQ | 0.772 | 0.286 | 0.709 | 0.947 | 0.735 | 414 / 401 |
| J03WR9 | 0.753 | 0.292 | 0.755 | 0.955 | 0.773 | 368 / 370 |
| **mean** | **0.750** | 0.296 | 0.719 | 0.950 | 0.749 | |

The same model on copies whose ball loss ignores stoppages, as these held-out games were scored
before: 0.649 (the model trained that way scores 0.675 there).

The tags' share of that is what tags *like the look-alikes* are worth; what the real tags are
worth only the final score can show. If the check refuses them, the model is the one in the
"without tags" column.

`train`'s full settings (stride 10, two noise copies, 200 trees) against `--quick` on four of
the games: 0.636 against 0.630, +0.006 (90% paired interval −0.004 to +0.018), better on 3 of 4,
for about six times the fitting time.

What it does and does not say:

- It is DFL labels on DFL tracking with simulated noise. The provided game adds real ATD noise
  and Metrica's label conventions; this is an estimate, not a forecast.
- The baseline is the published method run on the same noisy DFL copies. It scores 0.296 here
  (macroF1 0.749) and 0.243 (0.672) on the provided game's full match, so these copies are
  easier than the real tracking. Live play loses the ball on only 2–15% of frames here, but that
  is not all of it: on copies losing it on 30% of live frames the baseline scores 0.269 and its
  macroF1 does not move (0.745), so the rest is in dead ball — how the ATD loses the ball through
  stoppages, or how the ground truth marks them. Before stoppage losses it scored 0.210 (0.376):
  harder.
- Measured under scikit-learn 1.9.1. Under 1.8.0 the seven DFL games score the same, +0.001
  [−0.007, +0.008], and Metrica's three −0.010 [−0.016, −0.004] (0.720), so the lock file
  matters (Compliance below).

### Under the measured profile

The same seven folds with every copy made from the profile `profile` measured on the provided
ATD, instead of the defaults. Sharp: the copies keep DFL's sharp paths. Smoothed: the held-out
copy is smoothed like the ATD ([Path shape](#the-method)) — the column that resembles the
provided game.

| held out | sharp model, sharp copy | sharp model, smoothed copy | smoothed model, smoothed copy | macroF1 | baseline | possessions predicted / true |
|---|---|---|---|---|---|---|
| J03WMX | 0.735 | 0.526 | 0.669 | 0.940 | 0.180 | 360 / 511 |
| J03WN1 | 0.781 | 0.567 | 0.723 | 0.939 | 0.185 | 386 / 500 |
| J03WOH | 0.788 | 0.567 | 0.730 | 0.947 | 0.190 | 351 / 402 |
| J03WOY | 0.818 | 0.600 | 0.766 | 0.950 | 0.173 | 373 / 456 |
| J03WPY | 0.788 | 0.589 | 0.759 | 0.951 | 0.182 | 332 / 421 |
| J03WQQ | 0.793 | 0.553 | 0.721 | 0.936 | 0.197 | 354 / 401 |
| J03WR9 | 0.776 | 0.563 | 0.788 | 0.960 | 0.168 | 350 / 370 |
| **mean** | 0.783 | 0.566 | **0.737** | 0.946 | 0.182 | |

Smoothing costs a model that never saw it 0.217; training on it wins back 0.170 [+0.159,
+0.180]. What is left, 0.046, is mostly short possessions after a turnover. Of those under 1 s,
the smoothed model finds 23% against 37% on sharp copies; from 1 to 2 s, 44% against 58%. The
truth copied into every possession under 2 s after a turnover would add 0.120 here (0.086 on
sharp copies).

The table is from the profile's first reading, which counted every frame's false detections
and so put 1.3% of flickers into the copies' gaps. Read again in play only, with the tail that
rules out false detections inside the tracks ([Path shape](#the-method)), and with touch clocks
from 1 m/s, the smoothed column reads 0.741 (macroF1 0.948) — no different from 0.744 with the
flickers, −0.002 [−0.010, +0.005]. The published baseline moves more: 0.323 (macroF1 0.744)
without the flickers, 0.182 (0.453) with them, against 0.243 (0.672) on the provided game.

The mismatch costs the other way too. On Metrica's three games, through their own tracker's
paths, which are sharp (joined straight lines), the model trained on sharp copies scores 0.763
and the one trained on smoothed copies 0.674: −0.089 [−0.107, −0.071], worse on 3 of 3. So the
switch is right only because the profile says the provided ATD smooths, and it says so plainly:
its ball reads no more than 37 m/s² over one frame on 99% of frames, where unsmoothed tracking,
DFL's or Metrica's, reads over 200.

### How it got here

Every row is one change, measured against the row before it on the same held-out games with
`compare`: the mean spellF1 difference, its 90% paired bootstrap interval, and on how many games
it helped. A change was kept only when the interval excluded zero.

| version | mean spellF1 | change [90% interval] | better on |
|---|---|---|---|
| first build (4 games) | 0.497 | | |
| referees no longer filed under the away team | 0.539 | | |
| short timing windows; boundary target names the destination | 0.554 | +0.016 [+0.008, +0.025] | 4 / 4 |
| gate reads every expert over ±10 s | 0.600 | +0.046 [+0.034, +0.057] | 4 / 4 |
| touch features | 0.612 | +0.012 [+0.002, +0.023] | 3 / 4 |
| touch expert | 0.630 | +0.017 [+0.008, +0.027] | 3 / 4 |
| the same, seven games | 0.641 | +0.012 [+0.005, +0.019] over touch features | 6 / 7 |
| Smart Tagging behind the check, look-alike tags (seven games) | 0.666 | +0.024 [+0.016, +0.033] | 6 / 7 |
| ball velocity from fitted paths, not single positions (seven games) | 0.675 | +0.009 [+0.003, +0.016]; without tags +0.003 [−0.002, +0.008] | 6 / 7; 5 / 7 |
| ball lost through stoppages in training (seven games; held-out copies lose it too, so 0.733 before) | 0.750 | +0.017 [+0.008, +0.025]; Metrica's three games, DFL training only, +0.070 [+0.053, +0.089] | 6 / 7; 3 / 3 |
| paths smoothed like the ATD (seven games, measured profile; held-out copies smoothed too, so 0.566 before) | 0.737 | +0.170 [+0.159, +0.180] | 7 / 7 |
| touch clocks from a 1 m/s change, not 5 | 0.744 | +0.007 [+0.0001, +0.014] | 4 / 7 |
| the ball drifting slowly, 0.5 m, in training (refreshed profile; 0.741 before, held-out copies not drifting) | 0.751 | +0.009 [+0.003, +0.015]; held-out copies drifting too, +0.011 [+0.005, +0.017] | 6 / 7; 6 / 7 |
| set pieces read for their timing, not their team (held-out copies drifting, so 0.730 before) — kept because the provided game's set pieces fail the check and took every tag with them: without tags, 0.703 | 0.733 | +0.003 [−0.001, +0.007]; over no tags +0.030 [+0.022, +0.038] | 5 / 7; 7 / 7 |
| the LSTM flow model blended half and half with the gate (weight fixed beforehand); Metrica's three games, real tracker output, DFL training only: 0.704 to 0.724 | 0.753 | +0.019 [+0.013, +0.026]; Metrica +0.020 [+0.009, +0.032] | 7 / 7; 3 / 3 |
| the flow's probabilities measured as `predict` computes them, on the CPU (a fix to the measurement, not the model: the harness had computed them on Apple's GPU in the process that trained the net, where batches shaped as the training ones come out wrong; reports/claude/diag/refirst.py). Ten games from here, the seven DFL and Metrica's three; 0.744 as measured before | 0.737 | −0.007 [−0.012, −0.003] | 3 / 10 |
| the flow's probabilities sharpened before the blend, temperature 0.5 (config.FLOW_TEMPERATURE; chosen on the seven DFL games, where the grid is flat; Metrica's three, not used in choosing, +0.012) | 0.742 | +0.005 [+0.002, +0.008] | 6 / 10 |

Tried and not kept, each against the version it would have joined (four games unless stated):

| idea | change [90% interval] | why it was plausible |
|---|---|---|
| carrier model (who is on the ball, players only) | −0.002 [−0.012, +0.008] | reads possession through ball-lost gaps |
| change probability from the transition expert instead of the gate | −0.002 [−0.009, +0.006] | |
| wider decoder search | −0.006 to +0.002 per game | |
| boundary tolerance 1 or 2 frames instead of 3 | −0.013 [−0.021, −0.005]; −0.003 [−0.012, +0.006] | sharper edges |
| short-possession expert | −0.002 [−0.012, +0.006] over touch features, 46% more time | the possessions spellF1 misses |
| reach-first (time for each team to reach the ball) | −0.005 [−0.013, +0.004] | interceptions |
| player acceleration | −0.006 [−0.016, +0.004] | receivers brake |
| touch window ±6 frames | −0.008 [−0.016, −0.000] | ball-free touches cannot be timed to 3 |
| last-touch clocks in the gate | −0.001 [−0.010, +0.008] | possession is who touched last |
| a second gate pass over the first gate's output | −0.003 [−0.011, +0.006] | |
| gate rows weighted by possession length^−0.5 | −0.005 [−0.016, +0.005] | spellF1 counts possessions, not frames |
| gate's team model on every sampled row | −0.006 [−0.016, +0.005] | |
| gap bridging and contested-ball features | +0.001 [−0.007, +0.009] | loose balls |
| Smart Tagging always read | see below | |
| fragmentation-risk expert (possession changing hands twice within 5 s), seven games | +0.005 [−0.002, +0.011]; without tags +0.000 [−0.004, +0.005] | the scrambles short turnovers come in |
| decoder change cost lowered where the gate expects many changes, seven games | −0.000 | |
| the flow model alone, its first step's argmax, instead of the mixture, seven games | −0.012 [−0.021, −0.003], 3 / 7 | a sequence model reading every feature directly |
| the flow model's own sampling (eight rounds, most confident frames first) instead of its first step's argmax | −0.009 [−0.013, −0.005], 2 / 7; on Metrica's three +0.020, where the first step reads up to 9% too many possessions | possession edges decided last, with both sides known |
| the flow model's first step alone through the mixture's decoder (weight 1, not ½) | −0.004 against ½: 0.749, +0.016 [+0.007, +0.024] over the mixture | |
| set pieces dropped from the tags rather than read without their team, seven games | −0.005 [−0.010, −0.000] | the provided game's set pieces fail the check |
| both models trained on J03WMX alone (its halves as two games for the gate's cross-fitting), the blend scored on the other six DFL games and Metrica's three | −0.029 [−0.038, −0.020], 0 / 9; the mixture alone −0.042 [−0.051, −0.033], 0 / 9; on J03WN1, the other top-flight game, −0.051 | Köln–Bayern is the seven's most Barcelona–Athletic-like game (a possession side against a man-marking press; the most possessions, half under 3 s) and the lowest scored, 0.698 |
| both models trained with SkillCorner's two 2019/20 Clásicos (broadcast tracking, labels from StatsBomb's events; skillcorner.py) beside the seven DFL games, DFL held out in turn, Metrica's three by a model of all nine | −0.008 [−0.015, −0.002], 5 / 10 (Metrica +0.001 to +0.009, DFL −0.042 to +0.009); the mixture alone −0.017 [−0.024, −0.011], 1 / 10; the Clásicos alone −0.212 [−0.226, −0.199], 0 / 10 | Barcelona's own style, but a broadcast's partial view (13 of 22 players, 27% of play unseen) and labels from another provider's events |
| decoder grid widened (alpha 0.01–0.15, beta 0.5–3, minimum possession 5–30 frames, minimum stoppage 5–20, change cost −2 to +2), searched as `train` searches, on each fold's out-of-fold gate output; the blend decoded with it, seven DFL games and Metrica's three | −0.004 [−0.008, +0.001], 4 / 10; the mixture alone −0.003 [−0.008, +0.001]; chosen instead on the other games' held-out blends −0.004 [−0.008, −0.001]; the best single setting in hindsight over all ten +0.006, an upper bound | every fold had picked the old grid's edge in three of its four settings (alpha 0.05, beta 1, minimum possession 15, change cost 0) — a flat optimum, not a clipped one |
| the flow model's share of the blend 0.3–0.8 instead of ½, ten games, from saved probabilities | best 0.6–0.7, +0.003 in hindsight; chosen per game on the other DFL games +0.002 | only ½ and 1 had been tried |
| the decoder's change probabilities refitted from the gate's outputs and the flow's first step read at shifts up to 1 s (a boundary model on saved held-out outputs, fitted on the other games), blended half and half with the gate's, ten games | +0.003 [−0.000, +0.006], 7 / 10; over the same refit from the gate alone +0.003 [−0.001, +0.006] | the decoder weighs change probabilities twenty times its per-frame evidence, and the flow reached only the latter |
| the flow model trained with every label masked on half its windows (flow.FULL_MASK 0.5), seven DFL games and Metrica's three | blend +0.004 [−0.000, +0.008], 7 / 10 (J03WR9 −0.013, Metrica +0.006 to +0.017); the flow's first step alone −0.009 [−0.014, −0.004], 3 / 10 | the first step, the only one the pipeline reads, is every label masked, which block masking reaches only as t nears 0; the labels left showing are what teach it, and two GPU trainings differ by about this much |
| a player-set branch in the flow model (every visible track slot an anonymous token, its position and motion relative to the ball turned to its team's attacking direction; a shared MLP and attention pooling per team, after Ball Radar's set encoder; reports/claude/diag/flow_players.py), seven DFL games and Metrica's three | blend −0.008 [−0.013, −0.003], 2 / 10; the flow's first step alone −0.009 [−0.015, −0.002], 4 / 10; training loss unchanged (0.065 at 3,000 steps, against 0.064). With a self-attention layer across the players first (a Set Transformer, as Ball Radar's is): blend −0.006 [−0.011, −0.001], 3 / 10; the flow alone +0.001 [−0.006, +0.008], 5 / 10 (+0.010 to +0.040 on J03WQQ, J03WR9 and Metrica's three, −0.017 to −0.026 on the other five) | the literature's possession models read every player, not team statistics; here the per-player view fitted the training games no better, so the team features already carry what it saw |
| two or three flow models' first steps averaged before the blend (the base, FULL_MASK 0.5, the player set), from saved probabilities | base + FULL_MASK +0.001 [−0.002, +0.004], 7 / 10; base + player set −0.006; all three −0.001 | if the flow's error were training noise, averaging models would cut it |
| curriculum, ball seen → ball lost: the flow model trained at first on copies of the same games with the ball never lost, phased out by 60% of training, then on the lossy copies alone (reports/claude/diag/flow_curriculum.py, CURRICULUM=ball), seven DFL games and Metrica's three | blend +0.002 [−0.002, +0.006], 7 / 10; the flow's first step alone +0.006 [+0.001, +0.011], 6 / 10 (J03WQQ +0.051, J03WOH −0.017) | the ball is lost on 41% of the provided match's graded frames; a model that learns possession first where the ball is seen might carry it over to where it is not. The flow's gain is about what two GPU trainings differ by, and the blend does not keep it |
| curriculum, settled → turnovers: of 8 window starts, one drawn by a softmax over the share of the window in possessions under 2 s, its weight running from settled (−2) to turnovers (+2) over 70% of training, then windows as they come (CURRICULUM=scenario) | blend +0.002 [−0.002, +0.006], 6 / 10 (Metrica +0.001 to +0.018, J03WR9 −0.015); the flow's first step alone −0.008 [−0.013, −0.003], 3 / 10, and it splits possessions more (J03WQQ 440 spells against 401). Averaging the ball and scenario flows before the blend +0.002 [−0.001, +0.006]; with the base flow as well +0.001 | short possessions after turnovers are where the README's bounds put most of what is lost (+0.35 copying their starts); shown more of them, the flow finds more, not the right ones |
| curriculum, labels shown → none shown: the share of labels left showing drawn from U(lo, 1), lo falling from 0.5 to 0 by half way, then every label masked on a share of windows rising to 0.3 (CURRICULUM=mask) | blend −0.001 [−0.005, +0.003], 5 / 10; the flow's first step alone −0.014 [−0.019, −0.008], 1 / 10 | as FULL_MASK, trained toward the one case the pipeline reads; and as with FULL_MASK, windows with nothing showing teach the first step less than windows with labels to fill in between |
| self-training on the held-out game (pseudo-labels from an algorithm, as RULES.md allows on the provided match): the base blend decoded, frames with a blended top probability of at least 0.6 taken as labels (96% of play; right on 96% of DFL frames, 90% of Metrica's), the base flow fine-tuned 750 steps on them, a quarter of its windows from that game and the rest from the training copies, at 0.3 of the learning rate (reports/claude/diag/flow_self.py); measured on the CPU | blend −0.000 [−0.002, +0.002], 3 / 10; with the flow at temperature 0.5 −0.001 [−0.004, +0.002], 4 / 10 | the game's own play, read through the pipeline's confident labels, might adapt the flow to it; but the labels it can trust are the ones it already had right, and the errors are in short possessions it is not confident about |

The flow rows above, from FULL_MASK on, were measured before the GPU fix. Measured again from the
same saved nets on the CPU, against the base on the CPU (blend at ½, ten games): FULL_MASK 0.5
+0.001 [−0.003, +0.005], the flow alone −0.009; ball seen → lost −0.003 [−0.007, +0.000], alone
−0.009; settled → turnovers −0.003 [−0.007, +0.000], alone −0.004; labels shown → none shown
−0.004 [−0.008, −0.001], alone −0.020. None is kept. The GPU's error moved single games by up to
0.04 either way while leaving the mean where it was; some of what had been read as two trainings
differing may be this.
| ball drift 1.0 m instead of 0.5, seven games | −0.027 [−0.035, −0.020] on held-out copies without drift, 0 / 7; −0.010 [−0.018, −0.003] on copies drifting 0.5 m, 3 / 7 | if a little drift helps, more might |
| losses in play as long as the provided ATD's gaps say (75–125 frames, not its mean gap of 431, which stoppages dominate), seven games, held-out copies fitted too | −0.010 [−0.018, −0.002], 3 / 7 | the copies' gaps read 1.3–10× the ATD's at every quantile |
| live play loses the ball on 30% of frames in training (20%), seven games and Metrica's three | DFL copies +0.010 [+0.002, +0.018]; Metrica −0.022 [−0.038, −0.007], 0 / 3 (20%: −0.006, −0.005, both intervals across zero) | the ATD likely loses it more often in play than the DFL copies do |

The two gate-training changes are the instructive ones: each found clearly more real
possessions (0.565 and 0.569 of them, from 0.542) and about as many false ones. The gate can be
pushed to call more turnovers, not to call them more accurately; the gains came from new
evidence (touches), not from rebalancing.

### Metrica's games as training data

Held out both ways — each DFL game against a model of the other DFL games, each Metrica game
against a model of the seven DFL games (and, when they are on, the other two Metrica games):

| training | DFL held out (7) | Metrica held out (3) |
|---|---|---|
| DFL, before fitted paths | 0.666 | 0.649 |
| DFL, fitted paths | 0.675 | 0.659 |
| DFL + Metrica, before fitted paths | 0.665 | 0.728 |
| DFL + Metrica, fitted paths | 0.663 | 0.742 |
| DFL, fitted paths, ball lost through stoppages (the default) | **0.750**\* | 0.729 |
| DFL + Metrica, the same | 0.735\* | 0.734 |

\* held-out copies that lose the ball through stoppages too; not comparable with the rows above.

With fitted paths on, the Metrica games cost the DFL games −0.012 [−0.019, −0.004] (worse on 7
of 7) and gained the Metrica games +0.083 [+0.069, +0.097] (better on 3 of 3). Once the DFL
copies lose the ball through stoppages, a model of the DFL games alone gets nearly all of that
(+0.070 [+0.053, +0.089] on the Metrica games), and the Metrica games add nothing more there
(+0.005 [−0.006, +0.016]) while still costing the DFL games −0.014 [−0.023, −0.005]. So they are
off (`config.METRICA_GAMES`). Two things explain the first split:

- **Metrica's ball path is smoothed** — 0.001 m of jitter, and the same touch detector finds
  32–36 touches a minute on it against DFL's 47–71, where the events show 40–68. Fitted paths
  read sharp changes of ball velocity, so a model fitted to both learns a blurred touch.
- **Metrica's tracking has no ball through stoppages** — from the frame after each one starts to
  its restart, on 98% of dead-ball frames, against 7–13% of live ones. The DFL games track the
  ball throughout, and their noisy copies lost it as often in a stoppage as in play. About half
  of the Metrica gain is dead-ball timing (+0.038 with the true dead ball given to both).

### Smart Tagging

No DFL game has Smart Tagging, so `tagging.py` writes look-alike tags for them from their labels
and clean tracking — each code's rule follows its name, thinned to the provided file's rate per
code, then corrupted per copy (edges moved, teams swapped, tags dropped and invented) — and the
tags become features and a tags expert. Read always, they were a bet: on four held-out games
they gained +0.014 (90% interval +0.002 to +0.026) with tags like the look-alikes, and lost when
the held-out tags misled (second column below). Training on harder corruption, or keeping the
tags inside their own expert, made it no safer and removed the gain.

So the provided game's tags are checked before any training (`tagging.check`,
`config.TAGS_CHECK`), against nothing but its own tracking: how often the team a tag says has
the ball is the team nearest the ball. That needs no label of any game. The look-alike tags of
the training games, on copies corrupted like the provided ATD, say what good tags score —

- **per code group** (attacking phases, defending phases), the worst agreement of any
  look-alike set is the *floor* the game's tags must reach;
- **a group that agrees with its teams reversed** is read reversed: a wrong club-name match, or
  a code whose Team means the other side;
- **tags running late or early** are moved back: the offset is where the game's agreement curve
  (agreement against every shift of the tags) best lines up with the look-alikes' — unbiased,
  and within 20 frames on 80% of look-alike sets moved by 0 to 4 s; under 1 s it is left alone;
- **a group must also beat chance** — random teams on the same tags, put through the same
  steps — to be judged at all;
- **set pieces are read for their timing, not their team**, so they are never judged (below);
- **otherwise the tags are not read**, and the model is the one without them.

On four held-out games, the same tags models, the check deciding per held-out game:

| held-out tags | always read | checked |
|---|---|---|
| like the look-alikes | +0.014 | +0.014 |
| none | −0.011 | 0.000 |
| random teams | −0.022 | 0.000 (refused) |
| every edge 1 s early | | +0.011 |
| every edge 1 s late | | +0.004 (under the 1 s threshold, left) |
| every edge 2 s late | −0.021 | +0.014 (moved back) |
| every edge 6 s late | | +0.013 (moved back) |
| Defending codes meaning the opposite | −0.014 | +0.014 (flipped) |
| every team reversed | −0.077 | +0.014 (flipped) |
| every team reversed and 2 s late | | +0.014 |

On all seven folds the decisions were the same kind: look-alike tags read, random teams
refused, reversed groups flipped, late tags moved back.

Set pieces were a third judged group until the provided game's failed it. Only 33 of its 59 name
a team — every throw-in is `N/A` ([DATA.md](../DATA.md)) — and a group that small judges
poorly: under the refreshed profile its attacking and defending tags cleared their floors easily
(0.657 against 0.594, 0.700 against 0.595), its set pieces read 0.599 against 0.605, and one
failing group refuses every tag. Under the earlier noise the floor was 0.541 and the same tags
passed; and on one held-out DFL game a look-alike set group, teams correct, was flipped. So
`config.TAGS_SET_TEAMS` is off: each set piece is read as "a restart here", like the throw-ins.
Seven held-out games, look-alike tags, held-out copies drifting:

| set pieces | spellF1 | against no tags |
|---|---|---|
| no tags at all | 0.703 | |
| dropped | 0.728 | +0.025 [+0.016, +0.034], 5 / 7 |
| **read without their team** | **0.733** | **+0.030 [+0.022, +0.038], 7 / 7** |
| read with their team, judged | 0.730 | +0.027 [+0.018, +0.036], 7 / 7 |

Their teams add nothing (−0.003 [−0.007, +0.001]); their timing adds a little over dropping
them (+0.005 [+0.000, +0.010]). What it cannot catch is tags that agree
with the tracking yet mean something other than the look-alikes do; and the gain itself is the
look-alikes' — the real tags' worth is whatever the final score says.

### Where the remaining error is

From the DFL labels alone (no model): merging every possession under 10 frames still scores
0.96, and moving every boundary by a random 5 frames (0.2 s) scores 0.93; at 10 frames it is 0.85.

From the held-out folds, copying the truth into one kind of possession at a time (upper bounds
for choosing what to build, never scores), before the touch work:

| copy the truth into | spellF1 gain |
|---|---|
| possessions that start when the other team loses the ball (78% of possessions) | +0.35 |
| possessions shorter than 2 s (32%) | +0.21 |
| every frame where the tracking has lost the ball | +0.07 |
| possessions after a dead ball (22%) | +0.03 |
| the ±12 frames around every true change | +0.00 |

So the points are in open-play turnovers, and mostly short ones. On seven games with the touch
expert, turnover possessions under 1 s are found 11% of the time, 1–2 s 21%, 2–5 s 46%, over 5 s
82%. The gate already puts the right team on top for most frames of 42% of the short ones, but
its probability there averages 0.40–0.46: the evidence, not the decoder, is what limits them.
Stretches where the gate flips to the other team are mostly fragments of longer true possessions
(median 7 frames of about 100), which is why a turnover checker placed on top of the decoder
(tried) made things worse — with the true edges those same stretches would be worth +0.13.

## Compliance with RULES.md

- **No hand labels by us.** Every DFL label comes from DFL's tracking: possession and ball status
  from its per-frame fields, touches from its clean ball track. The Metrica sample games' labels
  and touches come from Metrica's event data by the fixed rule above (`metrica.event_labels`).
- **Human data about other games, documented.** Metrica's event data of its three sample games is
  human-collected and trains the model, through the labels above; RULES.md allows data of this
  kind about other games, in any amount. Its repository states no licence: it asks users to "be
  responsible with the use of this data" and to acknowledge the source in anything public, which
  the technical document should do. DFL's event data for its games (CC BY 4.0, same dataset) was
  used once, to check the touch targets against it (above); nothing trains on it.
- **No ground truth of the provided game.** `guard.check_input` refuses `evaluation/gt/` and any
  `*_possession_gt*` file on every read path; a test checks it.
- **No human data about the provided game.** Training uses only the seven DFL matches and
  Metrica's three anonymised sample games, none of them the provided game. Smart
  Tagging is Metrica's automatic tagging of the provided video, an input like the ATD, not a
  label. `train` reads it and the ATD to decide whether the model reads the tags; the decision
  uses no label of any game, only how well the tags agree with the tracking, against floors set
  on DFL.
- **No frame-specific constants.** The only frame numbers used are the published half blocks
  from `games.py`, which decide where X goes.
- **Exact regeneration.** Fixed seeds everywhere, no randomised early stopping, deterministic
  decoding. `predict --check` regenerates the file from scratch and compares sha256.
  Freeze package versions before the final run (`pip freeze > requirements-lock.txt`).
- **No data in the repository.** `cache/`, `models/` and `reports/` are gitignored.
- **No hosted model sees the data.** Everything runs locally. `profile`, `train` and `predict`
  are run by the team, and their output stays on the team's machine.
- **The public window is a check, not a tuning set.** Every version above was chosen on held-out
  DFL games. Score the final file against the public five minutes once, to confirm it is aligned
  and well-formed.

### What may be committed

The code, `requirements-lock.txt` and `models/moe.pkl` (`.gitignore` skips `models/`, so
`git add -f` it). With the flow model's weights in it, the model file is what regenerates the
submission byte for byte; a GPU retrain need not. It holds trees fitted to the DFL and Metrica sample games, the flow model's weights and feature
scaling (means and spreads over the DFL games' noisy copies), the noise profile, which is a
handful of aggregate rates (missing fraction, mean gap length), and the Smart Tagging decision —
per code group a verdict and an agreement rate, and the offset. Never commit `cache/` (it holds features of the provided
game, and the downloaded DFL and Metrica files) or a submission CSV other than the one you submit.

## Known limits

- **Label conventions.** The model learns DFL's notion of "dead" and "owning", and on the Metrica
  games our events rule's — not the procedure behind the provided game's ground truth, which is
  not published. In one DFL match 44% of frames are dead ball and stoppages are long (median 23 s);
  the events rule marks 27–36% of the Metrica games dead. If the ground truth disagrees with both
  — how soon a throw-in counts as live — no noise model removes that gap.
- **Noise realism.** Identity switches and fragmented slots are not simulated (no feature can see
  them). The ATD's smoothing is matched by its accelerations, not its filter: a different filter
  with the same accelerations would blur touches a little differently. How often the ATD loses the ball through a
  stoppage, and how soon, is an assumption (0.9, about a second in from each end) — measuring it
  takes labels; and on DFL copies it leaves live play fewer losses than the ATD likely has.
  Training with more of them helps on simulated losses and costs on Metrica's real ones (Noise
  injection), so how the ATD loses the ball in play matters as much as how often, and only real
  tracking can test it.
- **Ten games** is a small training set. A held-out gap smaller than the spread across games is
  not a result.
- **Smart Tagging's meaning.** The check catches open-play tags that are missing, random,
  reversed or out of step. It cannot catch tags that agree with the tracking yet mean something
  other than the look-alikes do; only the final score can show that. Set pieces are read for
  their timing alone, and nothing checks that timing: the provided game's set pieces agree with
  its tracking less well than any look-alike set does, and whether that is their teams or their
  extent cannot be told without labels.
- **Not used:** the homography, the video and the broadcast tracking. Spotting the ball where the tracking lost it is worth at most +0.07
  even if perfect, so the video is not where the points are.
- **The flow model's training is not reproducible bit for bit** on a GPU, so a retrained model
  file can give a slightly different submission; the model file, weights included, is what
  reproduces it. Its share in the blend (½) was fixed before it was measured, and only ½ and 1
  were tried; the decoder's costs were tuned on the gate's out-of-fold output alone, not on the
  blend's.
