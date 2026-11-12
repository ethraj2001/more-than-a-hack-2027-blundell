"""Tests for the possession pipeline. Synthetic data only -- no match data, no network.

    python -m pytest tests/ -q

The end-to-end test trains a small model, so it takes a minute or two. Nothing it scores says
anything about the real match: synthetic games are far easier than real ones. It checks that the
parts connect, that the output is deterministic, and that the decoder beats the baseline's
smoothing on the same inputs.
"""

import os
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import (broadcast, carrier, config, decoder, dfl, features, guard,  # noqa
                        metrica, noise,
                        pipeline,
                        scoring, synthetic, tagging)
from possession.scoring import A, D, H, X, spell_f1  # noqa: E402
from possession.tracking import period_blocks, read_atd  # noqa: E402


def test_guard_refuses_ground_truth(tmp_path):
    public = os.path.join(config.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    with pytest.raises(guard.GroundTruthAccess):
        guard.check_input(public)
    with pytest.raises(guard.GroundTruthAccess):
        guard.check_input(str(tmp_path / "copied_possession_gt.csv"))
    harmless = str(tmp_path / "J03WPY_tracking.xml")
    assert guard.check_input(harmless) == harmless


def test_atd_round_trip(tmp_path):
    tracking, _ = synthetic.match("rt", n_half=200, seed=3)
    tracking.players[5:9, 2] = np.nan
    tracking.ball[10:20] = np.nan
    meta, raw = synthetic.write_atd(tracking, str(tmp_path))
    blocks = [(half, start, stop - 1) for start, stop, half in period_blocks(tracking.period)]
    back = read_atd(meta, raw, tracking.n_frames, tracking.pitch, blocks)
    assert np.array_equal(back.team, tracking.team)
    assert np.array_equal(back.period, tracking.period)
    assert np.array_equal(np.isnan(back.players), np.isnan(tracking.players))
    assert np.allclose(back.players, tracking.players, atol=1e-3, equal_nan=True)
    assert np.allclose(back.ball, tracking.ball, atol=1e-3, equal_nan=True)


def test_dfl_parse(tmp_path):
    meta = tmp_path / "meta.xml"
    meta.write_text('<?xml version="1.0"?><Root><Teams><Team TeamId="T-H" Role="home"/>'
                    '<Team TeamId="T-G" Role="guest"/></Teams></Root>')
    frames = lambda rows: "".join(rows)  # noqa: E731
    ball = ['<Frame N="{}" X="0" Y="0" BallPossession="{}" BallStatus="{}"/>'.format(n, p, s)
            for n, p, s in ((100, 1, 1), (101, 2, 1), (102, 2, 0))]
    tracking = tmp_path / "tracking.xml"
    tracking.write_text(
        '<?xml version="1.0"?><Positions><MetaData><PitchSize X="100" Y="60"/></MetaData>'
        '<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">{}</FrameSet>'
        '<FrameSet GameSection="firstHalf" TeamId="T-H" PersonId="P1">{}</FrameSet>'
        '<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="R1">{}</FrameSet>'
        '<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="B">{}</FrameSet>'
        '</Positions>'.format(
            frames(ball), '<Frame N="101" X="-50" Y="30"/>', '<Frame N="100" X="5" Y="5"/>',
            '<Frame N="500" X="1" Y="2" BallPossession="1" BallStatus="1"/>'))
    tr, labels = dfl.parse(str(meta), str(tracking))
    assert tr.pitch == (100.0, 60.0)
    assert labels[:3].tolist() == [H, A, D]
    assert labels[3:3 + dfl.PERIOD_GAP].tolist() == [X] * dfl.PERIOD_GAP
    assert labels[-1] == H and tr.period[-1] == 2
    assert tr.players[1, 0].tolist() == [0.0, 60.0]   # corner origin
    assert tr.team.tolist() == [0]                    # the referee is in neither team


def test_rolling_mean_matches_naive():
    rng = np.random.default_rng(0)
    values = rng.normal(size=300)
    values[rng.random(300) < 0.3] = np.nan
    for back, ahead in ((5, 5), (25, -1), (-1, 25)):
        fast = features.rolling_mean(values, back, ahead)
        for t in (0, 7, 150, 299):
            low = t + 1 if back < 0 else max(0, t - back)
            high = t if ahead < 0 else min(300, t + ahead + 1)
            window = values[low:high]
            expected = np.nanmean(window) if np.any(~np.isnan(window)) else np.nan
            assert np.isclose(fast[t], expected, equal_nan=True)


@pytest.mark.parametrize("optional", [False, True])
def test_features_are_team_symmetric(optional, monkeypatch):
    if optional:  # every feature family that is off by default, on
        for flag in ("REACH_FEATURES", "DYNAMICS_FEATURES", "GAP_FEATURES", "CONTEST_FEATURES",
                     "FIT_FEATURES"):
            monkeypatch.setattr(config, flag, True)
    tracking, _ = synthetic.match("sym", n_half=600, seed=1)
    tracking = noise.corrupt(tracking, dict(config.DEFAULT_NOISE), np.random.default_rng(1))
    feature_set = features.compute(tracking)
    names = feature_set.names()
    ab, ba = feature_set.matrix((0, 1)), feature_set.matrix((1, 0))
    for j, name in enumerate(names):
        prefix, key = name.split(".", 1)
        if prefix == "A":
            mirror = ba[:, names.index("B." + key)]
        elif prefix == "AmB":
            mirror = -ba[:, j]
        else:
            continue
        assert np.allclose(ab[:, j], mirror, equal_nan=True), name


def test_fitted_velocity_recovers_a_kick_through_jitter():
    rng = np.random.default_rng(3)
    n, kick = 200, 100
    velocity = np.where(np.arange(n)[:, None] < kick, [[0.2, 0.0]], [[0.2, 0.6]])  # m per frame
    path = np.cumsum(velocity, axis=0) + rng.normal(0, 0.05, (n, 2))
    path[rng.random(n) < 0.3] = np.nan
    before, after = features._fitted_velocity(path, 12)
    assert np.allclose(before[kick - 1], [0.2, 0.0], atol=0.05)
    assert np.allclose(after[kick + 1], [0.2, 0.6], atol=0.05)
    change = np.hypot(*(after - before).T)
    assert abs(int(np.nanargmax(change)) - kick) <= 2


def test_metrica_events_become_possession_labels():
    period = np.array([1] * 100 + [0] * 5 + [2] * 50, dtype=np.int8)
    row_of = {f: f - 1 for f in range(1, 101)}
    row_of.update({f: f + 4 for f in range(101, 151)})
    events = [(0, "SET PIECE", "KICK OFF", 1, 1), (0, "PASS", "", 1, 10),
              (0, "BALL LOST", "INTERCEPTION", 20, 30), (1, "RECOVERY", "INTERCEPTION", 31, 31),
              (1, "CHALLENGE", "GROUND-WON", 40, 40),   # not an on-ball event: no change
              (1, "BALL OUT", "", 45, 60), (0, "SET PIECE", "THROW IN", 80, 80),
              (0, "PASS", "", 80, 90), (1, "SET PIECE", "KICK OFF", 101, 101)]
    labels = metrica.event_labels(events, row_of, period)
    assert (labels[0:30] == H).all() and (labels[30:59] == A).all()  # rows are frames - 1
    assert (labels[59:79] == D).all() and (labels[79:100] == H).all()
    assert (labels[100:105] == X).all() and (labels[105:] == A).all()
    rows, teams = metrica.event_touches(events, row_of)
    assert 9 in rows.tolist() and teams[rows.tolist().index(30)] == 1  # a reception; the recovery


def test_metrica_csv_reads_both_teams_in_metres(tmp_path):
    header = ",,,Home,,Home,,,\n,,,1,,2,,,\nPeriod,Frame,Time [s],Player1,,Player2,,Ball,\n"
    for side, x in (("Home", 0.1), ("Away", 0.9)):
        rows = "".join("1,{},{},{},0.5,{},0.25,0.5,0.5\n".format(f, f * 0.04, x, x)
                       for f in (1, 2, 3))
        (tmp_path / side).write_text(header.replace("Home", side) + rows)
    players, team, ball, period, frame = metrica.parse_csv(str(tmp_path / "Home"),
                                                           str(tmp_path / "Away"))
    assert players.shape == (3, 4, 2) and team.tolist() == [0, 0, 1, 1]
    assert np.allclose(players[0, 0], [10.5, 34.0]) and np.allclose(players[0, 3], [94.5, 17.0])
    assert np.allclose(ball[0], [52.5, 34.0]) and frame.tolist() == [1, 2, 3]


def test_ball_losses_top_up_a_ball_already_lost():
    tracking, _ = synthetic.match("lost", n_half=6000, seed=4)
    rng = np.random.default_rng(0)
    ball = tracking.ball.copy()
    ball[rng.random(len(ball)) < 0.3] = np.nan
    profile = dict(config.DEFAULT_NOISE)
    noisy = noise.corrupt(tracking.replace(ball=ball), profile, np.random.default_rng(1))
    play = tracking.period > 0
    assert abs(np.isnan(noisy.ball[play, 0]).mean() - profile["ball_missing_frac"]) < 0.05


def test_stoppages_lose_the_ball():
    tracking, labels = synthetic.match("stops", n_half=6000, seed=5)
    dead = labels == D
    assert dead.any()
    profile = dict(config.DEFAULT_NOISE)
    plain = noise.corrupt(tracking, profile, np.random.default_rng(1))
    saved = config.DEAD_BALL_LOSS
    try:
        config.DEAD_BALL_LOSS = 0.0
        off = noise.corrupt(tracking, profile, np.random.default_rng(1), dead=dead)
        config.DEAD_BALL_LOSS = 1.0
        on = noise.corrupt(tracking, profile, np.random.default_rng(1), dead=dead)
    finally:
        config.DEAD_BALL_LOSS = saved
    assert np.array_equal(plain.ball, off.ball, equal_nan=True)  # off: the copy is unchanged
    missing = np.isnan(on.ball[:, 0])
    live = (labels == H) | (labels == A)
    assert missing[dead].mean() > 0.8 > missing[live].mean()


def test_live_play_loses_the_ball_at_its_own_rate():
    tracking, labels = synthetic.match("live", n_half=6000, seed=6)
    dead = labels == D
    live = (labels == H) | (labels == A)
    saved = config.DEAD_BALL_LOSS, config.LIVE_BALL_LOSS
    try:
        config.DEAD_BALL_LOSS, config.LIVE_BALL_LOSS = 1.0, 0.25
        noisy = noise.corrupt(tracking, dict(config.DEFAULT_NOISE), np.random.default_rng(2),
                              dead=dead)
    finally:
        config.DEAD_BALL_LOSS, config.LIVE_BALL_LOSS = saved
    missing = np.isnan(noisy.ball[:, 0])
    assert abs(missing[live].mean() - 0.25) < 0.05
    assert missing[dead].mean() > 0.8


def test_noise_matches_profile():
    tracking, _ = synthetic.match("noise", n_half=6000, seed=2)
    profile = dict(config.DEFAULT_NOISE)
    noisy = noise.corrupt(tracking, profile, np.random.default_rng(0))
    measured = noise.measure(noisy)
    assert abs(measured["ball_missing_frac"] - profile["ball_missing_frac"]) < 0.06
    assert abs(measured["players_visible_mean"] - profile["players_visible_mean"]) < 0.5


def test_path_smoothing_matches_a_smoothed_tracker():
    tracking, labels = synthetic.match("smooth", n_half=6000, seed=4)
    rng = np.random.default_rng(0)
    ball = noise.smooth_paths(tracking.ball[:, None, :], tracking.period, tracking.fps,
                              ("gauss", 5.0))[:, 0]
    target = tracking.replace(ball=ball + rng.normal(0.0, 0.002, ball.shape).astype(np.float32),
                              players=noise.smooth_paths(tracking.players, tracking.period,
                                                         tracking.fps, ("lowpass", 1.0)))
    profile = dict(config.DEFAULT_NOISE)
    profile.update(noise.measure(target), ball_outlier_rate=0.02, ball_outlier_rate_play=0.01,
                   ball_outlier_in_track=0.0, ball_missing_frac=0.4)
    saved = config.PATH_SMOOTHING
    try:
        config.PATH_SMOOTHING = False
        sharp = noise.measure(noise.corrupt(tracking, profile, np.random.default_rng(1)))
        config.PATH_SMOOTHING = True
        smooth = noise.measure(noise.corrupt(tracking, profile, np.random.default_rng(1)))
    finally:
        config.PATH_SMOOTHING = saved
    for name in ("ball", "player"):
        ratio = np.asarray(smooth[name + "_accel_p99"]) / np.asarray(profile[name + "_accel_p99"])
        assert np.all(np.abs(np.log(ratio)) < np.log(1.3)), (name, ratio)
        assert abs(smooth[name + "_straight_frac"] - profile[name + "_straight_frac"]) < 0.1
    assert sharp["ball_accel_p99"][0] > 5 * profile["ball_accel_p99"][0]
    # false detections: as often as measured, but outside the tracks once those are smooth
    assert abs(smooth["ball_outlier_rate_play"] - profile["ball_outlier_rate_play"]) < 0.005
    assert smooth["ball_outlier_in_track"] < 0.2 < 0.8 < sharp["ball_outlier_in_track"]
    assert noise.calibrate_paths(dict(config.DEFAULT_NOISE), tracking) is None
    # all of them inside the tracks, as measured: only if the tail has room for them
    in_track = dict(profile, ball_outlier_in_track=1.0)
    spiky = dict(in_track, ball_accel_tail=list(profile["ball_accel_tail"][:-1]) + [5000.0])
    try:
        config.PATH_SMOOTHING = True
        tails = [noise.measure(noise.corrupt(tracking, p, np.random.default_rng(1)))["ball_accel_tail"]
                 for p in (in_track, spiky)]
    finally:
        config.PATH_SMOOTHING = saved
    assert tails[0][-1] < 2 * profile["ball_accel_tail"][-1] < tails[1][-1] / 10


def test_ball_lead_reads_a_lagging_ball():
    tracking, _ = synthetic.match("lead", n_half=6000, seed=4)
    _, on_time = noise.ball_lead(tracking)
    _, late = noise.ball_lead(tracking.replace(ball=np.roll(tracking.ball, 6, axis=0)))
    assert abs(on_time) < 1 and abs(late - (on_time - 6)) < 1
    assert "ball_lead_frames" in noise.measure(tracking)


def test_ball_drift_moves_only_the_ball(monkeypatch):
    tracking, labels = synthetic.match("drift", n_half=6000, seed=4)
    profile = dict(config.DEFAULT_NOISE)
    monkeypatch.setattr(config, "BALL_DRIFT_M", 0.0)
    still = noise.corrupt(tracking, profile, np.random.default_rng(1), dead=labels == D)
    monkeypatch.setattr(config, "BALL_DRIFT_M", 0.5)
    moved = noise.corrupt(tracking, profile, np.random.default_rng(1), dead=labels == D)
    assert np.array_equal(still.players, moved.players, equal_nan=True)
    assert np.array_equal(np.isnan(still.ball), np.isnan(moved.ball))
    offset = (moved.ball - still.ball)[~np.isnan(still.ball[:, 0])]
    assert np.all(np.abs(offset.std(axis=0) - 0.5) < 0.1)


def test_ball_losses_take_the_length_the_gaps_say():
    tracking, labels = synthetic.match("gaps", n_half=6000, seed=7)
    dead = labels == D
    profile = dict(config.DEFAULT_NOISE, ball_missing_frac=0.4)
    for mean in (40.0, 300.0):
        target = noise.measure(noise.corrupt(tracking, dict(profile, ball_gap_mean=mean),
                                             np.random.default_rng(2), dead=dead))
        fitted = noise.calibrate_losses(dict(profile, ball_gap_quantiles=target["ball_gap_quantiles"],
                                             ball_run_quantiles=target["ball_run_quantiles"]),
                                        tracking, dead)
        assert mean / 2 < fitted < mean * 2, (mean, fitted)
    assert noise.calibrate_losses(profile, tracking, dead) == profile["ball_gap_mean"]


def test_path_shape_tells_joined_lines_from_jitter():
    n = 5000
    period = np.ones(n, dtype=np.int8)
    rng = np.random.default_rng(0)
    knots = np.arange(0, n + 25, 25)
    joined = np.stack([np.interp(np.arange(n), knots, rng.uniform(0, 100, len(knots)))
                       for _ in range(2)], axis=1)[:, None, :]
    jittered = joined + rng.normal(0, 0.05, joined.shape)
    straight, accel = noise.path_shape(joined, period, 25)
    assert straight > 0.9 and len(accel) == len(noise.SHAPE_LAGS)
    assert noise.path_shape(jittered, period, 25)[0] < 0.1
    assert noise.path_shape(joined[:100], period[:100], 25) == (None, None)


def test_decoder_respects_minimum_durations():
    rng = np.random.default_rng(0)
    n = 3000
    truth = np.repeat(np.array([D, H, A, H, D, A], dtype=np.int8), n // 6)
    period = np.ones(n, dtype=np.int8)
    probs = rng.dirichlet([1, 1, 1], size=n)
    probs[np.arange(n), truth] += 0.6
    probs /= probs.sum(axis=1, keepdims=True)
    log_pi = decoder.transition_prior([truth], [period])
    params = dict(config.DECODER_DEFAULT, min_possession=40, min_dead=40)
    out = decoder.decode(probs, np.full(n, 0.01), period, params, log_pi)
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([-1], out, [-1]))) != 0))
    assert runs.min() >= 40
    assert spell_f1(truth, out) > 0.9
    assert spell_f1(truth, np.argmax(probs, axis=1).astype(np.int8)) < 0.5


def _chain_viterbi(probs, boundary, params, log_pi):
    """The decoder as first written, over every chain state: the reference the fast one must
    reproduce exactly."""
    n = len(probs)
    lengths = [max(2, int(params["min_possession"]))] * 2 + [max(2, int(params["min_dead"]))]
    first = np.cumsum([0] + lengths[:-1])
    last = first + np.array(lengths) - 1
    label_of = np.repeat(np.arange(3), lengths)
    chained = np.setdiff1d(np.arange(int(sum(lengths))), first)
    last_pos = np.searchsorted(chained, last)
    emit = params["alpha"] * np.log(np.clip(probs, decoder.EPS, 1.0))
    b = np.clip(boundary, decoder.EPS, 1 - decoder.EPS)
    stay = params["beta"] * np.log1p(-np.clip(b.sum(axis=1), decoder.EPS, 1 - decoder.EPS))
    move = params["beta"] * np.log(b) - params["switch_penalty"]
    delta = emit[0, label_of].copy()
    back = np.empty((n, len(label_of)), dtype=np.int64)
    back[0] = np.arange(len(label_of))
    new = np.empty(len(label_of))
    for t in range(1, n):
        step, step_from = delta[chained - 1], chained - 1
        better = delta[last] > step[last_pos]
        step[last_pos] = np.where(better, delta[last], step[last_pos])
        step_from[last_pos] = np.where(better, last, step_from[last_pos])
        new[chained] = step + stay[t]
        back[t, chained] = step_from
        scores = delta[last][:, None] + log_pi + move[t]
        source = np.argmax(scores, axis=0)
        new[first] = scores[source, np.arange(3)]
        back[t, first] = last[source]
        delta = new + emit[t, label_of]
    state, path = int(np.argmax(delta)), np.empty(n, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        path[t], state = state, back[t, state]
    return np.asarray(decoder.LABELS, dtype=np.int8)[label_of[path]]


def test_fast_decoder_matches_the_chain_decoder():
    rng = np.random.default_rng(7)
    truth = np.repeat(rng.integers(0, 3, size=40), rng.integers(3, 60, size=40)).astype(np.int8)
    probs = rng.dirichlet([1, 1, 1], size=len(truth))
    probs[np.arange(len(truth)), truth] += 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    boundary = rng.dirichlet([1, 1, 1, 6], size=len(truth))[:, :3]
    log_pi = decoder.transition_prior([truth], [np.ones(len(truth), dtype=np.int8)])
    for params in (dict(config.DECODER_DEFAULT),
                   dict(config.DECODER_DEFAULT, alpha=0.05, min_possession=5, min_dead=30),
                   dict(config.DECODER_DEFAULT, alpha=0.4, beta=0.5, switch_penalty=0.0)):
        for n in (len(truth), 25, 3):  # including inputs shorter than a chain
            assert np.array_equal(decoder.viterbi(probs[:n], boundary[:n], params, log_pi),
                                  _chain_viterbi(probs[:n], boundary[:n], params, log_pi))


def test_end_to_end_is_deterministic_and_beats_baseline(tmp_path):
    games = [synthetic.match("e2e{}".format(i), n_half=3000, seed=10 + i) for i in range(4)]
    profile = dict(config.DEFAULT_NOISE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games[:3], profile, quick=True)
        test, labels = games[3]
        noisy = noise.corrupt(test, profile, pipeline.rng_for(test.name, pipeline.TEST_COPY))
        first = pipeline.predict(bundle, noisy)
        saved = pipeline.load(pipeline.save(bundle, str(tmp_path / "moe.pkl")))
        second = pipeline.predict(saved, noisy)
    assert np.array_equal(first, second)      # deterministic, and survives the model file
    assert np.all(first[test.period == 0] == X)
    assert spell_f1(labels, first) > spell_f1(labels, pipeline.baseline(noisy)) + 0.2


def test_boundary_targets_name_the_destination():
    labels = np.array([H] * 10 + [A] * 10 + [D] * 10, dtype=np.int8)
    period = np.ones(30, dtype=np.int8)
    target = pipeline.boundary_targets(labels, period, tolerance=2)
    assert target[8:13].tolist() == [1 + A] * 5
    assert target[18:23].tolist() == [1 + D] * 5
    assert target[:8].sum() == 0 and target[13:18].sum() == 0 and target[23:].sum() == 0


def test_short_targets_mark_only_short_possessions():
    labels = np.array([H] * 60 + [A] * 20 + [D] * 5 + [H] * 10 + [X] * 3 + [H] * 4, dtype=np.int8)
    period = np.array([1] * 98 + [2] * 4, dtype=np.int8)
    period[95:98] = 0
    target = pipeline.short_targets(labels, period, longest=50)
    assert target[:60].sum() == 0                              # 60 frames is not short
    assert target[60:80].tolist() == [1 + A] * 20
    assert target[80:85].sum() == 0                            # dead balls never are
    assert target[85:95].tolist() == [1 + H] * 10
    assert target[95:98].sum() == 0 and target[98:].tolist() == [1 + H] * 4


def test_risk_targets_mark_quick_turnovers_not_restarts():
    labels = np.array([H] * 100 + [A] * 20 + [H] * 100 + [D] * 10 + [A] * 10 + [H] * 60,
                      dtype=np.int8)
    period = np.ones(len(labels), dtype=np.int8)
    target = pipeline.risk_targets(labels, period, half=30, changes=2)
    # turnovers at 100 and 120 both lie within 30 frames of frames 90..130
    assert np.flatnonzero(target[:220]).tolist() == list(range(90, 131))
    # H -> D -> A is a restart, so only the A -> H turnover at 240 is left near the end
    assert target[220:].sum() == 0
    period[:110] = 1
    period[110:] = 2  # the two turnovers now sit in different periods
    assert pipeline.risk_targets(labels, period, half=30, changes=2)[:220].sum() == 0


def test_touch_targets_find_the_player_who_turned_the_ball():
    from possession.tracking import Tracking
    n = 60
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 1] = 30.0
    ball[:, 0] = 50.0 - 0.5 * np.abs(np.arange(n) - 20)          # in to x = 50 at frame 20, back out
    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, 0] = (50.5, 30.0)                                  # team 1, where the ball turns
    players[:, 1] = (20.0, 30.0)                                  # team 0, far away
    tracking = Tracking("turn", players, np.array([1, 0], dtype=np.int8), ball,
                        np.ones(n, dtype=np.int8), (105.0, 68.0))
    frames, teams = pipeline.touch_frames(tracking)
    assert frames.tolist() == [20] and teams.tolist() == [1]
    target = pipeline.touch_targets(tracking, tolerance=2)
    assert target[18:23].tolist() == [1 + 1] * 5 and target.sum() == 2 * 5


def test_touch_clocks_follow_the_team_that_touched():
    tracking, _ = synthetic.match("touch", n_half=600, seed=5)
    feature_set = features.compute(tracking)
    since = [feature_set.team[k]["since_touch.0"] for k in (0, 1)]
    play = tracking.period > 0
    assert np.all(since[0][play] >= 0) and np.all(since[0][play] <= config.CAP_FRAMES)
    assert np.all(np.isnan(since[0][~play]))
    assert min(np.sum(s[play] == 0) for s in since) > 0         # both teams touch the ball


def test_carrier_reads_who_is_on_the_ball():
    profile = dict(config.DEFAULT_NOISE)
    (train_game, _), (test_game, labels) = [synthetic.match("car{}".format(i), n_half=3000,
                                                            seed=30 + i) for i in range(2)]
    noisy_train = noise.corrupt(train_game, profile, np.random.default_rng(0))
    noisy_test = noise.corrupt(test_game, profile, np.random.default_rng(1))
    frames = np.flatnonzero(noisy_test.period > 0)
    _, slots, rows = carrier.candidates(noisy_test, frames)
    targets = carrier._targets(noisy_test, test_game.ball, rows, slots)
    assert np.bincount(rows[targets == 1]).max() <= 1          # at most one carrier a frame
    with threadpool_limits(limits=1):
        model = carrier.Carrier(dict(config.CARRIER_HGB, max_iter=40)).fit(
            [(noisy_train, train_game.ball)], stride=5)
        team, _ = model.series(noisy_test)
    lost = (noisy_test.period > 0) & np.isnan(noisy_test.ball[:, 0]) & (labels < D)
    leader = (team[1]["car_sum"] > team[0]["car_sum"]).astype(np.int8)
    assert np.mean(leader[lost] == labels[lost]) > 0.7


def test_jitter_estimate_recovers_added_noise():
    tracking, _ = synthetic.match("jit", n_half=3000, seed=4)
    profile = dict(config.DEFAULT_NOISE, player_jitter_m=0.15, ball_missing_frac=0.0,
                   players_visible_mean=22)
    measured = noise.measure(noise.corrupt(tracking, profile, np.random.default_rng(0)))
    assert abs(measured["player_jitter_m"] - 0.15) < 0.02
    assert measured["team_overfull_frac"] == 0.0


def test_paired_bootstrap_sees_a_real_difference():
    truth = np.repeat(np.array([H, A] * 30, dtype=np.int8), 50)
    noisy = truth.copy()
    noisy[::7] = A          # flicker: many spurious short possessions
    point, low, high = scoring.paired_spell_bootstrap([(truth, noisy, truth)], n_draws=200)
    assert point > 0 and low > 0
    point, low, high = scoring.paired_spell_bootstrap([(truth, truth, truth)], n_draws=50)
    assert point == low == high == 0


def test_row_percentiles_match_numpy_exactly():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(2000, 22)) * 30
    values[rng.random(values.shape) < 0.5] = np.nan
    values[:10] = np.nan
    values[10:20, 1:] = np.nan
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanpercentile(values, (10, 90), axis=1)
    assert np.array_equal(np.array(features._row_percentiles(values, (10, 90))), expected,
                          equal_nan=True)


def test_bank_cache_fits_each_bank_once(tmp_path, monkeypatch):
    fits = []

    def fit():
        fits.append(1)
        return {"bank": len(fits)}

    run, profile = pipeline.settings(quick=True), dict(config.DEFAULT_NOISE)
    banks = pipeline.BankCache(run, profile, folder=str(tmp_path))
    assert banks.get(["g1", "g2"], fit) == {"bank": 1}
    assert banks.get(["g1", "g2"], fit) == {"bank": 1} and len(fits) == 1
    assert banks.get(["g1", "g3"], fit) == {"bank": 2}
    monkeypatch.setattr(config, "GATE_TEMPORAL", not config.GATE_TEMPORAL)
    same = pipeline.BankCache(run, profile, folder=str(tmp_path))
    assert same.get(["g1", "g2"], fit) == {"bank": 1}                      # gate settings are not
    monkeypatch.setattr(config, "TRAIN_STRIDE", config.TRAIN_STRIDE + 1)
    changed = pipeline.BankCache(run, profile, folder=str(tmp_path))
    assert changed.get(["g1", "g2"], fit) == {"bank": 3}                   # config is in the key
    assert not [p for p in os.listdir(str(tmp_path)) if p.endswith(".part")]


def test_macro_f1_matches_the_scorer():
    rng = np.random.default_rng(9)
    truth = np.repeat(rng.integers(0, 3, size=50), 30).astype(np.int8)
    pred = truth.copy()
    pred[rng.random(len(pred)) < 0.2] = rng.integers(0, 3)
    assert np.isclose(scoring.macro_f1(truth, pred), scoring.pm.evaluate(truth, pred)["macro_f1"])


def test_refiner_trains_and_predicts(monkeypatch):
    monkeypatch.setattr(config, "GATE_PASSES", 2)
    games = [synthetic.match("rf{}".format(i), n_half=1500, seed=70 + i) for i in range(4)]
    profile = dict(config.DEFAULT_NOISE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games[:3], profile, quick=True)
        test, labels = games[3]
        noisy = noise.corrupt(test, profile, pipeline.rng_for(test.name, pipeline.TEST_COPY))
        first, second = pipeline.predict(bundle, noisy), pipeline.predict(bundle, noisy)
    assert bundle["refiner"] is not None and np.array_equal(first, second)
    assert spell_f1(labels, first) > spell_f1(labels, pipeline.baseline(noisy))


def test_risk_expert_trains_and_feeds_the_gate(monkeypatch):
    monkeypatch.setattr(config, "RISK_EXPERT", True)
    monkeypatch.setattr(config, "RISK_HALF", 50)
    games = [synthetic.match("rk{}".format(i), n_half=1500, seed=80 + i) for i in range(4)]
    profile = dict(config.DEFAULT_NOISE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games[:3], profile, quick=True)
        test, labels = games[3]
        noisy = noise.corrupt(test, profile, pipeline.rng_for(test.name, pipeline.TEST_COPY))
        first, second = pipeline.predict(bundle, noisy), pipeline.predict(bundle, noisy)
    assert "risk.F" in bundle["gate"].output_names
    swap = bundle["bank"].swap_index()
    column = bundle["bank"].outputs().index("risk.F")
    assert swap[column] == column  # the same in either team order
    assert np.array_equal(first, second)
    assert spell_f1(labels, first) > spell_f1(labels, pipeline.baseline(noisy))


def test_smart_tagging_teams_are_matched_by_club_name():
    events = [{"code": "Build Up", "start_frame": 10, "end_frame": 20, "tags": {"Team": "Barcelona"}},
              {"code": "BUILD UP", "start_frame": 30, "end_frame": 40,
               "tags": {"Team": "Atletic Club"}},
              {"code": "SET PIECES", "start_frame": 50, "end_frame": 60, "tags": {"Team": "N/A"}}]
    tags = tagging.read(events, "FC Barcelona", "Athletic Club")
    assert [(t[2], t[3]) for t in tags] == [("BUILD UP", 0), ("BUILD UP", 1), ("SET PIECES", None)]


def test_simulated_tags_follow_possession(monkeypatch):
    tracking, labels = synthetic.match("tags", n_half=6000, seed=11)
    tags = tagging.simulate(tracking, labels, np.random.default_rng(0))
    assert tags and all(0 <= s < e <= tracking.n_frames and t in (0, 1, None)
                        for s, e, code, t in tags)
    team, shared = tagging.series(tags, tracking.period)
    alive = np.isin(labels, (H, A))
    covered = [(team[k]["tag_att"] == 1) & alive for k in (0, 1)]
    agree = np.mean(np.concatenate([labels[covered[k]] == k for k in (0, 1)]))
    assert agree > 0.6                       # corrupted, but mostly right
    monkeypatch.setattr(config, "USE_TAGS", True)
    tracking.tags = tags
    feature_set = pipeline.compute_features(tracking)
    names = feature_set.names()
    ab, ba = feature_set.matrix((0, 1)), feature_set.matrix((1, 0))
    for j, name in enumerate(names):
        if ".tag_" in name and name.startswith("A."):
            assert np.allclose(ab[:, j], ba[:, names.index("B." + name[2:])], equal_nan=True)


def test_tags_train_and_predict_with_or_without_a_file(monkeypatch):
    monkeypatch.setattr(config, "USE_TAGS", True)
    games = [synthetic.match("tg{}".format(i), n_half=1500, seed=80 + i) for i in range(4)]
    profile = dict(config.DEFAULT_NOISE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games[:3], profile, quick=True)
        test, labels = games[3]
        noisy = pipeline.corrupt(test, labels, profile, pipeline.TEST_COPY)
        tagged = pipeline.predict(bundle, noisy)
        untagged = pipeline.predict(bundle, noisy.replace(tags=None))
    assert "tags" in [e.name for e in bundle["bank"].experts]
    assert spell_f1(labels, tagged) > spell_f1(labels, pipeline.baseline(noisy))
    assert len(untagged) == len(tagged)


def test_run_lengths_count_each_run():
    from possession.gate import run_lengths
    labels = np.array([H, H, H, A, D, D, H, H], dtype=np.int8)
    period = np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int8)
    assert run_lengths(labels, period).tolist() == [3, 3, 3, 1, 2, 2, 2, 2]



def _label_holders(labels, name, lead=50, wrong=0.15):
    """A stand-in for tagging.holders() with a known quality: the labelled team, wrong on a
    share of frames, running lead frames ahead -- synthetic games are too short and too even
    for the nearest player to say much about who has the ball."""
    rng = pipeline.rng_for(name, 7)
    out = np.where(np.isin(labels, (H, A)), labels, -1).astype(np.int64)
    wrong = (out >= 0) & (rng.random(len(out)) < wrong)
    out[wrong] = 1 - out[wrong]
    return np.concatenate([out[lead:], np.full(lead, -1)])


def _tag_reference(games):
    return tagging.reference([(_label_holders(y, t.name), tagging.simulate(
        t, y, pipeline.rng_for(t.name, k, seed=1), hard=False)) for t, y in games
        for k in range(4)])


def test_tag_check_flips_reversed_teams_and_refuses_random_ones():
    games = [synthetic.match("ck{}".format(i), n_half=20000, seed=90 + i) for i in range(4)]
    ref = _tag_reference(games[:3])
    test, labels = games[3]
    holder = _label_holders(labels, test.name)
    tags = tagging.simulate(test, labels, np.random.default_rng(5), hard=False)
    good = tagging.check(tags, holder, ref)
    assert good["use"] and good["groups"] == {"att": "keep", "def": "keep", "set": "blind"}
    # set pieces are read without their team, so their teams, even reversed, decide nothing
    set_reversed = [(s, e, c, 1 - t if c == tagging.SET_PIECES and t is not None else t)
                    for s, e, c, t in tags]
    assert tagging.check(set_reversed, holder, ref)["groups"] == good["groups"]
    reversed_ = [(s, e, c, None if t is None else 1 - t) for s, e, c, t in tags]
    flipped = tagging.check(reversed_, holder, ref)
    assert flipped["use"] and flipped["groups"]["att"] == flipped["groups"]["def"] == "flip"
    open_play = tagging.ATTACKING + tagging.DEFENDING
    assert ([t for t in tagging.apply(reversed_, flipped, test.n_frames) if t[2] in open_play]
            == [t for t in tagging.apply(tags, good, test.n_frames) if t[2] in open_play])
    rng = np.random.default_rng(0)
    random_teams = [(s, e, c, None if t is None else int(rng.integers(2))) for s, e, c, t in tags]
    assert not tagging.check(random_teams, holder, ref)["use"]
    chance = dict(ref, floor={group: 0.5 for group in ref["floor"]})
    assert not tagging.check(tags, holder, chance)["use"]   # nothing to judge by: not read


def test_set_pieces_are_read_for_their_timing_alone(monkeypatch):
    tags = [(10, 20, tagging.SET_PIECES, 0), (30, 40, "BUILD UP", 1)]
    period = np.ones(50, dtype=np.int8)
    team, shared = tagging.series(tags, period)
    assert team[0]["tag_set"].sum() == 0 and shared["tag_unknown"][10:20].all()
    assert team[1]["tag_att"][30:40].all() and shared["tag_any"].sum() == 20
    monkeypatch.setattr(config, "TAGS_SET_TEAMS", True)
    team, shared = tagging.series(tags, period)
    assert team[0]["tag_set"][10:20].all() and shared["tag_unknown"].sum() == 0


def test_tag_check_finds_late_tags():
    games = [synthetic.match("of{}".format(i), n_half=20000, seed=95 + i) for i in range(4)]
    ref = _tag_reference(games[:3])
    test, labels = games[3]
    holder = _label_holders(labels, test.name)
    tags = tagging.simulate(test, labels, np.random.default_rng(6), hard=False)
    late = tagging.check(tagging.moved(tags, 100), holder, ref)
    assert late["use"] and abs(late["late"] - tagging.check(tags, holder, ref)["late"] - 100) <= 25


def test_checked_tags_train_predict_and_guard_the_tag_file(monkeypatch):
    # The plan is judged on games long enough to judge by; the model is trained on short ones.
    profile = dict(config.DEFAULT_NOISE)
    long = [synthetic.match("pl{}".format(i), n_half=30000, seed=60 + i) for i in range(4)]
    short = [synthetic.match("ps{}".format(i), n_half=3000, seed=64 + i) for i in range(3)]
    labels_of = {t.name: y for t, y in long}
    monkeypatch.setattr(tagging, "holders", lambda tracking: _label_holders(
        labels_of[tracking.name], tracking.name, wrong=0.05))
    test, labels = long[3]
    with pipeline.using_tags(True):
        noisy = pipeline.corrupt(test, labels, profile, pipeline.TEST_COPY)
    plan = pipeline.tag_plan(long[:3], profile, noisy)
    assert plan["use"] and pipeline.describe_tags(plan).startswith("read")
    with threadpool_limits(limits=1):
        bundle = pipeline.train(short, profile, quick=True, tags=plan)
        predicted = pipeline.predict(bundle, noisy)
        assert "tags" in [e.name for e in bundle["bank"].experts] and not config.USE_TAGS
        assert spell_f1(labels, predicted) > spell_f1(labels, pipeline.baseline(noisy))
        with pytest.raises(ValueError):
            pipeline.predict(bundle, noisy.replace(tags=None))
        rng = np.random.default_rng(1)
        other = [(s, e, c, None if t is None else int(rng.integers(2)))
                 for s, e, c, t in noisy.tags]
        with pytest.raises(ValueError):
            pipeline.predict(bundle, noisy.replace(tags=other))
    assert not pipeline.tag_plan(long[:3], profile, noisy.replace(tags=None))["use"]


def test_train_without_the_match_trains_without_tags(monkeypatch):
    from possession import __main__ as cli, tracking

    def missing(*args, **kwargs):
        raise IOError("match data not found")
    monkeypatch.setattr(tracking, "load_atd", missing)
    assert not cli.tag_plan_for([], dict(config.DEFAULT_NOISE))["use"]


def test_flow_swapped_columns_are_the_other_team_order():
    from possession import flow
    tracking, labels = synthetic.match("fs", n_half=2000, seed=31)
    feature_set = features.compute(tracking)
    index, sign = flow.swap_columns(feature_set.names())
    ab, ba = feature_set.matrix((0, 1)), feature_set.matrix((1, 0))
    assert np.array_equal(ab[:, index] * sign.astype(np.float32), ba, equal_nan=True)


def test_flow_model_blends_reproducibly_and_weight_zero_is_the_mixture(monkeypatch):
    from possession import flow
    monkeypatch.setattr(flow, "TRAIN_STEPS", 3)
    monkeypatch.setattr(flow, "BATCH", 4)
    monkeypatch.setattr(config, "FLOW_COPIES", 1)
    games = [synthetic.match("fl{}".format(i), n_half=2500, seed=70 + i) for i in range(4)]
    profile = dict(config.DEFAULT_NOISE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games[:3], profile, quick=True)
        mixture = pipeline.predict(bundle, games[3][0])
        bundle["flow"] = pipeline.train_flow(games[:3], profile, bundle["tags"], "cpu")
        p = pipeline.flow_probs(bundle["flow"], games[3][0])
        assert p.shape == (games[3][0].n_frames, 3) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
        blended = pipeline.predict(bundle, games[3][0])
        assert np.array_equal(blended, pipeline.predict(bundle, games[3][0]))
        assert set(np.unique(blended)) <= {H, A, D, X}
        bundle["flow"]["weight"] = 0.0
        assert np.array_equal(pipeline.predict(bundle, games[3][0]), mixture)
    sharp = pipeline.sharpen(p, 0.5)
    assert np.allclose(sharp.sum(axis=1), 1) and np.array_equal(sharp.argmax(axis=1), p.argmax(axis=1))
    assert (sharp.max(axis=1) >= p.max(axis=1) - 1e-12).all() and pipeline.sharpen(p, 1.0) is p


def _lose_ball_in_runs(tracking, rng, seen_run=200, lost_run=130):
    ball, k, seen = tracking.ball.copy(), 0, True
    while k < len(ball):
        run = int(rng.geometric(1.0 / (seen_run if seen else lost_run)))
        if not seen:
            ball[k:k + run] = np.nan
        k, seen = k + run, not seen
    return tracking.replace(ball=ball)


def test_broadcast_aligns_to_tactical_and_fills_the_ball():
    tracking, _ = synthetic.match("bc", n_half=6000, seed=8)
    rng = np.random.default_rng(8)
    view, truth = synthetic.broadcast_view(tracking, rng)
    tactical = _lose_ball_in_runs(tracking, rng)
    alignment = broadcast.align(tactical, view)
    true_offsets = set((truth[truth >= 0] - np.flatnonzero(truth >= 0)).tolist())
    assert {o for o, _ in alignment["offsets"]} == true_offsets  # both halves, nothing else
    assert alignment["flip"] == (True, True)
    mapped = np.flatnonzero(alignment["t2b"] >= 0)
    assert np.mean(truth[alignment["t2b"][mapped]] == mapped) > 0.99  # to the frame
    assert alignment["live"][truth < 0].mean() < 0.2  # replays mostly dropped
    filled, which = broadcast.fill(tactical, alignment)
    assert which.sum() > 0 and np.isnan(filled.ball[:, 0]).sum() < np.isnan(tactical.ball[:, 0]).sum()
    assert np.median(np.hypot(*(filled.ball[which] - tracking.ball[which]).T)) < 1.0
    report = broadcast.describe(tactical, alignment)
    assert report["ball_missing_graded_after"] < report["ball_missing_graded_before"]
    assert broadcast.align(tactical, view)["t2b"].tolist() == alignment["t2b"].tolist()


def test_broadcast_fill_without_the_files_keeps_the_tactical_ball(tmp_path):
    tracking, _ = synthetic.match("nobc", n_half=200, seed=1)
    assert broadcast.fused(tracking, data_dir=str(tmp_path)) is tracking
