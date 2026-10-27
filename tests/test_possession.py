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

from possession import (carrier, config, decoder, dfl, features, guard, noise, pipeline,  # noqa
                        scoring, synthetic)
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


def test_features_are_team_symmetric():
    tracking, _ = synthetic.match("sym", n_half=600, seed=1)
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


def test_noise_matches_profile():
    tracking, _ = synthetic.match("noise", n_half=6000, seed=2)
    profile = dict(config.DEFAULT_NOISE)
    noisy = noise.corrupt(tracking, profile, np.random.default_rng(0))
    measured = noise.measure(noisy)
    assert abs(measured["ball_missing_frac"] - profile["ball_missing_frac"]) < 0.06
    assert abs(measured["players_visible_mean"] - profile["players_visible_mean"]) < 0.5


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


def test_end_to_end_is_deterministic_and_beats_baseline():
    games = [synthetic.match("e2e{}".format(i), n_half=3000, seed=10 + i) for i in range(4)]
    profile = dict(config.DEFAULT_NOISE)
    with threadpool_limits(limits=1):
        bundle = pipeline.train(games[:3], profile, quick=True)
        test, labels = games[3]
        noisy = noise.corrupt(test, profile, pipeline.rng_for(test.name, pipeline.TEST_COPY))
        first = pipeline.predict(bundle, noisy)
        second = pipeline.predict(bundle, noisy)
    assert np.array_equal(first, second)
    assert np.all(first[test.period == 0] == X)
    assert spell_f1(labels, first) > spell_f1(labels, pipeline.baseline(noisy)) + 0.2


def test_boundary_targets_name_the_destination():
    labels = np.array([H] * 10 + [A] * 10 + [D] * 10, dtype=np.int8)
    period = np.ones(30, dtype=np.int8)
    target = pipeline.boundary_targets(labels, period, tolerance=2)
    assert target[8:13].tolist() == [1 + A] * 5
    assert target[18:23].tolist() == [1 + D] * 5
    assert target[:8].sum() == 0 and target[13:18].sum() == 0 and target[23:].sum() == 0


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
