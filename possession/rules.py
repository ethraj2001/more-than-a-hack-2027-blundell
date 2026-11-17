"""The team's explicit possession rules, as evidence on P(H, A, D) just before the decoder.

The model learns possession from labels; these are the two written rules that could be applied on
top of it without lowering the held-out score (config.POSSESSION_RULES):

  duel   both teams have a player within DUEL_M of the ball: the split between H and A moves
         DUEL_WEIGHT of the way to the team whose player is nearer.
  gap    the tracking has lost the ball: it is dead only with slow players (at least SLOW_SHARE of
         them under config.STILL_SPEED) or a line crossing (last seen within LINE_M of a line);
         otherwise P(D) is multiplied by GAP_DEAD.

Together, against the submitted blend on the same held-out output: seven DFL games -0.001 (90%
interval -0.002 to +0.001), better on 3 of 7; Metrica's three -0.002 (-0.006 to +0.002), 1 of 3.
No gain is shown; they are in because the team's definition has them. Measured and left out:
the ball seen more than 0.5 m beyond a line forced dead, -0.002 (-0.003 to -0.000) and -0.005
(-0.008 to -0.003); a takeover needing a run of 2 s, -0.052 and -0.032; the gap rule at 0.1,
-0.004 on Metrica.
"""
import numpy as np

from possession import features

# Params
DUEL_M = 1.5          # metres: both teams this near the ball is a duel
DUEL_WEIGHT = 0.4     # how far the H/A split moves to the nearer team
SLOW_SHARE = 0.5      # share of players under config.STILL_SPEED that counts as slow players
LINE_M = 0.5          # metres from a line at the last sighting: a line crossing
GAP_DEAD = 0.5        # P(D) is multiplied by this in a gap with neither


def apply(probs, tracking):
    """probs (T, 3) with the duel and gap rules applied; rows still sum to one."""
    feature_set = features.compute(tracking)
    column = lambda name: feature_set.column(name).astype(np.float64)  # noqa: E731
    return gap(duel(probs, column("A.d_ball.0"), column("B.d_ball.0")), column("S.ball_vis.0"),
               column("S.all_still.0"), column("S.margin.0"))


def duel(probs, home, away):
    """home, away (T,): each team's nearest player's distance to the ball, NaN where unknown."""
    out = np.array(probs, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        both = (home < DUEL_M) & (away < DUEL_M)
        nearer = ((0, both & (home < away)), (1, both & (away < home)))
    mass = out[:, 0] + out[:, 1]
    for team, rows in nearer:
        share = out[rows, team] / np.maximum(mass[rows], 1e-9)
        share = (1 - DUEL_WEIGHT) * share + DUEL_WEIGHT
        out[rows, team], out[rows, 1 - team] = share * mass[rows], (1 - share) * mass[rows]
    return out


def gap(probs, seen, still, margin):
    """seen (T,): 1 where the ball is seen; still: the share of slow players; margin: the ball's
    distance inside the nearest line, NaN where the ball is lost (its last value is carried)."""
    with np.errstate(invalid="ignore"):
        line = _carried(margin) < LINE_M
        hit = (seen == 0) & ~(still >= SLOW_SHARE) & ~line
    out = np.array(probs, dtype=np.float64)
    out[hit, 2] *= GAP_DEAD
    return out / out.sum(axis=1, keepdims=True)


def _carried(values):
    """values with each NaN replaced by the last value before it."""
    index = np.where(~np.isnan(values), np.arange(len(values)), 0)
    np.maximum.accumulate(index, out=index)
    return values[index]
