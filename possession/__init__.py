"""Possession by a mixture of experts: four specialists, a learned gate, and a segment decoder.

    tracking -+- carrier model: who is on the ball, even while the tracking has lost it
              |        |
              +- features -+- ball expert --------+
                           |- shape expert -------+- gate: reads every expert over +-10 s
                           |- dead-ball expert ---+    |   P(H/A/D) per frame
                           `- transition expert --+    |   P(change to H/A/D) per frame
                                                       v
                                                    decoder -- submission

Every learned part is trained on OTHER games -- the open IDSSE/DFL Bundesliga matches, which carry
a per-frame ball status and possessing team -- after corrupting their clean tracking to look like
the provided automatic tracking. Nothing here reads a ground truth of the provided game, public or
private; possession/guard.py enforces that. See possession/README.md for the method and the
compliance checklist.

    python -m possession download                 # the open DFL matches, once
    python -m possession profile                  # noise statistics of the provided ATD
    python -m possession train                    # experts + gate + decoder -> models/moe.pkl
    python -m possession predict --check          # the submission, generated twice and compared
    python -m possession evaluate --quick         # leave-one-game-out score on DFL
    python -m possession compare A B              # paired difference between two evaluations
"""
