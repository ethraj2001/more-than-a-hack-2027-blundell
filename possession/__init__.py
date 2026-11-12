"""Possession by a mixture of experts: five specialists, a learned gate, and a segment decoder.

    tracking -- features -+- ball expert --------+
                          |- shape expert -------+- gate: reads every expert over +-10 s
                          |- dead-ball expert ---+    |   P(H/A/D) per frame
                          |- transition expert --+    |   P(change to H/A/D) per frame
                          `- touch expert -------+    v
                                                   decoder -- submission

Smart Tagging adds a sixth expert, and tag features, only if the provided game's tags pass a
check against its own tracking before training (tagging.check, config.TAGS_CHECK); otherwise the
model is the five above. A carrier model and a short-possession expert also exist; each was
measured and is off -- see possession/README.md.

Every learned part is trained on OTHER games -- the open IDSSE/DFL Bundesliga matches, which carry
a per-frame ball status and possessing team, and Metrica's open sample games, labelled from their
event data (metrica.py) -- after corrupting their tracking to look like the provided automatic
tracking. Nothing here reads a ground truth of the provided game, public or
private; possession/guard.py enforces that. See possession/README.md for the method and the
compliance checklist.

    python -m possession download                 # the open DFL matches, once
    python -m possession profile                  # noise statistics of the provided ATD
    python -m possession train                    # experts + gate + decoder -> models/moe.pkl
    python -m possession predict --check          # the submission, generated twice and compared
    python -m possession evaluate --quick         # leave-one-game-out score on DFL
    python -m possession compare A B              # paired difference between two evaluations
"""
