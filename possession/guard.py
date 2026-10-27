"""Compliance checks that run inside the pipeline rather than living only in a README.

Two rules from RULES.md are cheap to enforce in code, so they are:

  * No ground truth of the provided game -- public or private -- is ever an input. Every path the
    pipeline opens for reading goes through check_input(), which refuses the ground-truth folder
    and anything named like a ground-truth file.
  * The repository must regenerate the submission exactly. file_digest() is what `predict --check`
    compares across two independent runs.

The rest -- no hand labels, no human data about the game, no cloud model seeing the data -- are
about what people do, and live in the README checklist.
"""

import hashlib
import os

from possession import config

# Params
GT_DIR = os.path.realpath(os.path.join(config.ROOT, "evaluation", "gt"))
GT_MARKERS = ("_possession_gt", "possession_by_frame")


class GroundTruthAccess(RuntimeError):
    """Raised when the pipeline is asked to read a ground truth of the provided game."""


def check_input(path):
    """Return path unchanged if the pipeline may read it, else raise GroundTruthAccess."""
    real = os.path.realpath(path)
    name = os.path.basename(real).lower()
    if real == GT_DIR or real.startswith(GT_DIR + os.sep) or any(m in name for m in GT_MARKERS):
        raise GroundTruthAccess(
            "refusing to read {}: the ground truth of the provided game is not an input to this "
            "pipeline (RULES.md, rule 2). Score with evaluation/run_evaluation.py instead."
            .format(path))
    return path


def file_digest(path):
    """sha256 of a file's bytes, for the reproducibility check."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
