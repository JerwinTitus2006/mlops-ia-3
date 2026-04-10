import numpy as np
import pandas as pd

from src.drift.detector import _psi_for_column


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 500))
    shifted = pd.Series(rng.normal(1.5, 1, 500))

    score = _psi_for_column(reference, shifted)
    assert score > 0.2


def test_psi_no_shift_low_score() -> None:
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 500))
    current = pd.Series(rng.normal(0, 1, 500))

    score = _psi_for_column(reference, current)
    assert score < 0.2
