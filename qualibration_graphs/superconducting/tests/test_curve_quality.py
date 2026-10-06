"""Tests for GEF argmax quality grading (nodes 14a / 14b)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.common_utils.curve_quality import argmax_with_quality  # noqa: E402


def test_argmax_with_quality_accepts_a_clear_interior_peak():
    x = np.linspace(-1.0, 1.0, 21)
    y = np.exp(-((x - 0.1) ** 2) / 0.08) + 0.01 * np.sin(20 * x)
    quality = argmax_with_quality(x, y)
    assert quality.success
    assert quality.interior
    assert quality.prominence_snr >= 5.0


def test_argmax_with_quality_fails_a_flat_dead_curve():
    x = np.linspace(0.0, 1.0, 21)
    y = np.full_like(x, 0.3)
    y[10] += 1e-6
    quality = argmax_with_quality(x, y)
    assert not quality.success
    assert "flat/dead" in quality.note or "piecewise-constant" in quality.note
